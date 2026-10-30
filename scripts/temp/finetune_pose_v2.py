#!/usr/bin/env python3
"""Single-GPU pose-v2 trial with train-only statistics and fixed validation."""

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT.parent))
sys.path.insert(0, str(PROJECT / "scripts/dataset_raw"))

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])

from pose_v2_training_utils import (
    DATASET_NAME, DEFAULT_BASE, DEFAULT_DATA_ROOT, busy_gpu_processes, learning_rate_multiplier,
    make_pose_v2_datasets, validation_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_steps", type=int, default=200, help="Number of optimizer updates")
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--grad_accumulation_steps", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--eval_examples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--base_model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--run_root", type=Path, default=PROJECT / "runs/ft_pose_v2")
    parser.add_argument("--use_quantization", action="store_true",
                        help="4-bit NF4 LoRA training (bitsandbytes)")
    args = parser.parse_args()
    if not 0 <= args.warmup_steps < args.max_steps:
        parser.error("warmup_steps must be smaller than max_steps")
    if min(args.max_steps, args.grad_accumulation_steps, args.eval_steps, args.eval_examples) <= 0:
        parser.error("Step counts must be positive")
    if args.learning_rate <= 0:
        parser.error("learning_rate must be positive")
    return args


def main():
    args = parse_args()
    import numpy as np
    import torch
    import tensorflow as tf
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform
    from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

    tf.config.set_visible_devices([], "GPU")
    preflight_path = PROJECT / "datasets/pose_v2_preparation/training/preflight.json"
    preflight = json.loads(preflight_path.read_text())
    if not preflight.get("passed") or preflight.get("statistics_split") != "train":
        raise RuntimeError("Run prepare_pose_v2_training.py first")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")
    processes = subprocess.run(
        ["nvidia-smi", "-i", "0", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    competing = busy_gpu_processes(processes, os.getpid())
    if competing:
        raise RuntimeError("GPU compute process already running; wait before starting this trial: " + "; ".join(competing))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    need_mib = 10000 if args.use_quantization else 18000
    if free_bytes < need_mib * 1024**2:
        raise RuntimeError(f"Need at least {need_mib} MiB free GPU memory; close Isaac/model service first")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_frames, val_frames, statistics, manifest = make_pose_v2_datasets(args.data_root, seed=args.seed)
    if manifest != preflight["manifest"]:
        raise RuntimeError("Dataset changed since training preflight")
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(str(args.base_model), trust_remote_code=True, local_files_only=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    transform = RLDSBatchTransform(
        action_tokenizer, processor.tokenizer, processor.image_processor.apply_transform, PurePromptBuilder,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right",
    )
    indices = set(validation_indices(manifest["counts"]["val"]["actions"], args.eval_examples))
    # Keep only these 64 transformed examples; iterate the finite validation split once.
    validation = [transform(frame) for index, frame in enumerate(val_frames.as_numpy_iterator()) if index in indices]
    if len(validation) != len(indices):
        raise RuntimeError("Validation split was truncated")

    run = args.run_root / datetime.now().strftime("trial_%Y%m%d-%H%M%S-%f")
    run.mkdir(parents=True, exist_ok=False)
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    configuration.update(batch_size=1, lora_rank=32, max_grad_norm=1.0, statistics_split="train",
                         shuffle_buffer_size=manifest["counts"]["train"]["actions"], image_aug=False,
                         validation_indices=sorted(indices), source_fingerprint=manifest["source_fingerprint"],
                         inference_requires_adapter_loading_or_merging=True)
    (run / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    (run / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save_dataset_statistics({DATASET_NAME: statistics}, run)
    processor.save_pretrained(run)
    print(f"RUN_DIR {run}", flush=True)

    quantization_config = None
    if args.use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
        )
        import accelerate.big_modeling as _bm
        _orig_dispatch = _bm.dispatch_model

        def _dispatch_skip_to_for_bnb(model, *args, **kwargs):
            if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
                return model
            return _orig_dispatch(model, *args, **kwargs)

        _bm.dispatch_model = _dispatch_skip_to_for_bnb
    model = AutoModelForVision2Seq.from_pretrained(
        str(args.base_model), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, local_files_only=True,
        quantization_config=quantization_config,
        device_map={"": 0} if args.use_quantization else None,
    )
    if args.use_quantization:
        model = prepare_model_for_kbit_training(model)
    else:
        model = model.to(device)
    if tuple(model.config.image_sizes) != (224, 224):
        raise RuntimeError("Preflight image size differs from this model")
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear", init_lora_weights="gaussian",
    ))
    model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    def log(record):
        record["time"] = datetime.now().isoformat()
        with (run / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def forward(example):
        batch = collator([example])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(device, dtype=torch.bfloat16),
                labels=batch["labels"].to(device),
            )
        if not torch.isfinite(output.loss):
            raise RuntimeError("Non-finite loss; stopping before updating weights")
        return output, batch["labels"]

    def metrics(output, labels):
        prefix = output.logits.shape[1] - labels.shape[1]
        prediction = output.logits.detach()[:, prefix:-1].argmax(-1)
        target = labels[:, 1:].to(prediction.device)
        if prediction.shape != target.shape:
            raise RuntimeError("Action-logit alignment mismatch")
        mask = target > action_tokenizer.action_token_begin_idx
        count = int(mask.sum())
        if count != 7:
            raise RuntimeError(f"Expected 7 supervised action tokens, got {count}")
        pred_action = action_tokenizer.decode_token_ids_to_actions(prediction[mask].cpu().numpy())
        gt_action = action_tokenizer.decode_token_ids_to_actions(target[mask].cpu().numpy())
        return {"loss": float(output.loss.detach()),
                "teacher_forced_action_accuracy": float((prediction[mask] == target[mask]).float().mean()),
                "teacher_forced_normalized_l1": float(np.abs(pred_action - gt_action).mean())}

    @torch.inference_mode()
    def evaluate(step):
        model.eval()
        values = []
        for example in validation:
            output, labels = forward(example)
            values.append(metrics(output, labels))
        result = {key: float(np.mean([row[key] for row in values])) for key in values[0]}
        log(dict(phase="val", step=step, examples=len(validation), **result))
        model.train()
        return result

    best_loss = evaluate(0)["loss"]
    baseline_loss = best_loss
    best_step = 0
    (run / "best.json").write_text(json.dumps({"step": 0, "loss": best_loss, "adapter": None}) + "\n")
    train_iterator = iter(train_frames.shuffle(
        manifest["counts"]["train"]["actions"], seed=args.seed, reshuffle_each_iteration=True,
    ).repeat().as_numpy_iterator())
    model.train()
    for update in range(args.max_steps):
        lr = args.learning_rate * learning_rate_multiplier(update, args.max_steps, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        values = []
        for _ in range(args.grad_accumulation_steps):
            example = transform(next(train_iterator))
            output, labels = forward(example)
            values.append(metrics(output, labels))
            (output.loss / args.grad_accumulation_steps).backward()
            del output
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        step = update + 1
        record = {key: float(np.mean([row[key] for row in values])) for key in values[0]}
        log(dict(phase="train", step=step, learning_rate=lr, gradient_norm_before_clip=float(norm), **record))
        if step % args.eval_steps == 0 or step == args.max_steps:
            result = evaluate(step)
            checkpoint = run / f"adapter_step_{step:06d}"
            model.save_pretrained(checkpoint)
            processor.save_pretrained(checkpoint)
            save_dataset_statistics({DATASET_NAME: statistics}, checkpoint)
            # Small trial: keep adapters plus optimizer state; defer expensive full-model merging.
            torch.save({"optimizer": optimizer.state_dict(), "step": step}, checkpoint / "optimizer.pt")
            if result["loss"] < best_loss:
                best_loss, best_step = result["loss"], step
                (run / "best.json").write_text(json.dumps({
                    "step": step, "loss": best_loss, "adapter": checkpoint.name,
                }, indent=2) + "\n")
    log(dict(phase="complete", step=args.max_steps, baseline_val_loss=baseline_loss,
             best_val_loss=best_loss, best_step=best_step,
             peak_gpu_memory_mib=torch.cuda.max_memory_allocated(device) / 1024**2))
    print("POSE_V2_TRIAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
