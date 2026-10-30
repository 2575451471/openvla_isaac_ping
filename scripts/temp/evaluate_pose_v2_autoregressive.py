#!/usr/bin/env python3
"""Compare base/LoRA autoregressive actions on fixed pose-v2 recorded images."""

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT.parent))
sys.path.insert(0, str(PROJECT / "scripts/dataset_raw"))


def normalized_reference(action, statistics):
    action = np.asarray(action, dtype=float)
    low, high = np.asarray(statistics["q01"]), np.asarray(statistics["q99"])
    mask = np.asarray(statistics.get("mask", [True] * 7), dtype=bool)
    normalized = np.where(mask, np.clip(2 * (action - low) / (high - low + 1e-8) - 1, -1, 1), action)
    return np.where(mask & (np.asarray(statistics["min"]) == np.asarray(statistics["max"])), 0, normalized)


def summarize(rows):
    if not rows:
        raise ValueError("No predictions to summarize")
    pred = np.asarray([r["prediction"] for r in rows])
    truth = np.asarray([r["reference"] for r in rows])
    pn = np.asarray([r["normalized_prediction"] for r in rows])
    tn = np.asarray([r["normalized_reference"] for r in rows])
    counts = Counter(tuple(r["generated_tokens"]) for r in rows)
    moving = np.linalg.norm(truth[:, :3], axis=1) >= 0.001
    signed = np.sum(pred[:, :3] * truth[:, :3], axis=1)
    return {
        "examples": len(rows), "unique_actions": len(counts),
        "dominant_action_fraction": max(counts.values()) / len(rows),
        "invalid_action_tokens": sum(r["invalid_action_tokens"] for r in rows),
        "normalized_movement_l1": float(np.abs(pn[:, :6] - tn[:, :6]).mean()),
        "xyz_error_norm_mean_mm": float(np.linalg.norm(pred[:, :3] - truth[:, :3], axis=1).mean() * 1000),
        "xyz_mae_mm": (np.abs(pred[:, :3] - truth[:, :3]).mean(0) * 1000).tolist(),
        "rpy_mae_deg": np.rad2deg(np.abs(pred[:, 3:6] - truth[:, 3:6]).mean(0)).tolist(),
        "open_gripper_accuracy": float(np.mean((pred[:, 6] >= 0.5) == (truth[:, 6] >= 0.5))),
        "moving_reference_examples": int(moving.sum()),
        "opposite_translation_direction_fraction": float(np.mean(signed[moving] < 0)) if moving.any() else None,
        "translation_std_mm": (pred[:, :3].std(0) * 1000).tolist(),
    }


def load_plan(run):
    config = json.loads((run / "config.json").read_text())
    best = json.loads((run / "best.json").read_text())
    manifest = json.loads((run / "split_manifest.json").read_text())
    frozen = json.loads((PROJECT / "datasets/pose_v2_preparation/split_manifest.json").read_text())
    if manifest != frozen or config["statistics_split"] != "train":
        raise ValueError("Run does not match the frozen train/val preparation")
    if best.get("adapter") is None:
        raise ValueError("This trial did not improve over the base-model validation baseline")
    adapter = (run / best["adapter"]).resolve()
    if adapter.parent != run.resolve() or not (adapter / "adapter_model.safetensors").is_file():
        raise ValueError("Missing or unexpected adapter directory")
    statistics = json.loads((adapter / "dataset_statistics.json").read_text())
    if statistics["ur10e_push_sim"]["num_trajectories"] != len(manifest["splits"]["train"]):
        raise ValueError("Adapter statistics do not match the training split")
    raw = Path(manifest["raw_root"])
    inventory = {row["episode_id"]: row for row in manifest["episodes"]}
    samples = []
    train_ids = manifest["splits"]["train"]
    selected = {"val": manifest["splits"]["val"], "train": [train_ids[0], train_ids[len(train_ids) // 2]]}
    for split, episodes in selected.items():
        for episode in episodes:
            path = raw / episode
            for name in ("meta.json", "actions.npy"):
                if hashlib.sha256((path / name).read_bytes()).hexdigest() != inventory[episode]["file_hashes"][name]:
                    raise ValueError(f"Source changed: {episode}/{name}")
            metadata = json.loads((path / "meta.json").read_text())
            actions = np.load(path / "actions.npy").astype(np.float32)
            count = 6 if split == "val" else 4
            for index in np.linspace(0, len(actions) - 1, count, dtype=int):
                image = path / "images" / f"{index:06d}.png"
                if not image.is_file():
                    raise ValueError(f"Missing image: {image}")
                samples.append({"split": split, "episode": episode, "index": int(index),
                                "instruction": metadata["instruction"].lower(), "image": str(image),
                                "reference": actions[index].tolist(),
                                "normalized_reference": normalized_reference(actions[index], statistics["ur10e_push_sim"]["action"]).tolist()})
    return config, best, adapter, statistics, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--execute", action="store_true", help="Load the model and perform GPU inference")
    args = parser.parse_args()
    run = args.run
    if run is None:
        runs = sorted((PROJECT / "runs/ft_pose_v2").glob("trial_*"))
        if not runs:
            raise ValueError("No pose-v2 training run found")
        run = runs[-1]
    run = run.resolve()
    config, best, adapter, statistics, samples = load_plan(run)
    print(json.dumps({"run": str(run), "adapter": str(adapter), "best_step": best["step"],
                      "val_images": sum(s["split"] == "val" for s in samples),
                      "train_images": sum(s["split"] == "train" for s in samples),
                      "execute": args.execute}), flush=True)
    if not args.execute:
        return

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    start = time.monotonic()
    from pose_v2_training_utils import busy_gpu_processes
    rows = subprocess.run(["nvidia-smi", "-i", "0", "--query-compute-apps=pid,process_name,used_memory",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.splitlines()
    busy = busy_gpu_processes(rows, os.getpid())
    if busy:
        raise RuntimeError("Other GPU compute jobs are running: " + "; ".join(busy))

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform
    from prismatic.vla.datasets.rlds.obs_transforms import decode_and_resize

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    free, _ = torch.cuda.mem_get_info(0)
    if free < 16000 * 1024**2:
        raise RuntimeError("Need at least 16000 MiB of free GPU memory")
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(config["base_model"], trust_remote_code=True, local_files_only=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    transform = RLDSBatchTransform(action_tokenizer, processor.tokenizer,
                                   processor.image_processor.apply_transform, PurePromptBuilder)
    prepared = []
    for sample in samples:
        raw_image = Image.open(sample["image"]).convert("RGB")
        image224 = decode_and_resize({"image_primary": tf.constant(Path(sample["image"]).read_bytes())},
                                     resize_size=(224, 224), depth_resize_size={})["image_primary"].numpy()
        prompt = f"In: What action should the robot take to {sample['instruction']}?\nOut:"
        training = transform({"dataset_name": b"ur10e_push_sim", "action": np.asarray(sample["normalized_reference"])[None],
                              "observation": {"image_primary": image224[None]},
                              "task": {"language_instruction": sample["instruction"].encode()}})
        train_prefix = training["input_ids"][:-8].tolist()
        for mode, image in (("training_resize", Image.fromarray(image224)), ("deployment_raw", raw_image)):
            inputs = processor(prompt, image)
            prompt_ids = inputs["input_ids"][0].tolist()
            if prompt_ids[-1] != 29871:
                prompt_ids.append(29871)
            if train_prefix != prompt_ids:
                raise ValueError(f"Training/inference prompt mismatch: {train_prefix} != {prompt_ids}")
            if mode == "training_resize" and not torch.equal(inputs["pixel_values"][0], training["pixel_values"]):
                raise ValueError("Training/inference image transform mismatch")
            prepared.append((sample, mode, inputs))
    core = AutoModelForVision2Seq.from_pretrained(
        config["base_model"], torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, local_files_only=True, attn_implementation="sdpa",
    ).to(device)
    model = PeftModel.from_pretrained(core, str(adapter), is_trainable=False)
    model.eval()
    core.norm_stats = statistics
    original_generate = core.generate
    trace = {}

    def traced_generate(*values, **kwargs):
        result = original_generate(*values, **kwargs)
        trace["tokens"] = result[0, -7:].detach().cpu().tolist()
        return result

    core.generate = traced_generate
    records = []
    report_dir = PROJECT / "logs" / datetime.now().strftime("pose_v2_autoregressive_%Y%m%d-%H%M%S-%f")
    report_dir.mkdir(parents=True, exist_ok=False)
    print("REPORT_DIR", report_dir, flush=True)
    for label in ("base", "adapter"):
        context = model.disable_adapter() if label == "base" else nullcontext()
        with context, torch.inference_mode():
            for sample, mode, cpu_inputs in prepared:
                inputs = cpu_inputs.to(device, dtype=torch.bfloat16)
                torch.cuda.synchronize()
                before = time.monotonic()
                prediction = core.predict_action(**inputs, unnorm_key="ur10e_push_sim", do_sample=False)
                torch.cuda.synchronize()
                if np.asarray(prediction).shape != (7,) or not np.isfinite(prediction).all():
                    raise ValueError("Invalid generated action")
                tokens = trace["tokens"]
                index = np.clip(core.vocab_size - np.asarray(tokens) - 1, 0, len(core.bin_centers) - 1)
                normalized = core.bin_centers[index]
                record = dict(sample, model=label, image_mode=mode, prediction=prediction.tolist(),
                              normalized_prediction=normalized.tolist(), generated_tokens=tokens,
                              invalid_action_tokens=sum(not action_tokenizer.action_token_begin_idx < t < core.vocab_size for t in tokens),
                              elapsed_seconds=time.monotonic() - before,
                              image_sha256=hashlib.sha256(Path(sample["image"]).read_bytes()).hexdigest())
                records.append(record)
                with (report_dir / "predictions.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                if len(records) % 16 == 0:
                    print("GENERATED", len(records), "/", len(prepared) * 2, flush=True)
    result = {"run": str(run), "adapter": str(adapter), "best_step": best["step"],
              "mode": "greedy autoregressive generation; no reference action tokens provided",
              "prompt_and_training_image_transform_match": True,
              "normalization_statistics_split": "train", "groups": {},
              "elapsed_seconds": time.monotonic() - start,
              "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
              "scope": "Recorded-image action prediction; not simulation success rate"}
    for label in ("base", "adapter"):
        for mode in ("training_resize", "deployment_raw"):
            for split in ("train", "val"):
                subset = [r for r in records if (r["model"], r["image_mode"], r["split"]) == (label, mode, split)]
                result["groups"][f"{label}/{mode}/{split}"] = summarize(subset)
    (report_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("AUTOREGRESSIVE_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
