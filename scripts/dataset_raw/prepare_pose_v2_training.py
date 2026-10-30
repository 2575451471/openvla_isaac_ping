#!/usr/bin/env python3
"""CPU-only training preflight; does not load the 7B model or start training."""

import json
import os
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT.parent))


def main():
    start = time.monotonic()
    import numpy as np
    import tensorflow as tf
    from transformers import AutoProcessor

    from pose_v2_training_utils import (
        DATASET_NAME, DEFAULT_BASE, make_pose_v2_datasets, validation_indices,
    )
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform
    from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

    tf.config.set_visible_devices([], "GPU")
    train, val, statistics, manifest = make_pose_v2_datasets()
    raw = Path(manifest["raw_root"])
    expected_train_actions = np.concatenate([
        np.load(raw / episode / "actions.npy").astype(np.float32)
        for episode in manifest["splits"]["train"]
    ])
    for key, expected in {
        "mean": expected_train_actions.mean(0), "std": expected_train_actions.std(0),
        "min": expected_train_actions.min(0), "max": expected_train_actions.max(0),
        "q01": np.quantile(expected_train_actions, 0.01, axis=0),
        "q99": np.quantile(expected_train_actions, 0.99, axis=0),
    }.items():
        np.testing.assert_allclose(statistics["action"][key], expected, rtol=1e-5, atol=1e-8)
    processor = AutoProcessor.from_pretrained(str(DEFAULT_BASE), trust_remote_code=True, local_files_only=True)
    tokenizer = ActionTokenizer(processor.tokenizer)
    transform = RLDSBatchTransform(
        tokenizer, processor.tokenizer, processor.image_processor.apply_transform, PurePromptBuilder,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right",
    )
    counts = {}
    for split, dataset in (("train", train), ("val", val)):
        count, open_count, minimum, maximum = 0, 0, 1.0, -1.0
        first = None
        for frame in dataset.as_numpy_iterator():
            action = frame["action"][0]
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError(f"Invalid normalized action in {split}")
            if action[-1] != 1.0:
                raise ValueError(f"Open gripper label was changed in {split}: {action[-1]}")
            if frame["observation"]["image_primary"].shape != (1, 224, 224, 3):
                raise ValueError("Unexpected image layout")
            if not frame["task"]["language_instruction"]:
                raise ValueError("Missing language instruction")
            count += 1
            open_count += int(action[-1] == 1.0)
            minimum = min(minimum, float(action[:6].min()))
            maximum = max(maximum, float(action[:6].max()))
            if first is None:
                example = transform(frame)
                collator([example])
                labels = example["labels"].numpy()
                action_labels = labels[labels > tokenizer.action_token_begin_idx]
                if len(action_labels) != 7:
                    raise ValueError(f"Expected 7 action token labels, got {len(action_labels)}")
                first = {"sequence_tokens": len(labels), "action_tokens": len(action_labels)}
        counts[split] = {"actions": count, "open_gripper_actions": open_count,
                         "normalized_movement_min": minimum, "normalized_movement_max": maximum,
                         "tokenization": first}
        if count != manifest["counts"][split]["actions"]:
            raise ValueError(f"Unexpected {split} action count")
    if counts["train"]["actions"] != int(statistics["num_transitions"]):
        raise ValueError("Train frame count differs from train-only normalization statistics")
    state = PROJECT / "datasets/pose_v2_preparation/training"
    state.mkdir(parents=True, exist_ok=True)
    save_dataset_statistics({DATASET_NAME: statistics}, state)
    report = {
        "passed": True, "statistics_split": "train",
        "statistics_episodes": int(statistics["num_trajectories"]),
        "statistics_actions": int(statistics["num_transitions"]), "counts": counts,
        "validation_probe_indices": validation_indices(counts["val"]["actions"], 64),
        "manifest": manifest, "gpu_visible_devices": [str(d) for d in tf.config.get_visible_devices("GPU")],
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "elapsed_seconds": time.monotonic() - start,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    (state / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "manifest"}, indent=2))
    print("TRAINING_PREFLIGHT_PASSED", flush=True)


if __name__ == "__main__":
    main()
