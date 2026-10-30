"""Finite, disjoint pose-v2 data loading and short-run training helpers."""

import copy
import math
from pathlib import Path

DATASET_NAME = "ur10e_push_sim"
DEFAULT_DATA_ROOT = Path("/home/dt/tensorflow_datasets_pose_v2")
DEFAULT_BASE = Path(
    "/home/dt/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/"
    "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def learning_rate_multiplier(update_index, total_updates, warmup_updates):
    """Learning rate for a zero-based optimizer update, including warmup."""
    if not 0 <= update_index < total_updates or not 0 <= warmup_updates < total_updates:
        raise ValueError("Invalid update or warmup range")
    if update_index < warmup_updates:
        return (update_index + 1) / warmup_updates
    progress = (update_index - warmup_updates) / max(total_updates - warmup_updates - 1, 1)
    return 0.1 + 0.45 * (1 + math.cos(math.pi * progress))


def validation_indices(total, limit):
    """Select fixed examples across the whole finite validation split."""
    if total <= 0 or limit <= 0:
        raise ValueError("Validation sizes must be positive")
    count = min(total, limit)
    if count == 1:
        return [total // 2]
    return [i * (total - 1) // (count - 1) for i in range(count)]


def busy_gpu_processes(rows, own_pid):
    """Ignore desktop display services while preserving checks for compute jobs."""
    display_services = {"gnome-remote-desktop-daemon", "gnome-shell", "Xorg", "Xwayland"}
    busy = []
    for line in rows:
        columns = [column.strip() for column in line.split(",", 2)]
        if not line.strip() or columns[0] == str(own_pid):
            continue
        if len(columns) >= 2 and Path(columns[1]).name in display_services:
            continue
        busy.append(line)
    return busy


def checked_builder(data_root=DEFAULT_DATA_ROOT):
    import json

    import tensorflow_datasets as tfds

    state = PROJECT_ROOT / "datasets/pose_v2_preparation"
    # The build sidecar ties these records to the frozen preparation manifest.
    manifest_path = state / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    builder = tfds.builder(DATASET_NAME, data_dir=str(data_root))
    if str(builder.version) != "2.0.0" or set(builder.info.splits) != {"train", "val"}:
        raise ValueError("Expected pose-v2 TFDS 2.0.0 with explicit train/val splits")
    sidecar = json.loads((Path(builder.data_dir) / "pose_v2_preparation.json").read_text())
    if sidecar.get("pilot") or sidecar["manifest"] != manifest:
        raise ValueError("Formal TFDS manifest mismatch or pilot data selected")
    for split in ("train", "val"):
        expected = len(manifest["splits"][split])
        if builder.info.splits[split].num_examples != expected:
            raise ValueError(f"Unexpected {split} episode count")
    return builder, manifest


def make_pose_v2_datasets(data_root=DEFAULT_DATA_ROOT, resize=(224, 224), seed=20261017):
    """Return finite train/val frames, sharing statistics computed only on train."""
    import tensorflow as tf

    from prismatic.vla.datasets.rlds.dataset import (
        apply_frame_transforms,
        apply_trajectory_transforms,
        make_dataset_from_rlds,
    )
    from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
    from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

    tf.random.set_seed(seed)
    builder, manifest = checked_builder(data_root)
    kwargs_list, _ = get_oxe_dataset_kwargs_and_weights(
        Path(data_root), [(DATASET_NAME, 1.0)], load_camera_views=("primary",),
        load_depth=False, load_proprio=False, load_language=True,
        action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
    )
    kwargs = kwargs_list[0]
    kwargs.pop("dataset_frame_transform_kwargs", None)
    kwargs.update(dataset_statistics_split="train", num_parallel_reads=1, num_parallel_calls=1)
    train, statistics = make_dataset_from_rlds(**copy.deepcopy(kwargs), train=True, shuffle=False)
    val, val_statistics = make_dataset_from_rlds(
        **copy.deepcopy(kwargs), train=False, shuffle=False, dataset_statistics=statistics,
    )
    if int(statistics["num_trajectories"]) != builder.info.splits["train"].num_examples:
        raise ValueError("Normalization statistics do not match the training split")
    for key in ("mean", "std", "q01", "q99"):
        tf.debugging.assert_equal(statistics["action"][key], val_statistics["action"][key])

    def frames(dataset, training):
        dataset = apply_trajectory_transforms(
            dataset, train=training, window_size=1, future_action_window_size=0,
            skip_unlabeled=True, num_parallel_calls=1,
        ).flatten(num_parallel_calls=1)
        return apply_frame_transforms(
            dataset, train=False, resize_size=resize, num_parallel_calls=1,
        ).with_ram_budget(1)

    return frames(train, True), frames(val, False), statistics, manifest
