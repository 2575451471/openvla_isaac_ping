"""Validate corrected episodes and freeze a reproducible episode-level split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import numpy as np

CONVENTION = "world_from_local_column_v2"
VERSION = "2.0.0"
SPLIT_SEED = "ur10e-pose-v2-episode-split-20261017"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_episode_inputs(episode):
    episode = Path(episode)
    meta = json.loads((episode / "meta.json").read_text())
    if meta.get("success") is not True:
        raise ValueError(f"{episode}: expected a successful curated episode")
    if any(meta.get(key) != CONVENTION for key in ("pose_convention", "actions_pose_convention")):
        raise ValueError(f"{episode}: uncorrected pose/action convention")
    if meta.get("action_frame") != "ee" or meta.get("rotation_representation") != "rpy_left_multiply":
        raise ValueError(f"{episode}: incompatible action frame/rotation")
    if (meta.get("action_units") != ["m", "m", "m", "rad", "rad", "rad", "binary"]
            or meta.get("quaternion_order") != "wxyz"):
        raise ValueError(f"{episode}: incompatible units/quaternion order")
    if meta.get("episode_id") != episode.name or meta.get("image_resolution") != [1280, 720]:
        raise ValueError(f"{episode}: inconsistent episode ID or image resolution")
    arrays = {name: np.load(episode / (name + ".npy"), allow_pickle=False) for name in
              ("actions", "ee_poses_world", "joint_positions", "timestamps")}
    actions, poses, joints, times = (arrays[name] for name in arrays)
    n = len(poses)
    if n < 2 or poses.shape != (n, 7) or actions.shape != (n - 1, 7) or joints.shape != (n, 6) or times.shape != (n,):
        raise ValueError(f"{episode}: inconsistent array shapes")
    if not all(np.isfinite(value).all() for value in arrays.values()) or not np.all(np.diff(times) > 0):
        raise ValueError(f"{episode}: non-finite data or invalid timestamps")
    if meta.get("num_observations") != n or meta.get("num_actions") != n - 1:
        raise ValueError(f"{episode}: metadata counts disagree with arrays")
    if not np.allclose(actions[:, 6], meta["gripper_open_value"]):
        raise ValueError(f"{episode}: inconsistent gripper labels")
    return meta, actions.astype(np.float32), poses.astype(np.float32)


def inventory(raw_root):
    raw_root = Path(raw_root).resolve()
    episodes = sorted(p for p in raw_root.glob("episode_*") if p.is_dir())
    if len(episodes) < 2:
        raise ValueError("At least two corrected episodes are needed")
    entries = []
    for episode in episodes:
        meta, actions, poses = load_episode_inputs(episode)
        images = []
        for index in range(len(poses)):
            path = episode / "images" / f"{index:06d}.png"
            with path.open("rb") as stream:
                header = stream.read(24)
            if (len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n"
                    or header[12:16] != b"IHDR" or struct.unpack(">II", header[16:24]) != (1280, 720)):
                raise ValueError(f"{path}: invalid PNG header or dimensions")
            stat = path.stat()
            images.append([path.name, stat.st_size, stat.st_mtime_ns])
        hashes = {name: sha256_file(episode / name) for name in (
            "meta.json", "actions.npy", "ee_poses_world.npy", "joint_positions.npy", "timestamps.npy")}
        entries.append({"episode_id": episode.name, "observations": len(poses), "actions": len(actions),
                        "image_bytes": sum(row[1] for row in images), "file_hashes": hashes,
                        "image_stat_fingerprint": hashlib.sha256(json.dumps(images).encode()).hexdigest()})
    fingerprint = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return raw_root, entries, fingerprint


def select_splits(episode_ids, validation_count=4):
    ids = sorted(episode_ids)
    if len(ids) != len(set(ids)) or not 0 < validation_count < len(ids):
        raise ValueError("Require unique episodes and a nonempty train/val split")
    ranked = sorted(ids, key=lambda value: hashlib.sha256((SPLIT_SEED + ":" + value).encode()).hexdigest())
    val = set(ranked[:validation_count])
    return {"train": [value for value in ids if value not in val], "val": sorted(val)}


def make_manifest(raw_root, validation_count=4):
    root, entries, fingerprint = inventory(raw_root)
    splits = select_splits([entry["episode_id"] for entry in entries], validation_count)
    by_id = {entry["episode_id"]: entry for entry in entries}
    counts = {split: {"episodes": len(ids), "actions": sum(by_id[value]["actions"] for value in ids)}
              for split, ids in splits.items()}
    return {"schema_version": 1, "dataset_name": "ur10e_push_sim", "dataset_version": VERSION,
            "pose_convention": CONVENTION, "raw_root": str(root), "split_seed": SPLIT_SEED,
            "source_fingerprint": fingerprint, "splits": splits, "counts": counts, "episodes": entries}


def validate_manifest(manifest, raw_root, *, check_source=True):
    if manifest.get("schema_version") != 1 or manifest.get("dataset_version") != VERSION:
        raise ValueError("Unsupported split manifest/version")
    if manifest.get("pose_convention") != CONVENTION or manifest.get("raw_root") != str(Path(raw_root).resolve()):
        raise ValueError("Split manifest refers to another source/convention")
    splits = manifest.get("splits", {})
    if set(splits) != {"train", "val"} or not all(splits.values()):
        raise ValueError("Expected nonempty train and val splits")
    combined = splits["train"] + splits["val"]
    expected = [entry["episode_id"] for entry in manifest["episodes"]]
    if len(combined) != len(set(combined)) or sorted(combined) != sorted(expected):
        raise ValueError("Episode overlap, duplication, or missing membership")
    expected_counts = {split: {"episodes": len(ids), "actions": sum(
        entry["actions"] for entry in manifest["episodes"] if entry["episode_id"] in ids)} for split, ids in splits.items()}
    if manifest.get("counts") != expected_counts:
        raise ValueError("Split counts disagree with membership")
    if check_source:
        _, entries, fingerprint = inventory(raw_root)
        if entries != manifest["episodes"] or fingerprint != manifest["source_fingerprint"]:
            raise ValueError("Source changed after split was frozen; use a new preparation directory")


def freeze_manifest(raw_root, path, validation_count=4):
    path = Path(path)
    if path.exists():
        manifest = json.loads(path.read_text())
        validate_manifest(manifest, raw_root)
        if len(manifest["splits"]["val"]) != validation_count:
            raise ValueError("Existing manifest has another validation count")
        return manifest
    manifest = make_manifest(raw_root, validation_count)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest
