#!/usr/bin/env python3
"""Repair legacy row-rotation recordings into a separate, verified dataset directory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from dataset_raw.make_actions import make_actions_for_episode
from dataset_raw.pose_math import POSE_CONVENTION, conjugate_quaternion
from dataset_raw.verify_pose_dataset import verify_dataset


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repair_pose_array(poses, *, allow_missing=False):
    result = np.asarray(poses, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 7:
        raise ValueError("Expected (N, 7) xyz+wxyz poses")
    for row in result:
        if allow_missing and np.isnan(row).all():
            continue
        if not np.isfinite(row).all():
            raise ValueError("Non-finite pose")
        row[3:] = conjugate_quaternion(row[3:])
    return result


def repair_dataset(source, destination, *, validator=verify_dataset):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Destination exists; refusing to overwrite: {destination}")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate directories")
    selected = []
    skipped = []
    for path in sorted(source.glob("episode_*/meta.json")):
        meta = json.loads(path.read_text())
        if meta.get("success") is not True:
            skipped.append(path.parent.name)
            continue
        if "pose_convention" in meta:
            raise ValueError(f"{path}: not an unversioned legacy recording; refusing double correction")
        if meta.get("quaternion_order") != "wxyz":
            raise ValueError(f"{path}: unknown quaternion order")
        selected.append((path.parent, meta))
    if not selected:
        raise ValueError("No successful legacy recordings to repair")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + destination.name + ".repair-", dir=destination.parent))
    original_hashes = {}
    try:
        for ep, original_meta in selected:
            target = staging / ep.name
            target.mkdir()
            for path in ep.iterdir():
                if path.name == "images":
                    if not path.is_dir():
                        raise ValueError(f"Missing image directory: {path}")
                    # Immutable image inputs are shared; all metadata/arrays are independent copies.
                    (target / "images").symlink_to(path.resolve(), target_is_directory=True)
                elif path.is_file():
                    original_hashes[str(path.relative_to(source))] = file_hash(path)
                    if path.name not in ("meta.json", "actions.npy", "ee_poses_world.npy", "cube_poses_world.npy"):
                        shutil.copy2(path, target / path.name)
                else:
                    raise ValueError(f"Unexpected episode entry: {path}")
            ee = np.load(ep / "ee_poses_world.npy", allow_pickle=False)
            np.save(target / "ee_poses_world.npy", repair_pose_array(ee))
            cube = ep / "cube_poses_world.npy"
            if cube.exists():
                np.save(target / cube.name, repair_pose_array(np.load(cube, allow_pickle=False), allow_missing=True))
            meta = dict(original_meta)
            meta["base_world_quat_wxyz"] = conjugate_quaternion(meta["base_world_quat_wxyz"]).tolist()
            meta["pose_convention"] = POSE_CONVENTION
            meta["actions_generated"] = False
            meta["actions_file"] = None
            meta["pose_repair"] = {"source_episode": str(ep), "method": "conjugate_legacy_row_rotation_wxyz",
                                   "repaired_utc": datetime.now(timezone.utc).isoformat()}
            (target / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
            make_actions_for_episode(target)
        report = validator(staging)
        if not report["passed"]:
            raise ValueError("Repaired data failed execution-FK checks: " + json.dumps(report))
        for name, expected in original_hashes.items():
            if file_hash(source / name) != expected:
                raise ValueError(f"Source file changed during repair: {source / name}")
        report["dataset"] = str(destination)
        (staging / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
        manifest = {"source": str(source), "destination": str(destination),
                    "pose_convention": POSE_CONVENTION, "episodes_repaired": len(selected),
                    "skipped_unsuccessful_episodes": skipped, "source_sha256": original_hashes,
                    "images": "Symlinks to untouched source image directories; keep the source dataset."}
        (staging / "repair_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if destination.exists():
            raise ValueError("Destination appeared during repair; refusing to replace it")
        staging.rename(destination)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    report = repair_dataset(args.source, args.destination)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
