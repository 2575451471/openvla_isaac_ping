#!/usr/bin/env python3
"""Offline: ee_poses_world.npy -> actions.npy matching openvla_closed_loop.

Usage (no Isaac / TensorFlow needed):
  python3 scripts/dataset_raw/make_actions.py datasets/raw_pose_v2/episode_000002
  python3 scripts/dataset_raw/make_actions.py datasets/raw_pose_v2 --all --success-only

--all walks every episode_* directory. A directory without meta.json is skipped.
--success-only keeps only meta.success true, which is the set TFDS will train on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset_raw.pose_math import (  # noqa: E402
    POSE_CONVENTION,
    delta_action_closed_loop,
    pose7_world_to_fk,
)


def _load_meta(ep: Path) -> dict:
    with open(ep / "meta.json", encoding="utf-8") as f:
        return json.load(f)


def skip_reason(ep: Path, success_only: bool) -> str | None:
    """Incomplete folders are left from an aborted recording and cannot be labeled."""
    meta_path = ep / "meta.json"
    if not meta_path.is_file():
        return "no meta.json"
    if not (ep / "ee_poses_world.npy").is_file():
        return "no ee_poses_world.npy"
    if success_only and _load_meta(ep).get("success") is not True:
        return "success is not true"
    return None


def make_actions_for_episode(ep: Path, overwrite: bool = False) -> Path:
    ep = Path(ep)
    out = ep / "actions.npy"
    if out.is_file() and not overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite")

    meta = _load_meta(ep)
    if meta.get("pose_convention") != POSE_CONVENTION:
        raise ValueError(
            f"{ep}: legacy or unknown pose convention. Do not overwrite old labels; "
            "use repair_pose_dataset.py to create a validated dataset copy first."
        )
    poses = np.load(ep / "ee_poses_world.npy")
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"ee_poses_world.npy expected (N,7), got {poses.shape}")
    n = poses.shape[0]
    if n < 2:
        raise ValueError("need >=2 poses")

    action_frame = str(meta.get("action_frame", "ee"))
    gripper = float(meta.get("gripper_open_value", 1.0))
    rot_rep = str(meta.get("rotation_representation", "rpy_left_multiply"))
    if rot_rep != "rpy_left_multiply":
        raise ValueError(
            f"unsupported rotation_representation={rot_rep!r}; "
            "closed_loop expects rpy_left_multiply"
        )

    base_xyz = np.asarray(meta["base_world_xyz"], dtype=float)
    base_quat = np.asarray(meta["base_world_quat_wxyz"], dtype=float)

    fk_poses = [
        pose7_world_to_fk(poses[i], base_xyz, base_quat) for i in range(n)
    ]
    actions = np.zeros((n - 1, 7), dtype=np.float32)
    for t in range(n - 1):
        p0, R0 = fk_poses[t]
        p1, R1 = fk_poses[t + 1]
        actions[t] = delta_action_closed_loop(
            p0, R0, p1, R1, action_frame=action_frame, gripper=gripper
        )

    np.save(out, actions)
    meta["num_actions"] = int(actions.shape[0])
    meta["actions_file"] = "actions.npy"
    meta["actions_generated"] = True
    meta["actions_pose_convention"] = POSE_CONVENTION
    with open(ep / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    norms = np.linalg.norm(actions[:, :3], axis=1)
    print(
        "{}: actions {}  |Δxyz| mean={:.4f} m max={:.4f} m  |Δrpy| max={:.4f} rad".format(
            ep.name,
            actions.shape,
            float(norms.mean()),
            float(norms.max()),
            float(np.linalg.norm(actions[:, 3:6], axis=1).max()),
        )
    )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="episode dir or raw root with --all")
    ap.add_argument("--all", action="store_true", help="process all episode_* under path")
    ap.add_argument(
        "--success-only",
        action="store_true",
        help="with --all, label only episodes whose meta.success is true",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.all:
        eps = sorted(p for p in args.path.glob("episode_*") if p.is_dir())
        if not eps:
            raise SystemExit(f"no episode_* under {args.path}")
        labeled = 0
        for ep in eps:
            reason = skip_reason(ep, args.success_only)
            if reason:
                print(f"skip {ep.name}: {reason}")
                continue
            make_actions_for_episode(ep, overwrite=args.overwrite)
            labeled += 1
        print(f"labeled {labeled}, skipped {len(eps) - labeled}")
        if labeled == 0:
            raise SystemExit("no episodes labeled")
    else:
        make_actions_for_episode(args.path, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
