#!/usr/bin/env python3
"""Check saved poses/actions against the execution controller's FK, without ROS or motion."""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from dataset_raw.pose_math import POSE_CONVENTION, pose7_world_to_fk, rpy_to_matrix


def load_execution_fk():
    """Load only pure FK definitions from ee_nudge; never import/start a ROS node."""
    path = SCRIPTS / "ros/ee_nudge.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    wanted = {"_xyz", "_rpy", "_rotz", "fk"}
    nodes = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_UR10E_ORIGIN" for target in node.targets
        ):
            nodes.append(node)
            found.add("_UR10E_ORIGIN")
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes.append(node)
            found.add(node.name)
    if found != wanted | {"_UR10E_ORIGIN"}:
        raise ValueError("Execution FK definitions changed; inspect ee_nudge.py before validating")
    namespace = {"np": np, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["fk"]


def rotation_error_deg(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1, 1))))


def verify_dataset(root, *, require_convention=True, reference_fk=None):
    root = Path(root)
    fk = reference_fk or load_execution_fk()
    errors = {key: [] for key in (
        "pose_position_mm", "pose_rotation_deg", "action_position_mm", "action_rotation_deg"
    )}
    episodes = actions_count = 0
    for path in sorted(root.glob("episode_*/meta.json")):
        meta = json.loads(path.read_text())
        if meta.get("success") is not True:
            continue
        ep = path.parent
        if require_convention and (
            meta.get("pose_convention") != POSE_CONVENTION
            or meta.get("actions_pose_convention") != POSE_CONVENTION
        ):
            raise ValueError(f"{ep}: unverified pose/action convention")
        q = np.load(ep / "joint_positions.npy", allow_pickle=False)
        poses = np.load(ep / "ee_poses_world.npy", allow_pickle=False)
        actions = np.load(ep / "actions.npy", allow_pickle=False)
        timestamps = np.load(ep / "timestamps.npy", allow_pickle=False)
        count = len(poses)
        if count < 2 or poses.shape != (count, 7) or q.shape != (count, 6):
            raise ValueError(f"{ep}: pose/joint shapes do not match")
        if actions.shape != (count - 1, 7) or timestamps.shape != (count,):
            raise ValueError(f"{ep}: action/timestamp shapes do not match")
        if not all(np.isfinite(array).all() for array in (q, poses, actions, timestamps)):
            raise ValueError(f"{ep}: non-finite data")
        if not np.all(np.diff(timestamps) > 0):
            raise ValueError(f"{ep}: timestamps are not strictly increasing")
        if meta.get("action_frame") not in ("ee", "base"):
            raise ValueError(f"{ep}: unsupported action frame")
        if meta.get("rotation_representation") != "rpy_left_multiply":
            raise ValueError(f"{ep}: unsupported rotation representation")
        if not np.allclose(actions[:, 6], meta["gripper_open_value"]):
            raise ValueError(f"{ep}: inconsistent gripper labels")
        transforms = [fk(row) for row in q]
        for index, (pose, actual) in enumerate(zip(poses, transforms)):
            position, rotation = pose7_world_to_fk(
                pose, meta["base_world_xyz"], meta["base_world_quat_wxyz"]
            )
            errors["pose_position_mm"].append(float(np.linalg.norm(position - actual[:3, 3]) * 1000))
            errors["pose_rotation_deg"].append(rotation_error_deg(rotation @ actual[:3, :3].T))
            if index == count - 1:
                continue
            image = ep / "images" / f"{index:06d}.png"
            if not image.is_file():
                raise ValueError(f"Missing image: {image}")
            action, following = actions[index], transforms[index + 1]
            delta = actual[:3, :3] @ action[:3] if meta["action_frame"] == "ee" else action[:3]
            predicted_position = actual[:3, 3] + delta
            predicted_rotation = rpy_to_matrix(*action[3:6]) @ actual[:3, :3]
            errors["action_position_mm"].append(float(np.linalg.norm(predicted_position - following[:3, 3]) * 1000))
            errors["action_rotation_deg"].append(rotation_error_deg(predicted_rotation @ following[:3, :3].T))
        episodes += 1
        actions_count += len(actions)
    if not episodes:
        raise ValueError(f"No successful episodes under {root}")
    metrics = {key: {
        "mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)), "n": len(values),
    } for key, values in errors.items()}
    limits = {"pose_position_mm": 0.01, "pose_rotation_deg": 0.01,
              "action_position_mm": 0.01, "action_rotation_deg": 0.01}
    passed = all(metrics[key]["max"] <= limit for key, limit in limits.items())
    return {"dataset": str(root.resolve()), "episodes": episodes, "actions": actions_count,
            "passed": passed, "metrics": metrics, "max_error_limits": limits,
            "scope": "Offline FK/label consistency; does not test IK, physics, camera timing, or model quality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--allow-legacy", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_dataset(args.root, require_convention=not args.allow_legacy)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.write_text(text + "\n")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
