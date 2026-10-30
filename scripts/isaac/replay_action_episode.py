"""Replay corrected actions through the production IK, without OpenVLA or ROS.

Terminal: python3 scripts/isaac/replay_action_episode.py --check
Isaac Script Editor: load this file and call start_replay().
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EPISODE = ROOT / "datasets/raw_pose_v2/episode_000001"
CONVENTION = "world_from_local_column_v2"


def load_module(path, name):
    # Kit may retain SourceFileLoader code across edits. Read and compile source
    # explicitly so a running Isaac session cannot reuse the old function API.
    path = Path(path)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def load_execution_kinematics():
    """Extract the controller's pure math definitions without importing ROS."""
    path = ROOT / "scripts/ros/ee_nudge.py"
    source = path.read_text()
    wanted = {"_xyz", "_rpy", "_rotz", "fk", "_rot_log", "_jacobian", "ik"}
    nodes, found = [], set()
    for node in ast.parse(source, filename=str(path)).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_UR10E_ORIGIN"
            for target in node.targets
        ):
            nodes.append(node)
            found.add("_UR10E_ORIGIN")
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes.append(node)
            found.add(node.name)
    if found != wanted | {"_UR10E_ORIGIN"}:
        raise ValueError("Controller kinematics changed; inspect ee_nudge.py before replay")
    namespace = {"np": np, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    namespace["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    return namespace


def rotation_error_deg(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a @ b.T) - 1) / 2, -1, 1))))


def solve_actions(actions, initial_q, kinematics):
    """Use only the initial recorded joints; seed each solve with its own result."""
    actions = np.asarray(actions, dtype=float)
    initial_q = np.asarray(initial_q, dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
        raise ValueError("Expected nonempty (N, 7) actions")
    if initial_q.shape != (6,) or not all(np.isfinite(x).all() for x in (actions, initial_q)):
        raise ValueError("Non-finite actions or invalid initial joints")
    fk, ik, rpy = (kinematics[name] for name in ("fk", "ik", "_rpy"))
    trajectory = [initial_q.copy()]
    position_errors, rotation_errors, jumps = [], [], []
    for index, action in enumerate(actions):
        seed = trajectory[-1]
        before = fk(seed)
        goal = before.copy()
        goal[:3, 3] += before[:3, :3] @ action[:3]
        goal[:3, :3] = rpy(*action[3:6])[:3, :3] @ before[:3, :3]
        # Preserve the original labels: no model scaling or action clipping.
        q = np.asarray(ik(seed.copy(), goal), dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError(f"Action {index}: IK returned invalid joints")
        actual = fk(q)
        position_mm = float(np.linalg.norm(actual[:3, 3] - goal[:3, 3]) * 1000)
        rotation_deg = rotation_error_deg(actual[:3, :3], goal[:3, :3])
        jump_deg = float(np.degrees(np.max(np.abs(q - seed))))
        if position_mm > 2 or rotation_deg > 1 or jump_deg > 20:
            raise ValueError(f"Action {index}: rejected IK, position={position_mm:.4f}mm "
                             f"rotation={rotation_deg:.4f}deg joint_jump={jump_deg:.4f}deg")
        trajectory.append(q.copy())
        position_errors.append(position_mm)
        rotation_errors.append(rotation_deg)
        jumps.append(jump_deg)
    return np.asarray(trajectory), {
        "actions_solved": len(actions), "seed_mode": "initial_joint_only_then_previous_solution",
        "max_ik_position_error_mm": max(position_errors),
        "max_ik_rotation_error_deg": max(rotation_errors),
        "max_joint_step_deg": max(jumps),
    }


def build_trajectory(episode, meta, recorded_q, times, *, kinematics=None):
    if any(meta.get(key) != CONVENTION for key in ("pose_convention", "actions_pose_convention")):
        raise ValueError("Use corrected raw_pose_v2 poses and actions")
    if meta.get("action_frame") != "ee" or meta.get("rotation_representation") != "rpy_left_multiply":
        raise ValueError("Expected ee-frame translation and left-multiplied RPY rotation")
    if meta.get("action_units") != ["m", "m", "m", "rad", "rad", "rad", "binary"]:
        raise ValueError("Unexpected action units")
    actions = np.load(Path(episode) / "actions.npy", allow_pickle=False)
    if actions.shape != (len(recorded_q) - 1, 7) or len(times) != len(recorded_q):
        raise ValueError("Action, joint, and timestamp counts differ")
    if not np.allclose(actions[:, 6], meta["gripper_open_value"]):
        raise ValueError("This push replay requires a constant recorded gripper state")
    kinematics = kinematics if kinematics is not None else load_execution_kinematics()
    q, report = solve_actions(actions, recorded_q[0], kinematics)
    fk = kinematics["fk"]
    reference_positions, reference_rotations = [], []
    # Recorded future joints are used only for diagnostics after solving.
    for planned, reference in zip(q, recorded_q):
        a, b = fk(planned), fk(reference)
        reference_positions.append(float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000))
        reference_rotations.append(rotation_error_deg(a[:3, :3], b[:3, :3]))
    report.update(
        controller_source_sha256=kinematics.get("source_sha256"),
        max_reference_position_error_mm=max(reference_positions),
        final_reference_position_error_mm=reference_positions[-1],
        max_reference_rotation_error_deg=max(reference_rotations),
        final_reference_rotation_error_deg=reference_rotations[-1],
        original_timestamps_preserved=True, action_clipping=False,
        limits={"ik_position_mm": 2, "ik_rotation_deg": 1, "joint_step_deg": 20,
                "reference_position_mm": 5, "reference_rotation_deg": 2},
    )
    if max(reference_positions) > 5 or max(reference_rotations) > 2:
        raise ValueError("Action trajectory diverges from recording: " + json.dumps(report))
    return q, report


def baseline_module():
    # Bypass loader/bytecode caches left by earlier Script Editor runs.
    return load_module(ROOT / "scripts/isaac/replay_recorded_episode.py", "joint_replay_for_actions")


def start_replay():
    baseline = baseline_module()
    baseline.ROOT, baseline.EPISODE = ROOT, EPISODE
    return baseline.start_replay(
        trajectory_builder=build_trajectory, mode="corrected_actions_production_ik",
        report_prefix="replay_actions",
        scope="Corrected actions -> sequential production IK -> original-time simulation playback. "
              "Does not test OpenVLA predictions, ROS transport, or live feedback/timing.",
    )


def start_reset(step=0):
    """Reset the same scene to recorded frame `step` for a separate ROS run; do not play a trajectory."""
    baseline = baseline_module()
    baseline.ROOT, baseline.EPISODE = ROOT, EPISODE
    return baseline.start_replay(
        reset_only=True, reset_step=step, mode="ros_replay_scene_reset", report_prefix="replay_reset",
        scope=f"Reset robot and cube to recorded frame {step}; no trajectory execution.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="CPU planning only; never commands simulation")
    parser.add_argument("--episode", type=Path, default=EPISODE)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.check:
        parser.error("Use --check for a terminal check, or start_replay() inside Isaac Script Editor")
    meta, q, times, _ = baseline_module().load_episode(args.episode)
    _, report = build_trajectory(args.episode, meta, q, times)
    report.update(episode=str(args.episode), passed=True, scope="CPU-only action/IK planning; no simulation motion")
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
