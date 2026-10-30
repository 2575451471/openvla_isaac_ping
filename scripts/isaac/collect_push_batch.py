"""Batch-collect successful demos: push red cube off the RIGHT side of the table.

Right = world -X (matches redo.py P6→P7 / episode_000001).
Uses isaac.arm_motion for IK/spline/tracking. Failed episodes keep success=false
(TFDS skips them).

Isaac: open assets/sence/sence1_gripper.usd → Play → run this in Script Editor.
"""
from __future__ import annotations

import asyncio
import os
import sys

import numpy as np

# Script Editor copies this file to /tmp/carb.*/script_*.py, so __file__ is useless.
_SCRIPTS = "/home/dt/openvla/shenle_projects/scripts"
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import omni.kit.app
import omni.timeline
from isaacsim.core.prims import SingleArticulation

from isaac.arm_motion import (  # noqa: E402
    EE_PRIM,
    ROBOT_PRIM,
    find_arm_indices,
    get_world_position,
    move_arm_to_q,
    resolve_cube_prim,
    run_waypoint_trajectory,
    set_prim_world_translation,
)

# ------------------------------------------------------------
# Batch knobs
# ------------------------------------------------------------
TARGET_SUCCESSES = int(os.environ.get("SHENLE_TARGET_SUCCESSES", "25"))
MAX_ATTEMPTS = int(os.environ.get("SHENLE_MAX_ATTEMPTS", "60"))
RNG_SEED = 0

ENABLE_RAW_RECORD = True
RECORD_SAMPLE_HZ = 5.0
RECORD_INSTRUCTION = "push the red cube to the right"
RECORD_CUBE_PRIM = None
RECORD_OUT_ROOT = "/home/dt/openvla/shenle_projects/datasets/raw_pose_v2"
RECORD_EPISODE_NAME = os.environ.get("SHENLE_EPISODE_NAME") or None

HOME_DEG = (0.0, -88.4, 80.1, 0.0, 254.2, 0.0)
HOME_SETTLE_FRAMES = 150
PRE_SUCCESS_SETTLE_FRAMES = 90
RESET_SETTLE_FRAMES = 30

CUBE_DEFAULT_XYZ = np.array([0.0, 0.0, 0.040], dtype=float)
CUBE_XY_NOISE = 0.0  # cube stays at the table center
CUBE_Z = 0.040

# P2 is the midpoint of measured P1 and this P3, so it is not an independent point.
# Z is the midpoint of the original P3 (0.77849) and P4 (0.34563).
P3_TEMPLATE = (0.45119, 0.16336, 0.56206)
P4_TEMPLATE = (0.26073, 0.02425, 0.34563)
APPROACH_XY_NOISE = 0.02
APPROACH_Z_NOISE = 0.01

APPROACH_OFFSET_X = 0.26
NEAR_OFFSET_X = 0.015
CONTACT_Z = 0.065
CONTACT_Z_NOISE = 0.01
PUSH_DISTANCE_RANGE = (0.55, 0.90)
LATERAL_NOISE = 0.02

TARGET_WORLD_RPY_DEG = (-74.5, 85.38, 98.53)
WAYPOINT_ORI_ALPHA = {
    "P1": 0.00,
    "P2": 0.33,
    "P3": 0.65,
    "P4": 0.85,
    "P5": 1.00,
    "P6": 1.00,
    "P7": 1.00,
}

# Right edge = world -X. From episode_000001: on-table min x≈-0.87, end≈(-0.93,0,-0.06).
TABLE_X_RIGHT = -0.80
TABLE_Y_HALF = 0.35
Z_OFF_TABLE = 0.03


def _jitter_approach(rng: np.random.Generator, xyz) -> np.ndarray:
    point = np.asarray(xyz, dtype=float)
    point[:2] += rng.uniform(-APPROACH_XY_NOISE, APPROACH_XY_NOISE, size=2)
    point[2] += float(rng.uniform(-APPROACH_Z_NOISE, APPROACH_Z_NOISE))
    return point


def build_waypoints(cube_xy: np.ndarray, rng: np.random.Generator, p1_world):
    """Arm is already at HOME/P1. P2 is the midpoint of that pose and P3."""
    cx, cy = float(cube_xy[0]), float(cube_xy[1])
    p1 = np.asarray(p1_world, dtype=float).reshape(3)
    p3 = _jitter_approach(rng, P3_TEMPLATE)
    p4 = _jitter_approach(rng, P4_TEMPLATE)
    p2 = (p1 + p3) / 2.0
    waypoints = [
        ("P2", float(p2[0]), float(p2[1]), float(p2[2])),
        ("P3", float(p3[0]), float(p3[1]), float(p3[2])),
        ("P4", float(p4[0]), float(p4[1]), float(p4[2])),
    ]

    lat = float(rng.uniform(-LATERAL_NOISE, LATERAL_NOISE))
    z5 = CONTACT_Z + 0.015 + float(rng.uniform(-CONTACT_Z_NOISE, CONTACT_Z_NOISE))
    z67 = CONTACT_Z + float(rng.uniform(-CONTACT_Z_NOISE, CONTACT_Z_NOISE))
    push_d = float(rng.uniform(*PUSH_DISTANCE_RANGE))

    waypoints.append(("P5", cx + APPROACH_OFFSET_X, cy + lat, z5))
    waypoints.append(("P6", cx + NEAR_OFFSET_X, cy + lat, z67))
    waypoints.append(("P7", cx - push_d, cy + lat * 0.5, z67))
    return waypoints


def success_push_off_right(cube_xyz: np.ndarray) -> bool:
    """True iff cube left via the right (-X) side of the table."""
    x, y, z = map(float, cube_xyz[:3])
    past_right = x < TABLE_X_RIGHT
    y_ok = abs(y) <= TABLE_Y_HALF
    off_table = z < Z_OFF_TABLE
    ok = past_right and y_ok and (off_table or x < TABLE_X_RIGHT - 0.05)
    print(
        "[success_check] cube=({:.3f},{:.3f},{:.3f}) past_right={} y_ok={} off_table={} -> {}".format(
            x, y, z, past_right, y_ok, off_table, ok
        )
    )
    return ok


async def reset_episode(robot, arm_idx, cube_prim: str, cube_xyz: np.ndarray, app) -> None:
    print("[reset] cube ->", np.round(cube_xyz, 4))
    set_prim_world_translation(cube_prim, cube_xyz)
    q_home = np.radians(np.asarray(HOME_DEG, dtype=float))
    print("[reset] arm home deg", HOME_DEG)
    await move_arm_to_q(robot, arm_idx, q_home, app=app, frames=HOME_SETTLE_FRAMES)
    for _ in range(RESET_SETTLE_FRAMES):
        await app.next_update_async()


async def run_batch():
    rng = np.random.default_rng(RNG_SEED)
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        raise RuntimeError("Please press Play before running collect_push_batch.py")

    cube_prim = resolve_cube_prim(RECORD_CUBE_PRIM)
    print("[batch] cube prim:", cube_prim)

    robot = SingleArticulation(prim_path=ROBOT_PRIM)
    robot.initialize()
    arm_idx = find_arm_indices(list(robot.dof_names))

    n_ok = 0
    n_try = 0
    print("")
    print("==========================================")
    print("Batch push-off-right collection")
    print("target successes:", TARGET_SUCCESSES, "max attempts:", MAX_ATTEMPTS)
    print("instruction:", RECORD_INSTRUCTION)
    print("==========================================")

    while n_ok < TARGET_SUCCESSES and n_try < MAX_ATTEMPTS:
        n_try += 1
        cube_xy = CUBE_DEFAULT_XYZ[:2] + rng.uniform(-CUBE_XY_NOISE, CUBE_XY_NOISE, size=2)
        cube_xyz = np.array([cube_xy[0], cube_xy[1], CUBE_Z], dtype=float)

        await reset_episode(robot, arm_idx, cube_prim, cube_xyz, app)
        cube_start = get_world_position(cube_prim).copy()
        p1_world = get_world_position(EE_PRIM).copy()
        print("[reset] P1 is HOME EE, not a motion target", np.round(p1_world, 4))
        waypoints = build_waypoints(cube_start[:2], rng, p1_world)

        print("")
        print(
            "----- attempt {}/{}  successes {}/{} -----".format(
                n_try, MAX_ATTEMPTS, n_ok, TARGET_SUCCESSES
            )
        )
        for wp in waypoints:
            print("  ", wp[0], np.round(wp[1:], 4))

        def success_predicate():
            return success_push_off_right(get_world_position(cube_prim))

        result = await run_waypoint_trajectory(
            waypoints,
            WAYPOINT_ORI_ALPHA,
            TARGET_WORLD_RPY_DEG,
            enable_raw_record=ENABLE_RAW_RECORD,
            record_instruction=RECORD_INSTRUCTION,
            record_sample_hz=RECORD_SAMPLE_HZ,
            record_out_root=RECORD_OUT_ROOT,
            record_episode_name=RECORD_EPISODE_NAME,
            record_cube_prim=cube_prim,
            success_predicate=success_predicate,
            pre_success_settle_frames=PRE_SUCCESS_SETTLE_FRAMES,
        )

        cube_end = get_world_position(cube_prim)
        print(
            "[batch] cube start",
            np.round(cube_start, 4),
            "end",
            np.round(cube_end, 4),
            "delta",
            np.round(cube_end - cube_start, 4),
        )

        if result.get("tracking_ok") and result.get("ok"):
            n_ok += 1
            print(
                "[batch] SUCCESS #{}/{}".format(n_ok, TARGET_SUCCESSES),
                result.get("episode_dir"),
            )
        else:
            print(
                "[batch] FAIL attempt",
                n_try,
                "error=",
                result.get("error"),
                "ep=",
                result.get("episode_dir"),
            )

    print("")
    print("==========================================")
    print("Batch done: successes={}/{} attempts={}".format(n_ok, TARGET_SUCCESSES, n_try))
    print("Next: python3 scripts/dataset_raw/make_actions.py datasets/raw_pose_v2 --all")
    print("Then tfds build --overwrite && LoRA finetune")
    print("==========================================")
    return n_ok


async def safe_run():
    try:
        await run_batch()
    except Exception as exc:
        print("")
        print("==========================================")
        print("BATCH STOPPED SAFELY")
        print(str(exc))
        print("==========================================")


if __name__ == "__main__" or os.environ.get("SHENLE_SKIP_COLLECT_AUTORUN") != "1":
    asyncio.ensure_future(safe_run())
