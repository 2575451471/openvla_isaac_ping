"""Single hand-tuned push trajectory (P1–P7). Motion/IK live in arm_motion.py.

For batch collection with reset + success gate, use collect_push_batch.py.
"""
import asyncio
import sys

# Script Editor copies this file to /tmp/carb.*/script_*.py, so __file__ is useless.
_SCRIPTS = "/home/dt/openvla/shenle_projects/scripts"
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from isaac.arm_motion import EE_PRIM, get_world_position, run_waypoint_trajectory  # noqa: E402
import numpy as np

# ============================================================
# World-frame EE waypoints (wrist_3_link)
# ============================================================
# P1 is the current pose (reset to HOME first). P2 is the midpoint of P1 and P3.
P3_WORLD = (0.45119, 0.16336, 0.56206)
WAYPOINTS_AFTER_P2 = [
    ("P3", *P3_WORLD),
    ("P4", 0.26073, 0.02425, 0.34563),
    ("P5", 0.26073, 0.02475, 0.080),
    ("P6", 0.01342, 0.02475, 0.065),
    ("P7", -0.68529, 0.02951, 0.065),
]

TARGET_WORLD_RPY_DEG = (-74.5, 85.38, 98.53)

WAYPOINT_ORI_ALPHA = {
    "P2": 0.33,
    "P3": 0.65,
    "P4": 0.85,
    "P5": 1.00,
    "P6": 1.00,
    "P7": 1.00,
}

ENABLE_RAW_RECORD = True
RECORD_SAMPLE_HZ = 5.0
RECORD_INSTRUCTION = "push the red cube to the right"
RECORD_CUBE_PRIM = None
RECORD_OUT_ROOT = "/home/dt/openvla/shenle_projects/datasets/raw_pose_v2"


async def run():
    print("")
    print("==========================================")
    print("UR10e continuous Cartesian trajectory")
    print("HOME/P1 -> P7 via isaac.arm_motion; P2 is the P1-P3 midpoint")
    print("==========================================")
    p1 = get_world_position(EE_PRIM)
    p2 = (p1 + np.asarray(P3_WORLD, dtype=float)) / 2.0
    print("P1", np.round(p1, 5), "P2", np.round(p2, 5))
    waypoints = [("P2", float(p2[0]), float(p2[1]), float(p2[2]))] + WAYPOINTS_AFTER_P2
    result = await run_waypoint_trajectory(
        waypoints,
        WAYPOINT_ORI_ALPHA,
        TARGET_WORLD_RPY_DEG,
        enable_raw_record=ENABLE_RAW_RECORD,
        record_instruction=RECORD_INSTRUCTION,
        record_sample_hz=RECORD_SAMPLE_HZ,
        record_out_root=RECORD_OUT_ROOT,
        record_cube_prim=RECORD_CUBE_PRIM,
    )
    if result.get("ok"):
        print("DONE: P7 reached; episode success=True")
        if result.get("episode_dir"):
            print("Next: python3 scripts/dataset_raw/make_actions.py", result["episode_dir"])
    else:
        print("DONE with failure:", result.get("error"), "tracking_ok=", result.get("tracking_ok"))


async def safe_run():
    try:
        await run()
    except Exception as exc:
        print("")
        print("==========================================")
        print("SCRIPT STOPPED SAFELY")
        print(str(exc))
        print("==========================================")


asyncio.ensure_future(safe_run())
