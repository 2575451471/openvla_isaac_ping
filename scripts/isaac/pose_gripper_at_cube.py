"""Move wrist_3 beside the red cube. Run in Script Editor while Play is on.

STANDOFF_X: wrist distance on the cube's +X side, meters.
Z_LIFT: wrist height relative to the cube center.
WRIST_3_DEG: extra spin of the flange joint. 80 leveled the fingers.
"""
import asyncio
import math
import sys

import numpy as np

_SCRIPTS = "/home/dt/openvla/shenle_projects/scripts"
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import omni.kit.app
import omni.timeline
from isaacsim.core.prims import SingleArticulation

from isaac.arm_motion import (
    BASE_PRIM,
    EE_PRIM,
    ROBOT_PRIM,
    find_arm_indices,
    get_world_matrix,
    get_world_position,
    get_world_rotation,
    ik_position_with_scheduled_orientation,
    isaac_base_to_fk,
    isaac_rot_to_fk,
    matrix_to_rpy_deg,
    move_arm_to_q,
    read_arm_q,
    world_to_isaac_base,
)

CUBE = "/World/RedCube"
STANDOFF_X = 0.16
Z_LIFT = 0.04
WRIST_3_DEG = 80.0


def level_push_rotation():
    """Wrist +Z points along world -X, with the side axis horizontal."""
    return np.array(
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=float,
    )


def roll_about_push_deg(rotation):
    """Tilt of the wrist X axis away from horizontal, around world X."""
    axis = np.asarray(rotation, dtype=float)[:, 0]
    return math.degrees(math.atan2(-axis[2], axis[1]))


async def pose():
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        raise RuntimeError("Press Play first, then run this script.")

    cube = get_world_position(CUBE)
    goal = np.array(
        [cube[0] + STANDOFF_X, cube[1], cube[2] + Z_LIFT],
        dtype=float,
    )
    rotation = level_push_rotation()
    base_rotation = get_world_rotation(BASE_PRIM)
    base_matrix = get_world_matrix(BASE_PRIM)
    goal_fk = isaac_base_to_fk(world_to_isaac_base(goal, base_matrix))
    rotation_fk = isaac_rot_to_fk(base_rotation.T @ rotation)

    robot = SingleArticulation(prim_path=ROBOT_PRIM)
    robot.initialize()
    arm_idx = find_arm_indices(list(robot.dof_names))
    seed = read_arm_q(robot, arm_idx)
    joints, pos_err, ori_err = ik_position_with_scheduled_orientation(
        seed, goal_fk, rotation_fk, 1.5
    )
    joints = np.asarray(joints, dtype=float).copy()
    joints[5] += math.radians(WRIST_3_DEG)
    await move_arm_to_q(robot, arm_idx, joints, app=omni.kit.app.get_app(), frames=180)

    reached = get_world_position(EE_PRIM)
    reached_error_mm = float(np.linalg.norm(reached - goal) * 1000)
    reached_rotation = get_world_rotation(EE_PRIM)
    approach = reached_rotation[:, 2]
    roll_deg = roll_about_push_deg(reached_rotation)
    print("cube", np.round(cube, 4))
    print("wrist goal", np.round(goal, 4), "reached", np.round(reached, 4))
    print("IK pos mm", round(pos_err * 1000, 2), "ori deg", round(math.degrees(ori_err), 2))
    print("measured error mm", round(reached_error_mm, 2))
    print("approach world", np.round(approach, 3), "roll about X deg", round(roll_deg, 1))
    print("joints deg", np.round(np.degrees(joints), 2))
    print("wrist rpy deg", np.round(matrix_to_rpy_deg(reached_rotation), 2))
    pointing_x = float(-approach[0])
    if reached_error_mm > 8.0 or abs(roll_deg) > 8.0 or pointing_x < 0.95:
        print("POSE_TILTED: do not use this pose for a straight push")
    else:
        print("POSE_LEVEL")


async def safe_pose():
    try:
        await pose()
    except Exception as exc:
        print("POSE FAILED:", exc)


asyncio.ensure_future(safe_pose())
