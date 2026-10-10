"""Pose math shared by offline action labeling (matches openvla_closed_loop + redo)."""

from __future__ import annotations

import math

import numpy as np

# Same as redo.py: Isaac base <-> custom UR FK differs by Rz(180).
R_ISAAC_TO_FK = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("zero quaternion")
    return q / n


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return normalize_quat_wxyz(np.array([w, x, y, z], dtype=float))


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Same ZYX convention as ee_nudge / openvla_closed_loop `_rpy`."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """Inverse of rpy_to_matrix (roll, pitch, yaw) in radians."""
    R = np.asarray(R, dtype=float)
    if abs(R[2, 0]) < 1.0 - 1e-8:
        pitch = -math.asin(R[2, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        if R[2, 0] <= -1.0:
            pitch = math.pi / 2.0
        else:
            pitch = -math.pi / 2.0
        roll = math.atan2(-R[0, 1], R[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def pose7_world_to_fk(
    pose7: np.ndarray,
    base_xyz: np.ndarray,
    base_quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """World xyz+wxyz -> (p_fk, R_fk)."""
    pose7 = np.asarray(pose7, dtype=float).reshape(7)
    p_w = pose7[:3]
    R_w = quat_wxyz_to_matrix(pose7[3:])
    R_wb = quat_wxyz_to_matrix(base_quat_wxyz)
    t_wb = np.asarray(base_xyz, dtype=float).reshape(3)
    p_isaac = R_wb.T @ (p_w - t_wb)
    R_isaac = R_wb.T @ R_w
    p_fk = R_ISAAC_TO_FK @ p_isaac
    R_fk = R_ISAAC_TO_FK @ R_isaac
    return p_fk, R_fk


def delta_action_closed_loop(
    p0_fk: np.ndarray,
    R0_fk: np.ndarray,
    p1_fk: np.ndarray,
    R1_fk: np.ndarray,
    action_frame: str = "ee",
    gripper: float = 1.0,
) -> np.ndarray:
    """7-vector matching openvla_closed_loop.py (ee/base + RPY left-multiply)."""
    dp_fk = np.asarray(p1_fk, dtype=float) - np.asarray(p0_fk, dtype=float)
    if action_frame == "ee":
        dxyz = R0_fk.T @ dp_fk
    elif action_frame == "base":
        dxyz = dp_fk
    else:
        raise ValueError("action_frame must be 'ee' or 'base'")
    R_delta = np.asarray(R1_fk, dtype=float) @ np.asarray(R0_fk, dtype=float).T
    drpy = matrix_to_rpy(R_delta)
    return np.array(
        [dxyz[0], dxyz[1], dxyz[2], drpy[0], drpy[1], drpy[2], float(gripper)],
        dtype=np.float32,
    )
