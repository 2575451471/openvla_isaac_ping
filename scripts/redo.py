import asyncio

import math

import numpy as np

import omni.kit.app

import omni.usd
import omni.timeline

from pxr import Gf

from isaacsim.core.prims import SingleArticulation

from isaacsim.core.utils.types import ArticulationAction

# ============================================================
# 1. Scene paths
# ============================================================
ROBOT_PRIM = "/World/ur10e"

BASE_PRIM = "/World/ur10e/base_link"

EE_PRIM = "/World/ur10e/wrist_3_link"

# ============================================================
# 2. World-frame EE waypoints
#    这些就是 wrist_3_link 要经过的位置
# ============================================================
WAYPOINTS_WORLD = [

    ("P1",  0.11502,  0.88779, 1.36614),

    ("P2",  0.45119,  0.60536, 1.31328),

    ("P3",  0.45119,  0.16336, 0.77849),

    ("P4",  0.26073,  0.02425, 0.34563),

    ("P5",  0.26073,  0.02475, 0.12029),

    ("P6",  0.01342,  0.02475, 0.10029),

    ("P7", -0.68529,  0.02951, 0.10029),

]

# ============================================================
# 3. 最终目标姿态（来自你第二张图）
#    单位：degree
#    这里按 roll(X), pitch(Y), yaw(Z) 解释
# ============================================================
TARGET_WORLD_RPY_DEG = (-90.0, -88.0, -82.9)

# ============================================================
# 4. 姿态渐变计划
#
# 含义：
# 0.0 = 完全保持起始姿态
# 1.0 = 完全到达目标姿态
#
# 你要求：
# P1 -> P4 逐渐变成目标姿态
# P4 之后保持
# ============================================================
WAYPOINT_ORI_ALPHA = {

    "P1": 0.00,

    "P2": 0.33,

    "P3": 0.65,

    "P4": 0.85,

    "P5": 1.00,

    "P6": 1.00,

    "P7": 1.00,

}

# ============================================================
# 5. Motion / IK settings
# ============================================================
CART_STEP_M = 0.005

# 速度倍率：1.0 为上一版速度，2.0 为两倍速度（运动时长减半）。
# 若出现跟踪误差超限，可调为 1.5；倍率提高也会提高加速度。
MOTION_SPEED_SCALE = 10.0

# 以下为 1.0 倍的基准参数。实际关节速度上限乘倍率，加速度上限乘倍率平方。
CART_SPEED_M_S = 0.06
JOINT_SPEED_RAD_S = 0.35
JOINT_ACCEL_RAD_S2 = 0.60
MIN_KNOT_TIME_S = 0.04
SPLINE_PATH_TOL_M = 0.008  # 关节插值相对原笛卡尔折线的允许偏差
FINAL_SETTLE_TIMEOUT_S = 5.0

IK_MAX_ITERS = 500

IK_POS_TOL_M = 0.003                  # 3 mm

ORI_GOAL_TOL_RAD = math.radians(20.0) # 姿态误差容忍度较大

IK_MAX_DQ = 0.08

# 位置优先，姿态软约束
ORI_GAIN_MIN = 0.08

ORI_GAIN_MAX = 0.35

POSTURE_GAIN = 0.06

# 关节跳变保护
MAX_CART_JOINT_JUMP = math.radians(15.0)

# 执行阶段
MAX_TRACK_FRAMES = 60

JOINT_TRACK_TOL = 0.02

TRACK_ABORT_RAD = 0.10

EE_ABORT_ERROR_M = 0.04

HOLD_FRAMES = 0  # 中间航点不再停留

END_HOLD_FRAMES = 80

ABORT_HOLD_FRAMES = 0  # 中间航点不再停留

FK_CHECK_TOL_M = 0.02

# ============================================================
# 6. UR10e arm joints
# ============================================================
ARM_JOINTS = (

    "shoulder_pan_joint",

    "shoulder_lift_joint",

    "elbow_joint",

    "wrist_1_joint",

    "wrist_2_joint",

    "wrist_3_joint",

)

# ============================================================
# 7. UR10e FK model
# ============================================================
_UR10E_ORIGIN = (

    ((0.0, 0.0, 0.1807), (0.0, 0.0, 0.0)),

    ((0.0, 0.0, 0.0), (1.570796327, 0.0, 0.0)),

    ((-0.6127, 0.0, 0.0), (0.0, 0.0, 0.0)),

    ((-0.57155, 0.0, 0.17415), (0.0, 0.0, 0.0)),

    ((0.0, -0.11985, 0.0), (1.570796327, 0.0, 0.0)),

    ((0.0, 0.11655, 0.0), (1.570796326589793, math.pi, math.pi)),

)

# ============================================================
# 8. Isaac base frame <-> custom FK frame
#    已验证两者相差 Rz(180 deg)
# ============================================================
R_ISAAC_TO_FK = np.array(

    [

        [-1.0,  0.0, 0.0],

        [ 0.0, -1.0, 0.0],

        [ 0.0,  0.0, 1.0],

    ],

    dtype=float,

)

def isaac_base_to_fk(p):

    return R_ISAAC_TO_FK @ np.asarray(p, dtype=float)

def fk_to_isaac_base(p):

    return R_ISAAC_TO_FK @ np.asarray(p, dtype=float)

def isaac_rot_to_fk(R_isaac):

    return R_ISAAC_TO_FK @ np.asarray(R_isaac, dtype=float)

def fk_rot_to_isaac(R_fk):

    return R_ISAAC_TO_FK @ np.asarray(R_fk, dtype=float)

# ============================================================
# 9. Basic transforms / rotations
# ============================================================
def _xyz(x, y, z):

    T = np.eye(4)

    T[0, 3] = x

    T[1, 3] = y

    T[2, 3] = z

    return T

def _rotz(q):

    c = math.cos(q)

    s = math.sin(q)

    T = np.eye(4)

    T[0, 0] = c

    T[0, 1] = -s

    T[1, 0] = s

    T[1, 1] = c

    return T

def _rpy(roll, pitch, yaw):

    cr = math.cos(roll)

    sr = math.sin(roll)

    cp = math.cos(pitch)

    sp = math.sin(pitch)

    cy = math.cos(yaw)

    sy = math.sin(yaw)

    T = np.eye(4)

    T[:3, :3] = np.array(

        [

            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],

            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],

            [-sp,     cp * sr,                cp * cr],

        ],

        dtype=float,

    )

    return T

def rpy_deg_to_matrix(rx_deg, ry_deg, rz_deg):

    rx = math.radians(rx_deg)

    ry = math.radians(ry_deg)

    rz = math.radians(rz_deg)

    return _rpy(rx, ry, rz)[:3, :3]

def matrix_to_rpy_deg(R):

    R = np.asarray(R, dtype=float)

    if abs(R[2, 0]) < 1.0 - 1e-8:

        pitch = -math.asin(R[2, 0])

        roll = math.atan2(R[2, 1], R[2, 2])

        yaw = math.atan2(R[1, 0], R[0, 0])

    else:

        # gimbal lock
        if R[2, 0] <= -1.0:

            pitch = math.pi / 2.0

        else:

            pitch = -math.pi / 2.0

        roll = math.atan2(-R[0, 1], R[1, 1])

        yaw = 0.0

    return np.degrees([roll, pitch, yaw])

def skew(v):

    x, y, z = v

    return np.array(

        [

            [0.0, -z,   y],

            [z,   0.0, -x],

            [-y,   x,  0.0],

        ],

        dtype=float,

    )

def rotvec_to_matrix(r):

    r = np.asarray(r, dtype=float)

    theta = float(np.linalg.norm(r))

    if theta < 1e-10:

        K = skew(r)

        return np.eye(3) + K

    k = r / theta

    K = skew(k)

    return (

        np.eye(3)

        + math.sin(theta) * K

        + (1.0 - math.cos(theta)) * (K @ K)

    )

def rotation_vector(R):

    R = np.asarray(R, dtype=float)

    cos_theta = (np.trace(R) - 1.0) * 0.5

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = math.acos(float(cos_theta))

    if theta < 1e-8:

        return 0.5 * np.array(

            [

                R[2, 1] - R[1, 2],

                R[0, 2] - R[2, 0],

                R[1, 0] - R[0, 1],

            ],

            dtype=float,

        )

    if abs(math.pi - theta) < 1e-4:

        A = 0.5 * (R + np.eye(3))

        axis = np.sqrt(np.maximum(np.diag(A), 0.0))

        if (R[2, 1] - R[1, 2]) < 0:

            axis[0] *= -1.0

        if (R[0, 2] - R[2, 0]) < 0:

            axis[1] *= -1.0

        if (R[1, 0] - R[0, 1]) < 0:

            axis[2] *= -1.0

        n = np.linalg.norm(axis)

        if n < 1e-8:

            axis = np.array([1.0, 0.0, 0.0], dtype=float)

        else:

            axis = axis / n

        return theta * axis

    axis = (

        1.0 / (2.0 * math.sin(theta))

    ) * np.array(

        [

            R[2, 1] - R[1, 2],

            R[0, 2] - R[2, 0],

            R[1, 0] - R[0, 1],

        ],

        dtype=float,

    )

    return theta * axis

def interpolate_rotation(R0, R1, alpha):

    alpha = float(np.clip(alpha, 0.0, 1.0))

    R_rel = R1 @ R0.T

    rv = rotation_vector(R_rel)

    return rotvec_to_matrix(alpha * rv) @ R0

def orientation_error(R_current, R_target):

    R_error = R_target @ R_current.T

    return rotation_vector(R_error)

# ============================================================
# 10. FK
# ============================================================
def fk(q):

    q = np.asarray(q, dtype=float)

    T = np.eye(4)

    for (xyz, rpy), qi in zip(_UR10E_ORIGIN, q):

        T = T @ _xyz(*xyz) @ _rpy(*rpy) @ _rotz(float(qi))

    return T

# ============================================================
# 11. Jacobians
# ============================================================
def jacobian_pose(q, eps=1e-5):

    q = np.asarray(q, dtype=float)

    T0 = fk(q)

    p0 = T0[:3, 3]

    R0 = T0[:3, :3]

    Jp = np.zeros((3, 6), dtype=float)

    Jr = np.zeros((3, 6), dtype=float)

    for i in range(6):

        q2 = q.copy()

        q2[i] += eps

        T2 = fk(q2)

        p2 = T2[:3, 3]

        R2 = T2[:3, :3]

        Jp[:, i] = (p2 - p0) / eps

        R_delta = R2 @ R0.T

        Jr[:, i] = rotation_vector(R_delta) / eps

    return Jp, Jr

# ============================================================
# 12. Scheduled orientation IK
#
# PRIMARY:
#   XYZ
#
# SECONDARY:
#   跟踪“当前阶段的目标姿态”
#
# THIRD:
#   尽量靠近上一小步的关节解
# ============================================================
def ik_position_with_scheduled_orientation(q_seed, p_goal, R_target, ori_gain):

    q_seed = np.asarray(q_seed, dtype=float)

    p_goal = np.asarray(p_goal, dtype=float).reshape(3)

    q = q_seed.copy()

    best_feasible_q = None

    best_feasible_pos = float("inf")

    best_feasible_ori = float("inf")

    for _ in range(IK_MAX_ITERS):

        T = fk(q)

        p_now = T[:3, 3]

        R_now = T[:3, :3]

        e_pos = p_goal - p_now

        pos_err = float(np.linalg.norm(e_pos))

        e_ori = orientation_error(R_now, R_target)

        ori_err = float(np.linalg.norm(e_ori))

        # 只要位置达标，就先记下来
        if pos_err <= IK_POS_TOL_M:

            if best_feasible_q is None or ori_err < best_feasible_ori:

                best_feasible_q = q.copy()

                best_feasible_pos = pos_err

                best_feasible_ori = ori_err

            if ori_err <= ORI_GOAL_TOL_RAD:

                return q, pos_err, ori_err

        Jp, Jr = jacobian_pose(q)

        # -------------------------
        # Position DLS
        # -------------------------
        if pos_err > 0.10:

            lambda_pos = 0.03

        elif pos_err > 0.03:

            lambda_pos = 0.01

        else:

            lambda_pos = 0.003

        A_pos = Jp @ Jp.T + (lambda_pos ** 2) * np.eye(3)

        Jp_pinv = Jp.T @ np.linalg.solve(A_pos, np.eye(3))

        dq_pos = Jp_pinv @ e_pos

        # Position null-space
        N = np.eye(6) - Jp_pinv @ Jp

        # -------------------------
        # Orientation in null-space
        # -------------------------
        JrN = Jr @ N

        lambda_ori = 0.03

        A_ori = JrN @ JrN.T + (lambda_ori ** 2) * np.eye(3)

        dq_ori = N @ JrN.T @ np.linalg.solve(A_ori, e_ori)

        # -------------------------
        # Posture regularization
        # -------------------------
        dq_posture = N @ (q_seed - q)

        dq = dq_pos + ori_gain * dq_ori + POSTURE_GAIN * dq_posture

        dq_norm = float(np.linalg.norm(dq))

        if dq_norm > IK_MAX_DQ:

            dq *= IK_MAX_DQ / dq_norm

        current_score = pos_err + 0.03 * ori_err

        improved = False

        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):

            q_try = q + alpha * dq

            T_try = fk(q_try)

            p_try = T_try[:3, 3]

            R_try = T_try[:3, :3]

            pos_try = float(np.linalg.norm(p_goal - p_try))

            ori_try = float(np.linalg.norm(orientation_error(R_try, R_target)))

            score_try = pos_try + 0.03 * ori_try

            if pos_try <= pos_err + 0.0002 and score_try < current_score:

                q = q_try

                improved = True

                break

        # 如果联合优化没改善，就退回纯 position
        if not improved:

            for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):

                q_try = q + alpha * dq_pos

                p_try = fk(q_try)[:3, 3]

                pos_try = float(np.linalg.norm(p_goal - p_try))

                if pos_try < pos_err:

                    q = q_try

                    improved = True

                    break

        if not improved:

            break

    if best_feasible_q is not None:

        return best_feasible_q, best_feasible_pos, best_feasible_ori

    T_final = fk(q)

    final_pos_err = float(np.linalg.norm(p_goal - T_final[:3, 3]))

    raise RuntimeError(

        "IK failed: position error {:.2f} mm, goal FK={}".format(

            final_pos_err * 1000.0,

            np.round(p_goal, 5),

        )

    )

# ============================================================
# 13. USD helpers
# ============================================================
def get_world_matrix(path):

    stage = omni.usd.get_context().get_stage()

    prim = stage.GetPrimAtPath(path)

    if not prim.IsValid():

        raise RuntimeError("Invalid prim: " + path)

    return omni.usd.get_world_transform_matrix(prim)

def gf_matrix_to_np_rotation(T):

    return np.array(

        [

            [T[0][0], T[0][1], T[0][2]],

            [T[1][0], T[1][1], T[1][2]],

            [T[2][0], T[2][1], T[2][2]],

        ],

        dtype=float,

    )

def get_world_position(path):

    T = get_world_matrix(path)

    p = T.ExtractTranslation()

    return np.array([p[0], p[1], p[2]], dtype=float)

def get_world_rotation(path):

    T = get_world_matrix(path)

    return gf_matrix_to_np_rotation(T)

# ============================================================
# 14. World <-> Isaac base
# ============================================================
def world_to_isaac_base(p_world, T_world_base):

    p_world = np.asarray(p_world, dtype=float)

    p = Gf.Vec3d(float(p_world[0]), float(p_world[1]), float(p_world[2]))

    p_local = T_world_base.GetInverse().Transform(p)

    return np.array([p_local[0], p_local[1], p_local[2]], dtype=float)

def isaac_base_to_world(p_base, T_world_base):

    p_base = np.asarray(p_base, dtype=float)

    p = Gf.Vec3d(float(p_base[0]), float(p_base[1]), float(p_base[2]))

    pw = T_world_base.Transform(p)

    return np.array([pw[0], pw[1], pw[2]], dtype=float)

# ============================================================
# 15. Joint helpers
# ============================================================
def find_arm_indices(dof_names):

    indices = []

    for target in ARM_JOINTS:

        hits = [

            i

            for i, name in enumerate(dof_names)

            if (

                name == target

                or name.endswith("/" + target)

                or name.endswith("_" + target)

            )

        ]

        if not hits:

            raise RuntimeError("Missing joint: " + target)

        indices.append(hits[0])

    return indices

def read_arm_q(robot, arm_idx):

    raw = robot.get_joint_positions()

    if raw is None:

        raise RuntimeError("Cannot read joints. Is Isaac Sim still Playing?")

    q_all = np.asarray(raw, dtype=float).reshape(-1)

    required_size = max(arm_idx) + 1

    if q_all.size < required_size:

        raise RuntimeError(

            "Joint state unavailable (simulation may have been stopped). "

            "Received {} values.".format(q_all.size)

        )

    return q_all[arm_idx].copy()

def send_arm_target(robot, arm_idx, q_arm):

    action = ArticulationAction(

        joint_positions=np.asarray(q_arm, dtype=float),

        joint_indices=np.asarray(arm_idx, dtype=np.int32),

    )

    robot.apply_action(action)

# ============================================================
# 16. Interpolation helpers
# ============================================================
def interpolate_segment(p0, p1):

    p0 = np.asarray(p0, dtype=float)

    p1 = np.asarray(p1, dtype=float)

    distance = float(np.linalg.norm(p1 - p0))

    steps = max(1, int(math.ceil(distance / CART_STEP_M)))

    result = []

    for i in range(1, steps + 1):

        t = i / float(steps)

        result.append((1.0 - t) * p0 + t * p1)

    return result

def spline_coefficients(q, h):
    """Clamped cubic spline: C2 continuous, endpoint velocities zero (NumPy only)."""
    n = len(q)
    delta = np.diff(q, axis=0) / h[:, None]
    lower = np.zeros(n)
    diag = np.zeros(n)
    upper = np.zeros(n)
    rhs = np.zeros_like(q)
    diag[0], upper[0], rhs[0] = 2*h[0], h[0], 6*delta[0]
    lower[-1], diag[-1], rhs[-1] = h[-1], 2*h[-1], -6*delta[-1]
    for i in range(1, n-1):
        lower[i], diag[i], upper[i] = h[i-1], 2*(h[i-1]+h[i]), h[i]
        rhs[i] = 6*(delta[i]-delta[i-1])
    for i in range(1, n):
        ratio = lower[i]/diag[i-1]
        diag[i] -= ratio*upper[i-1]
        rhs[i] -= ratio*rhs[i-1]
    m = np.empty_like(q)
    m[-1] = rhs[-1]/diag[-1]
    for i in range(n-2, -1, -1):
        m[i] = (rhs[i]-upper[i]*m[i+1])/diag[i]
    a = q[:-1]
    b = delta-h[:, None]*(2*m[:-1]+m[1:])/6
    c = m[:-1]/2
    d = (m[1:]-m[:-1])/(6*h[:, None])
    return np.stack((a,b,c,d), axis=1)


def evaluate_spline(times, coeff, t):
    i = min(max(int(np.searchsorted(times, t, side="right")-1), 0), len(coeff)-1)
    x = float(np.clip(t-times[i], 0.0, times[i+1]-times[i]))
    a,b,c,d = coeff[i]
    return a+x*(b+x*(c+x*d))


def build_continuous_spline(q_start, p_start, trajectory):
    speed_scale = float(MOTION_SPEED_SCALE)
    if not math.isfinite(speed_scale) or speed_scale <= 0.0:
        raise ValueError("MOTION_SPEED_SCALE must be finite and positive.")
    q = np.vstack([q_start]+[item["q"] for item in trajectory])
    p = np.vstack([p_start]+[item["target_fk"] for item in trajectory])
    h = np.maximum.reduce((
        np.linalg.norm(np.diff(p, axis=0), axis=1)/CART_SPEED_M_S,
        np.max(np.abs(np.diff(q, axis=0)), axis=1)/JOINT_SPEED_RAD_S,
        np.full(len(q)-1, MIN_KNOT_TIME_S),
    ))
    coeff = spline_coefficients(q,h)
    vmax, amax = 0.0, 0.0
    for i, (a,b,c,d) in enumerate(coeff):
        # 速度是二次式：端点及内部驻点给出精确最大值。
        candidates = [np.zeros(6), np.full(6,h[i])]
        x = np.divide(-c, 3*d, out=np.zeros(6), where=np.abs(d)>1e-14)
        candidates.append(np.clip(x,0,h[i]))
        for x in candidates:
            vmax = max(vmax,float(np.max(np.abs(b+2*c*x+3*d*x*x))))
        amax = max(amax,float(np.max(np.abs(2*c))),
                   float(np.max(np.abs(2*c+6*d*h[i]))))
        # Verify geometric deviation from each original Cartesian line segment.
        p0, p1 = p[i], p[i+1]
        line = p1-p0
        for u in np.linspace(0,1,9):
            x = u*h[i]
            actual = fk(a+x*(b+x*(c+x*d)))[:3,3]
            fraction = float(np.clip(np.dot(actual-p0,line)/max(np.dot(line,line),1e-16),0,1))
            deviation = np.linalg.norm(actual-(p0+fraction*line))
            if deviation > SPLINE_PATH_TOL_M:
                raise RuntimeError("Spline path deviation {:.2f} mm exceeds limit. Reduce CART_STEP_M.".format(1000*deviation))
    scale = max(1.0,vmax/JOINT_SPEED_RAD_S,math.sqrt(amax/JOINT_ACCEL_RAD_S2))*1.05
    baseline_duration = float(np.sum(h)) * scale
    h *= scale / speed_scale
    print("Spline retiming factor: {:.2f}; motion speed multiplier: {:.2f}".format(
        scale, speed_scale))
    print("Motion duration: {:.2f} s -> {:.2f} s".format(
        baseline_duration, float(np.sum(h))))
    print("Reference joint limits: velocity {:.2f} rad/s, acceleration {:.2f} rad/s^2".format(
        JOINT_SPEED_RAD_S * speed_scale, JOINT_ACCEL_RAD_S2 * speed_scale**2))
    coeff = spline_coefficients(q,h)
    return np.r_[0.0,np.cumsum(h)], coeff


async def hold_current_pose(robot, arm_idx, app):

    try:

        q_hold = read_arm_q(robot, arm_idx)

    except Exception:

        return

    for _ in range(ABORT_HOLD_FRAMES):

        send_arm_target(robot, arm_idx, q_hold)

        await app.next_update_async()

# ============================================================
# 17. MAIN
# ============================================================
async def run():

    print("")

    print("==========================================")

    print("UR10e continuous Cartesian trajectory V6")

    print("P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7")

    print("EE = wrist_3_link")

    print("XYZ PRIMARY + SCHEDULED ORIENTATION TARGET")

    print("P1~P5: orientation gradually approaches target")

    print("P5~P7: keep target orientation")

    print("==========================================")

    robot = SingleArticulation(prim_path=ROBOT_PRIM)

    robot.initialize()

    dof_names = list(robot.dof_names)

    arm_idx = find_arm_indices(dof_names)

    q_start = read_arm_q(robot, arm_idx)

    print("")

    print("Current arm q:", np.round(q_start, 5))

    T_world_base = get_world_matrix(BASE_PRIM)

    base_world_xyz = get_world_position(BASE_PRIM)

    R_world_base = get_world_rotation(BASE_PRIM)

    print("Base WORLD XYZ:", np.round(base_world_xyz, 5))

    # --------------------------------------------------------
    # Current FK
    # --------------------------------------------------------
    T_start_fk = fk(q_start)

    p_start_fk = T_start_fk[:3, 3].copy()

    R_start_fk = T_start_fk[:3, :3].copy()

    # --------------------------------------------------------
    # Desired FINAL world orientation
    # --------------------------------------------------------
    R_target_world = rpy_deg_to_matrix(*TARGET_WORLD_RPY_DEG)

    # world -> Isaac base
    R_target_isaac = R_world_base.T @ R_target_world

    # Isaac base -> custom FK
    R_target_fk = isaac_rot_to_fk(R_target_isaac)

    print("")

    print("Desired FINAL world orientation (deg):", TARGET_WORLD_RPY_DEG)

    # --------------------------------------------------------
    # FK / Isaac consistency check
    # --------------------------------------------------------
    ee_world_actual = get_world_position(EE_PRIM)

    ee_isaac_base = world_to_isaac_base(ee_world_actual, T_world_base)

    ee_fk_as_isaac = fk_to_isaac_base(p_start_fk)

    fk_error = float(np.linalg.norm(ee_isaac_base - ee_fk_as_isaac))

    print("")

    print("EE Isaac base XYZ:", np.round(ee_isaac_base, 5))

    print("EE corrected FK XYZ:", np.round(ee_fk_as_isaac, 5))

    print("FK / Isaac mismatch: {:.3f} mm".format(fk_error * 1000.0))

    if fk_error > FK_CHECK_TOL_M:

        raise RuntimeError(

            "FK mismatch too large: {:.1f} mm. Robot will NOT move.".format(

                fk_error * 1000.0

            )

        )

    # --------------------------------------------------------
    # Convert waypoints
    # --------------------------------------------------------
    waypoints = []

    print("")

    print("========== WAYPOINT CONVERSION ==========")

    for name, x, y, z in WAYPOINTS_WORLD:

        p_world = np.array([x, y, z], dtype=float)

        p_isaac = world_to_isaac_base(p_world, T_world_base)

        p_fk = isaac_base_to_fk(p_isaac)

        alpha_goal = WAYPOINT_ORI_ALPHA[name]

        waypoints.append(

            {

                "name": name,

                "world": p_world,

                "fk": p_fk,

                "alpha_goal": alpha_goal,

            }

        )

        print(

            "{}  WORLD={}  FK={}  ori_alpha={:.2f}".format(

                name,

                np.round(p_world, 5),

                np.round(p_fk, 5),

                alpha_goal,

            )

        )

    # ========================================================
    # PRECOMPUTE FULL TRAJECTORY
    # ========================================================
    print("")

    print("==========================================")

    print("V5 IK PRECOMPUTATION")

    print("XYZ = primary")

    print("Orientation target = scheduled")

    print("Robot is NOT moving yet.")

    print("==========================================")

    trajectory = []

    q_seed = q_start.copy()

    previous_fk = p_start_fk.copy()

    previous_alpha = 0.0

    for wp in waypoints:

        name = wp["name"]

        target_fk = wp["fk"]

        alpha_goal = wp["alpha_goal"]

        cart_points = interpolate_segment(previous_fk, target_fk)

        distance = float(np.linalg.norm(target_fk - previous_fk))

        print("")

        print(

            "{}: distance={:.3f} m, steps={}, alpha {:.2f} -> {:.2f}".format(

                name,

                distance,

                len(cart_points),

                previous_alpha,

                alpha_goal,

            )

        )

        for point_index, p_goal_fk in enumerate(cart_points):

            s = (point_index + 1) / float(len(cart_points))

            # 当前子点的姿态进度
            ori_alpha = (1.0 - s) * previous_alpha + s * alpha_goal

            # 当前子点的目标姿态：从起始姿态逐渐插值到目标姿态
            R_ref = interpolate_rotation(R_start_fk, R_target_fk, ori_alpha)

            # 越靠近目标姿态，约束稍微强一些
            ori_gain = ORI_GAIN_MIN + (ORI_GAIN_MAX - ORI_GAIN_MIN) * ori_alpha

            try:

                q_solution, pos_err, ori_err = ik_position_with_scheduled_orientation(

                    q_seed,

                    p_goal_fk,

                    R_ref,

                    ori_gain,

                )

            except Exception as exc:

                print("")

                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                print("IK PRECOMPUTATION FAILED")

                print("Robot has NOT moved.")

                print("Waypoint:", name)

                print("Subpoint: {}/{}".format(point_index + 1, len(cart_points)))

                print("Target orientation alpha:", round(ori_alpha, 3))

                print(str(exc))

                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                return

            dq_cart = q_solution - q_seed

            max_jump = float(np.max(np.abs(dq_cart)))

            if max_jump > MAX_CART_JOINT_JUMP:

                print("")

                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                print("IK BRANCH JUMP DETECTED")

                print("Robot has NOT moved.")

                print("Waypoint:", name)

                print("Subpoint: {}/{}".format(point_index + 1, len(cart_points)))

                print("Maximum joint jump: {:.2f} deg".format(math.degrees(max_jump)))

                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                return

            p_goal_isaac = fk_to_isaac_base(p_goal_fk)

            p_goal_world = isaac_base_to_world(p_goal_isaac, T_world_base)

            trajectory.append(

                {

                    "waypoint": name,

                    "q": q_solution.copy(),

                    "target_world": p_goal_world.copy(),
                    "target_fk": p_goal_fk.copy(),

                    "formal_world": wp["world"].copy(),

                    "is_waypoint": (point_index == len(cart_points) - 1),

                    "pos_error": pos_err,

                    "ori_error": ori_err,

                    "ori_alpha": ori_alpha,

                    "R_ref_fk": R_ref.copy(),

                }

            )

            q_seed = q_solution.copy()

        last_item = trajectory[-1]

        print(

            "{} IK OK | pos={:.2f} mm | orientation err={:.2f} deg | alpha={:.2f}".format(

                name,

                last_item["pos_error"] * 1000.0,

                math.degrees(last_item["ori_error"]),

                last_item["ori_alpha"],

            )

        )

        previous_fk = target_fk.copy()

        previous_alpha = alpha_goal

    print("")

    print("==========================================")

    print("ALL V5 IK SOLUTIONS SUCCESSFUL")

    print("Total Cartesian points:", len(trajectory))

    print("IK finished; building continuous spline before motion.")

    print("==========================================")

    # 整条轨迹统一时间参数化；只有开头/结尾的速度为零。
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        raise RuntimeError("Please press Play before running this script.")

    times, coeff = build_continuous_spline(q_start, p_start_fk, trajectory)
    print("Continuous trajectory duration: {:.2f} s".format(times[-1]))
    q_actual = read_arm_q(robot, arm_idx)
    if np.max(np.abs(q_actual - q_start)) > JOINT_TRACK_TOL:
        raise RuntimeError("Robot moved during planning. Re-run from its current pose.")

    start_time = timeline.get_current_time()
    previous_time = start_time
    reported_index = 0
    while True:
        if not timeline.is_playing():
            raise RuntimeError("Simulation paused/stopped during motion.")
        now = timeline.get_current_time()
        if now < previous_time - 1e-8:
            await hold_current_pose(robot, arm_idx, app)
            raise RuntimeError("Timeline rewound/looped. Motion interrupted.")
        previous_time = now
        elapsed = min(max(now - start_time, 0.0), times[-1])
        q_target = evaluate_spline(times, coeff, elapsed)

        # 比较当前姿态和当前时刻参考，再发送下一帧目标；不逐点等待到位。
        q_actual = read_arm_q(robot, arm_idx)
        joint_error = float(np.max(np.abs(q_actual - q_target)))
        p_ref_world = isaac_base_to_world(
            fk_to_isaac_base(fk(q_target)[:3, 3]), T_world_base
        )
        ee_error = float(np.linalg.norm(get_world_position(EE_PRIM) - p_ref_world))
        if joint_error > TRACK_ABORT_RAD or ee_error > EE_ABORT_ERROR_M:
            print("Tracking limit exceeded: joint={:.2f} deg, EE={:.2f} mm".format(
                math.degrees(joint_error), 1000.0 * ee_error))
            await hold_current_pose(robot, arm_idx, app)
            return
        send_arm_target(robot, arm_idx, q_target)

        # 航点只记录经过信息，不保持、不阻塞；这是参考经过而非实测到位。
        while reported_index < len(trajectory) and times[reported_index + 1] <= elapsed:
            item = trajectory[reported_index]
            if item["is_waypoint"]:
                print("Reference passed", item["waypoint"])
            reported_index += 1
        if elapsed >= times[-1]:
            break
        await app.next_update_async()

    # 仅在最终点收敛并保持；中间没有等待到位。
    q_final = trajectory[-1]["q"]
    settle_start = timeline.get_current_time()
    while True:
        if not timeline.is_playing():
            raise RuntimeError("Simulation paused/stopped at final point.")
        send_arm_target(robot, arm_idx, q_final)
        await app.next_update_async()
        now = timeline.get_current_time()
        if now < previous_time - 1e-8:
            await hold_current_pose(robot, arm_idx, app)
            raise RuntimeError("Timeline rewound at final point.")
        previous_time = now
        q_error = float(np.max(np.abs(read_arm_q(robot, arm_idx) - q_final)))
        ee_error = float(np.linalg.norm(get_world_position(EE_PRIM) - trajectory[-1]["formal_world"]))
        if q_error > TRACK_ABORT_RAD or ee_error > EE_ABORT_ERROR_M:
            await hold_current_pose(robot, arm_idx, app)
            raise RuntimeError("Final tracking limit exceeded.")
        if q_error <= JOINT_TRACK_TOL and ee_error <= IK_POS_TOL_M + fk_error:
            break
        if now - settle_start > FINAL_SETTLE_TIMEOUT_S:
            await hold_current_pose(robot, arm_idx, app)
            raise RuntimeError("Final point did not settle within timeout.")
    for _ in range(END_HOLD_FRAMES):
        if not timeline.is_playing():
            return
        send_arm_target(robot, arm_idx, q_final)
        await app.next_update_async()
    print("DONE: P7 reached; final pose held.")


# ============================================================
# 18. Safe entry
# ============================================================
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
