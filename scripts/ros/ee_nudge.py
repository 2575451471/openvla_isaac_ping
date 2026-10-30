#!/usr/bin/env python3
"""Nudge the UR10e end-effector by about 5 cm in the kinematic base frame.

Reads /joint_states, FK to the current flange pose, adds a translation
(default +x 5 cm, rotation unchanged), numerical IK, then interpolates
/joint_command. No MoveIt. Kinematics follow ur_description UR10e YAML.

Isaac Sim must already be playing.

Run:
    source /opt/ros/humble/setup.bash
    python3 /home/dt/openvla/shenle_projects/scripts/ros/ee_nudge.py

Try another axis:
    python3 scripts/ros/ee_nudge.py --ros-args -p dx:=0.0 -p dy:=0.05
"""

import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


ARM_JOINTS = (
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
)

# ur_description/config/ur10e/default_kinematics.yaml joint origins, then Rz(q).
_UR10E_ORIGIN = (
    ((0.0, 0.0, 0.1807), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.570796327, 0.0, 0.0)),
    ((-0.6127, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-0.57155, 0.0, 0.17415), (0.0, 0.0, 0.0)),
    ((0.0, -0.11985, 0.0), (1.570796327, 0.0, 0.0)),
    ((0.0, 0.11655, 0.0), (1.570796326589793, math.pi, math.pi)),
)


def _joint_key(name: str):
    for arm in ARM_JOINTS:
        if name == arm or name.endswith('/' + arm) or name.endswith('_' + arm):
            return arm
    return None


def _rotz(q: float) -> np.ndarray:
    c, s = math.cos(q), math.sin(q)
    T = np.eye(4)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    return T


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    return T


def _xyz(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    return T


def fk(q) -> np.ndarray:
    """Wrist-3 pose in the UR10e kinematic base (meters)."""
    T = np.eye(4)
    for (xyz, rpy), qi in zip(_UR10E_ORIGIN, q):
        T = T @ _xyz(*xyz) @ _rpy(*rpy) @ _rotz(float(qi))
    return T


def _rot_log(R: np.ndarray) -> np.ndarray:
    cos = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    ang = math.acos(cos)
    if ang < 1e-8:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return (0.5 * ang / math.sin(ang)) * w


def _jacobian(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    T0 = fk(q)
    p0, R0 = T0[:3, 3], T0[:3, :3]
    J = np.zeros((6, 6))
    for i in range(6):
        dq = q.copy()
        dq[i] += eps
        Ti = fk(dq)
        J[:3, i] = (Ti[:3, 3] - p0) / eps
        J[3:, i] = _rot_log(R0.T @ Ti[:3, :3]) / eps
    return J


def ik(q_seed, T_goal, iters: int = 80, elbow_branch: int = 1,
       position_limit: float = 2e-3, rotation_limit: float = None,
       require_rotation: bool = True) -> np.ndarray:
    """Damped least-squares IK. Centimeter-scale steps should stay near q_seed.

    UR default [0, -90, 0, 0, 0, 0] deg is near elbow singularity: full 6-D
    solve often stalls. Try full-pose and position-weighted solves. The default
    accepts 2 mm position and 1 degree rotation. require_rotation=False keeps
    the position limit and does not reject a candidate for rotation error.
    Adapt the rotation weight when the initial candidates trade off these errors.

    This project's push demonstrations use the positive-elbow branch. Near
    straight HOME, seed perturbations explicitly select it instead of allowing
    sensor noise to choose elbow-up/down. Use elbow_branch=-1 for the other
    branch, or 0 for unconstrained legacy behavior.
    """
    if elbow_branch not in (-1, 0, 1):
        raise ValueError('elbow_branch must be -1, 0, or 1')
    seed = np.array(q_seed, dtype=float)
    q = seed.copy()
    p_goal = T_goal[:3, 3]
    R_goal = T_goal[:3, :3]

    def _solve(q0: np.ndarray, w_rot: float, n_iters: int) -> np.ndarray:
        q_local = np.array(q0, dtype=float)
        for _ in range(n_iters):
            T = fk(q_local)
            dp = p_goal - T[:3, 3]
            dw = _rot_log(T[:3, :3].T @ R_goal)
            if np.linalg.norm(dp) < 5e-5 and (w_rot <= 0.0 or np.linalg.norm(dw) < 1e-4):
                return q_local
            J = _jacobian(q_local)
            # Heavier damping near singularity (tiny singular values).
            smin = float(np.linalg.svd(J, compute_uv=False)[-1])
            damp = max(1e-4, 1e-2 * (1e-3 / max(smin, 1e-12)) ** 0.5)
            damp = min(damp, 5e-2)
            W = np.diag([1.0, 1.0, 1.0, w_rot, w_rot, w_rot])
            Jw = W @ J
            ew = W @ np.concatenate([dp, dw])
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + damp * np.eye(6), ew)
            q_local = q_local + np.clip(dq, -0.2, 0.2)
        return q_local

    def branch_ok(candidate):
        return elbow_branch == 0 or elbow_branch * math.sin(float(candidate[2])) >= -1e-4

    seeds = [seed]
    if elbow_branch:
        elbow = math.atan2(math.sin(float(seed[2])), math.cos(float(seed[2])))
        if elbow_branch * math.sin(elbow) < -0.2:
            raise RuntimeError('Current elbow is on the opposite IK branch; reset to the recorded initial pose')
        if abs(elbow) < 0.2:
            # The straight HOME pose does not distinguish elbow-up/down.
            # Bias only the IK seed, compensating adjacent joints to preserve
            # total pitch. Future recorded joints are never consulted.
            seeds = []
            for bend in (0.10, 0.25):
                preferred = seed.copy()
                delta = elbow_branch * bend - elbow
                preferred[1] -= delta / 2
                preferred[2] += delta
                preferred[3] -= delta / 2
                seeds.append(preferred)

    if rotation_limit is None:
        rotation_limit = math.radians(1)
    preferred_jump = math.radians(20)
    valid_candidates = []
    evaluated = []

    def evaluate(start, weight, count):
        candidate = _solve(start, w_rot=weight, n_iters=count)
        actual = fk(candidate)
        position_error = float(np.linalg.norm(p_goal - actual[:3, 3]))
        rotation_error = float(np.linalg.norm(_rot_log(actual[:3, :3].T @ R_goal)))
        jump = float(np.max(np.abs(candidate - seed)))
        preferred_branch = branch_ok(candidate)
        item = (candidate, position_error, rotation_error, jump, preferred_branch)
        evaluated.append(item)
        rotation_ok = (not require_rotation) or rotation_error <= rotation_limit
        if position_error <= position_limit and rotation_ok and preferred_branch:
            valid_candidates.append(item)
        return item

    def usable(item):
        _, position_error, rotation_error, jump, preferred_branch = item
        rotation_ok = (not require_rotation) or rotation_error <= rotation_limit
        return (position_error <= position_limit and rotation_ok
                and jump <= preferred_jump and preferred_branch)

    endpoints = {}
    # Preserve the existing fast path, but do not return on position alone.
    for weight in (1.0, 0.05):
        for index, start in enumerate(seeds):
            item = evaluate(start, weight, iters)
            endpoints[index, weight] = item
            if usable(item):
                return item[0]

    # Search between accurate-orientation/poor-position and the reverse.
    # Use a geometric midpoint because the angular weight spans orders of magnitude.
    # All returned candidates still satisfy the original pose limits.
    for index, start in enumerate(seeds):
        low, high = 0.05, 1.0
        low_item, high_item = endpoints[index, low], endpoints[index, high]
        if low_item[1] > position_limit or high_item[2] > rotation_limit:
            continue
        for _ in range(6):
            weight = math.sqrt(low * high)
            item = evaluate(start, weight, iters)
            if usable(item):
                return item[0]
            if item[1] / position_limit > item[2] / rotation_limit:
                high = weight
            else:
                low = weight

    for start in seeds:
        item = evaluate(start, 0.0, 100)
        if usable(item):
            return item[0]
    if valid_candidates:
        # Non-controller callers may intentionally request a larger joint move.
        # The production controller separately enforces its unchanged 20-degree limit.
        return min(valid_candidates, key=lambda item: item[3])[0]
    best = min(evaluated, key=lambda item: max(item[1] / position_limit, item[2] / rotation_limit))
    raise RuntimeError(
        f'IK failed on preferred elbow branch {elbow_branch}: '
        f'position={best[1] * 1000:.3f}mm rotation={math.degrees(best[2]):.3f}deg '
        f'joint_step={math.degrees(best[3]):.3f}deg'
    )


def _cmd_qos():
    reliable = QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
    )
    sensor = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
    )
    return reliable, sensor


class EeNudge(Node):
    """One 5 cm Cartesian nudge, then hold the solved joints."""

    def __init__(self):
        super().__init__('ee_nudge')
        self.declare_parameter('command_topic', '/joint_command')
        self.declare_parameter('state_topic', '/joint_states')
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('duration', 4.0)
        self.declare_parameter('dx', 0.05)
        self.declare_parameter('dy', 0.0)
        self.declare_parameter('dz', 0.0)

        command_topic = str(self.get_parameter('command_topic').value)
        state_topic = str(self.get_parameter('state_topic').value)
        publish_rate = float(self.get_parameter('publish_rate').value)
        self.duration = float(self.get_parameter('duration').value)
        self.delta = np.array([
            float(self.get_parameter('dx').value),
            float(self.get_parameter('dy').value),
            float(self.get_parameter('dz').value),
        ])
        if publish_rate <= 0.0 or self.duration <= 0.0:
            raise ValueError('publish_rate and duration must be > 0')

        reliable, sensor = _cmd_qos()
        # Isaac QoS is not consistent; one of these will match, the other may warn.
        self.command_pubs = [
            self.create_publisher(JointState, command_topic, reliable),
            self.create_publisher(JointState, command_topic, sensor),
        ]
        self.create_subscription(JointState, state_topic, self._on_joint_states, sensor)
        self.create_subscription(JointState, state_topic, self._on_joint_states, reliable)

        self.dt = 1.0 / publish_rate
        self.joint_names = None
        self.arm_index = None
        self.q_start = None
        self.q_goal = None
        self.p_start = None
        self.p_target = None
        self.q_meas = None
        self.elapsed = 0.0
        self.done = False
        self.waiting_logs = 0

        self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f'Waiting for {state_topic}, then nudge '
            f'dxyz={self.delta.tolist()} m over {self.duration:.1f}s'
        )

    def _arm_q(self, names, pos):
        arm_index = []
        q = []
        for joint in ARM_JOINTS:
            found = None
            for i, name in enumerate(names):
                if _joint_key(name) == joint:
                    found = i
                    break
            if found is None:
                return None, None
            arm_index.append(found)
            q.append(float(pos[found]))
        return arm_index, np.array(q, dtype=float)

    def _on_joint_states(self, msg: JointState):
        if not msg.name or not msg.position:
            return
        count = min(len(msg.name), len(msg.position))
        names = list(msg.name[:count])
        pos = list(msg.position[:count])

        if self.joint_names is not None:
            _, q_meas = self._arm_q(self.joint_names, pos)
            if q_meas is not None:
                self.q_meas = q_meas
            return

        arm_index, q = self._arm_q(names, pos)
        if q is None:
            self.get_logger().error(f'Missing UR arm joints in /joint_states names={names}')
            return

        T = fk(q)
        T_goal = T.copy()
        T_goal[:3, 3] = T[:3, 3] + self.delta
        try:
            q_goal = ik(q, T_goal)
        except RuntimeError as exc:
            self.get_logger().error(str(exc))
            return

        p0, p1 = T[:3, 3], fk(q_goal)[:3, 3]
        self.joint_names = names
        self.arm_index = arm_index
        self.q_start = q
        self.q_goal = q_goal
        self.p_start = p0
        self.p_target = p1
        self.q_meas = q
        self.get_logger().info(
            f'Plan (kinematic base): {p0.round(4).tolist()} -> {p1.round(4).tolist()} m, '
            f'delta={(p1 - p0).round(4).tolist()} m. '
            f'q_start={np.round(np.degrees(q), 1).tolist()} deg'
        )

    def _tick(self):
        if self.joint_names is None:
            self.waiting_logs += 1
            if self.waiting_logs % 20 == 1:
                self.get_logger().info('Still waiting for /joint_states. Is Isaac Sim playing?')
            return

        self.elapsed = min(self.elapsed + self.dt, self.duration)
        alpha = self.elapsed / self.duration
        q = (1.0 - alpha) * self.q_start + alpha * self.q_goal

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_names)
        positions = [0.0] * len(self.joint_names)
        for i, qi in zip(self.arm_index, q):
            positions[i] = float(qi)
        msg.position = positions
        for pub in self.command_pubs:
            pub.publish(msg)

        if alpha >= 1.0 and not self.done:
            self.done = True
            subs = sum(pub.get_subscription_count() for pub in self.command_pubs)
            q_now = self.q_meas if self.q_meas is not None else q
            p_now = fk(q_now)[:3, 3]
            moved = p_now - self.p_start
            self.get_logger().info(
                f'Nudge done, holding. subscribers={subs}. '
                f'measured EE {p_now.round(4).tolist()} m, '
                f'moved {moved.round(4).tolist()} m '
                f'(norm {np.linalg.norm(moved)*100:.1f} cm), '
                f'planned {self.delta.tolist()} m'
            )
            if subs == 0:
                self.get_logger().warning(
                    '/joint_command has 0 matching Isaac subscribers. '
                    'Command likely never arrived. Check playback, Move Robot, topic name. '
                    'The RELIABLE/BEST_EFFORT warning on this topic is normal if only one side matches.'
                )
            elif np.linalg.norm(moved - self.delta) > 0.01:
                self.get_logger().warning(
                    'Joints moved, but not the planned 5 cm in the kinematic base. '
                    'USD and ur_description may disagree.'
                )


def main(args=None):
    rclpy.init(args=args)
    node = EeNudge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
