#!/usr/bin/env python3
"""Closed loop: /rgb -> OpenVLA /act -> clip -> IK -> interpolate -> /joint_command.

OpenVLA returns a 7-vector of EE deltas. Gripper is dropped. Translation is
capped at about 5 cm per step. Translation frame is selectable:
  delta_frame=base  add (dx,dy,dz) in the UR kinematic base (old behavior)
  delta_frame=ee    add R_ee @ (dx,dy,dz) in the current end-effector frame

Starts from the arm's current measured joint pose (no forced home). Inference
is ~0.8 s (~1 Hz). Joint commands publish faster (default 50 Hz) and linearly
interpolate from the previous command to each new IK goal so the arm does not
jump when a step arrives.

Run:
    source /opt/ros/humble/setup.bash
    python3 /home/dt/openvla/shenle_projects/scripts/ros/openvla_closed_loop.py
    python3 .../ros/openvla_closed_loop.py --ros-args -p delta_frame:=base -p max_steps:=20
"""

import json
import sys
import threading
import time
from datetime import datetime
from base64 import b64decode, b64encode
from pathlib import Path

import numpy as np
import requests
import rclpy
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ee_nudge import ARM_JOINTS, _cmd_qos, _joint_key, _rpy, fk, ik  # noqa: E402


def dumps_numpy(payload: dict) -> str:
    def default(obj):
        if isinstance(obj, np.ndarray):
            array = np.ascontiguousarray(obj)
            return {
                '__numpy__': b64encode(array.tobytes()).decode('ascii'),
                'dtype': dtype_to_descr(array.dtype),
                'shape': array.shape,
            }
        raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

    return json.dumps(payload, default=default)


def loads_numpy(text: str):
    def hook(dct):
        if '__numpy__' not in dct:
            return dct
        values = np.frombuffer(b64decode(dct['__numpy__']), descr_to_dtype(dct['dtype']))
        shape = dct.get('shape')
        return values.reshape(shape) if shape else values[0]

    return json.loads(text, object_hook=hook)


def image_to_rgb(msg: Image) -> np.ndarray:
    height = int(msg.height)
    width = int(msg.width)
    encoding = msg.encoding.lower()
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(encoding)
    if channels is None:
        raise ValueError(f'unsupported encoding {msg.encoding}')
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    row_bytes = int(msg.step) if msg.step else width * channels
    pixels = raw.reshape(height, row_bytes)[:, : width * channels].reshape(height, width, channels)
    if encoding == 'bgr8':
        pixels = pixels[:, :, ::-1]
    elif encoding == 'rgba8':
        pixels = pixels[:, :, :3]
    elif encoding == 'bgra8':
        pixels = pixels[:, :, 2::-1]
    return np.ascontiguousarray(pixels, dtype=np.uint8)


LOG_DIR = Path(__file__).resolve().parents[2] / 'logs'
LATEST_LOG = LOG_DIR / 'openvla_closed_loop.latest.log'


def _clip_vec(vec, max_norm):
    vec = np.asarray(vec, dtype=float)
    n = np.linalg.norm(vec)
    if n > max_norm > 0.0:
        return vec * (max_norm / n)
    return vec


class ClosedLoop(Node):
    """Stream EE deltas from OpenVLA into joint commands."""

    def __init__(self):
        super().__init__('openvla_closed_loop')
        self.declare_parameter('image_topic', '/rgb')
        self.declare_parameter('state_topic', '/joint_states')
        self.declare_parameter('command_topic', '/joint_command')
        self.declare_parameter('server_url', 'http://127.0.0.1:8000/act')
        self.declare_parameter('instruction', 'push the red cylinder to the right')
        self.declare_parameter('unnorm_key', 'bridge_orig')
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('interp_duration', 0.75)
        self.declare_parameter('max_trans_m', 0.05)
        self.declare_parameter('max_rot_rad', 0.10)
        self.declare_parameter('max_steps', 300)
        self.declare_parameter('delta_frame', 'ee')

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.server_url = str(self.get_parameter('server_url').value)
        self.instruction = str(self.get_parameter('instruction').value)
        self.unnorm_key = str(self.get_parameter('unnorm_key').value)
        publish_rate = float(self.get_parameter('publish_rate').value)
        self.interp_duration = float(self.get_parameter('interp_duration').value)
        self.max_trans_m = float(self.get_parameter('max_trans_m').value)
        self.max_rot_rad = float(self.get_parameter('max_rot_rad').value)
        self.max_steps = int(self.get_parameter('max_steps').value)
        self.delta_frame = str(self.get_parameter('delta_frame').value).strip().lower()
        command_topic = str(self.get_parameter('command_topic').value)
        state_topic = str(self.get_parameter('state_topic').value)
        if publish_rate <= 0.0 or self.max_trans_m <= 0.0 or self.interp_duration <= 0.0:
            raise ValueError('publish_rate, max_trans_m, and interp_duration must be > 0')
        if self.delta_frame not in ('base', 'ee'):
            raise ValueError("delta_frame must be 'base' or 'ee'")

        reliable, sensor = _cmd_qos()
        self.command_pubs = [
            self.create_publisher(JointState, command_topic, reliable),
            self.create_publisher(JointState, command_topic, sensor),
        ]
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, self.image_topic, self._on_image, image_qos)
        self.create_subscription(JointState, state_topic, self._on_joint_states, sensor)
        self.create_subscription(JointState, state_topic, self._on_joint_states, reliable)

        self._lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._latest_image = None
        self.joint_names = None
        self.arm_index = None
        self.q_meas = None
        self.q_pub = None
        self.q_interp_start = None
        self.q_interp_goal = None
        self.interp_elapsed = 0.0
        self.dt = 1.0 / publish_rate
        self._stop = threading.Event()
        self._homed = False
        self._steps = 0
        self._waiting = 0
        self._run_t0 = time.monotonic()
        self._log_path, self._log_fp, self._latest_fp = self._open_run_log()

        self.create_timer(self.dt, self._publish_command)
        self._worker = threading.Thread(target=self._infer_loop, daemon=True)
        self._worker.start()
        self.get_logger().info(
            f'Closed loop: {self.image_topic} + {state_topic} -> {self.server_url} '
            f'instr="{self.instruction}" clip {self.max_trans_m*100:.1f} cm, '
            f'delta_frame={self.delta_frame}, '
            f'publish {publish_rate:.0f} Hz, interp {self.interp_duration:.2f}s, '
            f'no forced home, max {self.max_steps} steps  '
            f'log={self._log_path}'
        )
        self._flog(
            f'start instr={self.instruction!r} unnorm={self.unnorm_key} '
            f'clip_trans_m={self.max_trans_m} clip_rot_rad={self.max_rot_rad} '
            f'delta_frame={self.delta_frame} '
            f'interp_s={self.interp_duration} max_steps={self.max_steps} '
            f'force_home=False'
        )

    def _open_run_log(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        path = LOG_DIR / f'openvla_closed_loop_{stamp}.log'
        fp = path.open('w', encoding='utf-8')
        latest = LATEST_LOG.open('w', encoding='utf-8')
        return path, fp, latest

    def _flog(self, line: str):
        stamp = datetime.now().strftime('%H:%M:%S')
        elapsed = time.monotonic() - self._run_t0
        text = f'{stamp} t={elapsed:7.2f}s  {line}\n'
        with self._log_lock:
            for fp in (self._log_fp, self._latest_fp):
                if fp.closed:
                    continue
                fp.write(text)
                fp.flush()

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

    def _on_image(self, msg: Image):
        try:
            image = image_to_rgb(msg)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return
        with self._lock:
            self._latest_image = image

    def _on_joint_states(self, msg: JointState):
        if not msg.name or not msg.position:
            return
        count = min(len(msg.name), len(msg.position))
        names = list(msg.name[:count])
        pos = list(msg.position[:count])
        arm_index, q = self._arm_q(names, pos)
        if q is None:
            return
        locked = False
        with self._lock:
            if self.joint_names is None:
                self.joint_names = names
                self.arm_index = arm_index
                self.q_pub = q.copy()
                self.q_interp_start = q.copy()
                self.q_interp_goal = q.copy()
                self.interp_elapsed = self.interp_duration
                self._homed = True
                locked = True
            self.q_meas = q
        if locked:
            self.get_logger().info(
                f'Locked joints {names}. Keeping current pose '
                f'(deg)={np.round(np.degrees(q), 2).tolist()}; starting OpenVLA.'
            )
            self._flog(
                f'locked_joints names={names} q_meas_deg={np.round(np.degrees(q), 2).tolist()} '
                f'force_home=False start_openvla'
            )

    def _set_interp_goal(self, q_goal: np.ndarray):
        """Start a new joint-space lerp from the current published command."""
        start = self.q_pub if self.q_pub is not None else q_goal
        self.q_interp_start = start.copy()
        self.q_interp_goal = q_goal.copy()
        self.interp_elapsed = 0.0

    def _infer_loop(self):
        while not self._stop.is_set() and rclpy.ok():
            if self._steps >= self.max_steps:
                time.sleep(0.2)
                continue
            with self._lock:
                homed = self._homed
                image = None if self._latest_image is None else self._latest_image.copy()
                q = None if self.q_meas is None else self.q_meas.copy()
            if not homed:
                time.sleep(0.05)
                continue
            if image is None or q is None:
                self._waiting += 1
                if self._waiting % 10 == 1:
                    self.get_logger().info(
                        'Waiting for /rgb and /joint_states. Is Isaac playing and OpenVLA up?'
                    )
                time.sleep(0.2)
                continue
            self._waiting = 0
            try:
                body = dumps_numpy({
                    'image': image,
                    'instruction': self.instruction,
                    'unnorm_key': self.unnorm_key,
                })
                response = requests.post(
                    self.server_url,
                    data=body,
                    headers={'Content-Type': 'application/json'},
                    timeout=120,
                )
            except requests.RequestException as exc:
                self.get_logger().error(f'OpenVLA request failed: {exc}')
                self._flog(f'error request {exc}')
                time.sleep(1.0)
                continue
            if response.status_code != 200:
                self.get_logger().error(f'HTTP {response.status_code}: {response.text[:300]}')
                self._flog(f'error http {response.status_code} {response.text[:200]!r}')
                time.sleep(1.0)
                continue
            action = np.asarray(loads_numpy(response.text), dtype=np.float64).reshape(-1)
            if action.size != 7:
                self.get_logger().error(f'Expected 7 values, got {action.tolist()}')
                self._flog(f'error bad_action {action.tolist()}')
                continue
            dp = _clip_vec(action[:3], self.max_trans_m)
            drpy = np.clip(action[3:6], -self.max_rot_rad, self.max_rot_rad)
            T = fk(q)
            if self.delta_frame == 'ee':
                dp_world = T[:3, :3] @ dp
            else:
                dp_world = dp
            T_goal = T.copy()
            T_goal[:3, 3] = T[:3, 3] + dp_world
            T_goal[:3, :3] = _rpy(drpy[0], drpy[1], drpy[2])[:3, :3] @ T[:3, :3]
            try:
                q_goal = ik(q, T_goal)
            except RuntimeError as exc:
                self.get_logger().warning(f'IK skip: {exc}')
                self._flog(f'ik_skip {exc} raw_xyz={np.round(action[:3], 4).tolist()}')
                continue
            with self._lock:
                self._set_interp_goal(q_goal)
                self._steps += 1
                step = self._steps
            p0, p1 = T[:3, 3], fk(q_goal)[:3, 3]
            line = (
                f'step {step}/{self.max_steps} '
                f'raw_xyz={np.round(action[:3], 4).tolist()} '
                f'clip_xyz={np.round(dp, 4).tolist()} '
                f'world_xyz={np.round(dp_world, 4).tolist()} '
                f'raw_rpy={np.round(action[3:6], 4).tolist()} '
                f'clip_rpy={np.round(drpy, 4).tolist()} '
                f'gripper={action[6]:.3f} '
                f'EE {np.round(p0, 4).tolist()} -> {np.round(p1, 4).tolist()} '
                f'q_meas_deg={np.round(np.degrees(q), 2).tolist()}'
            )
            self.get_logger().info(line)
            self._flog(line)
            if step >= self.max_steps:
                self.get_logger().info('Max steps reached, holding last command.')
                self._flog('max_steps reached, holding last command')

    def _publish_command(self):
        with self._lock:
            if self.joint_names is None or self.q_interp_goal is None:
                return
            self.interp_elapsed = min(self.interp_elapsed + self.dt, self.interp_duration)
            alpha = self.interp_elapsed / self.interp_duration
            q = (1.0 - alpha) * self.q_interp_start + alpha * self.q_interp_goal
            self.q_pub = q
            names = list(self.joint_names)
            arm_index = list(self.arm_index)
        msg = JointState()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        positions = [0.0] * len(names)
        for i, qi in zip(arm_index, q):
            positions[i] = float(qi)
        msg.position = positions
        for pub in self.command_pubs:
            pub.publish(msg)

    def destroy_node(self):
        self._stop.set()
        try:
            self._flog('stop')
            with self._log_lock:
                self._log_fp.close()
                self._latest_fp.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ClosedLoop()
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
