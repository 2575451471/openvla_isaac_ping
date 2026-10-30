#!/usr/bin/env python3
"""Closed loop: /rgb -> OpenVLA /act -> clip -> IK -> interpolate -> /joint_command.

OpenVLA returns a 7-vector of EE deltas. Gripper is dropped. Translation is
capped at about 5 cm per step. Translation frame is selectable:
  delta_frame=base  add (dx,dy,dz) in the UR kinematic base (old behavior)
  delta_frame=ee    add R_ee @ (dx,dy,dz) in the current end-effector frame

Starts from the arm's current measured joint pose (no forced home). Inference
is variable. Joint commands publish at 50 Hz. Each interpolation must finish
and measured joints must reach its target before a fresh image is requested.

Run:
    source /opt/ros/humble/setup.bash
    python3 /home/dt/openvla/shenle_projects/scripts/ros/openvla_closed_loop.py
    python3 .../ros/openvla_closed_loop.py --ros-args -p delta_frame:=base -p max_steps:=20
"""

import json
import sys
import threading
import time
from collections import deque
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

    def __init__(self, *, node_name='openvla_closed_loop', parameter_overrides=None):
        super().__init__(node_name, parameter_overrides=parameter_overrides)
        self.declare_parameter('image_topic', '/rgb')
        self.declare_parameter('state_topic', '/joint_states')
        self.declare_parameter('command_topic', '/joint_command')
        self.declare_parameter('server_url', 'http://127.0.0.1:8000/act')
        self.declare_parameter('instruction', 'push the red cube to the right')
        self.declare_parameter('unnorm_key', 'ur10e_push_sim')
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('interp_duration', 0.75)
        self.declare_parameter('max_trans_m', 0.05)
        self.declare_parameter('max_rot_rad', 0.10)
        self.declare_parameter('max_steps', 300)
        self.declare_parameter('delta_frame', 'ee')
        self.declare_parameter('joint_tolerance_rad', 0.001)
        self.declare_parameter('motion_timeout_s', 10.0)
        self.declare_parameter('max_observation_age_s', 1.0)
        self.declare_parameter('clip_actions', True)
        self.declare_parameter('max_ik_backtracks', 0)
        self.declare_parameter('ik_position_tol_m', 0.003)
        self.declare_parameter('ik_rotation_tol_deg', 20.0)
        self.declare_parameter('ik_require_rotation', False)
        self.declare_parameter('settle_duration_s', 0.10)
        self.declare_parameter('settle_velocity_rad_s', 0.01)

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
        self.joint_tolerance = float(self.get_parameter('joint_tolerance_rad').value)
        self.motion_timeout = float(self.get_parameter('motion_timeout_s').value)
        self.max_observation_age = float(self.get_parameter('max_observation_age_s').value)
        self.clip_actions = bool(self.get_parameter('clip_actions').value)
        self.max_ik_backtracks = int(self.get_parameter('max_ik_backtracks').value)
        if not 0 <= self.max_ik_backtracks <= 3:
            raise ValueError('max_ik_backtracks must be between 0 and 3')
        self.ik_position_tol = float(self.get_parameter('ik_position_tol_m').value)
        self.ik_rotation_tol = float(np.radians(self.get_parameter('ik_rotation_tol_deg').value))
        self.ik_require_rotation = bool(self.get_parameter('ik_require_rotation').value)
        if self.ik_position_tol <= 0 or self.ik_rotation_tol <= 0:
            raise ValueError('IK position and rotation tolerances must be positive')
        self.settle_duration = float(self.get_parameter('settle_duration_s').value)
        self.settle_velocity = float(self.get_parameter('settle_velocity_rad_s').value)
        command_topic = str(self.get_parameter('command_topic').value)
        state_topic = str(self.get_parameter('state_topic').value)
        if publish_rate <= 0.0 or self.max_trans_m <= 0.0 or self.interp_duration <= 0.0:
            raise ValueError('publish_rate, max_trans_m, and interp_duration must be > 0')
        if self.delta_frame not in ('base', 'ee'):
            raise ValueError("delta_frame must be 'base' or 'ee'")
        values = (publish_rate, self.interp_duration, self.max_trans_m, self.max_rot_rad,
                  self.joint_tolerance, self.motion_timeout, self.max_observation_age,
                  self.settle_duration, self.settle_velocity)
        if not all(np.isfinite(value) and value > 0 for value in values) or self.max_steps <= 0:
            raise ValueError('Rates, limits, tolerances, timeouts, and max_steps must be positive and finite')

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
        self._last_image_at = 0.0
        self._last_joint_at = 0.0
        self._last_joint_stamp_ns = -1
        self._joint_velocity_max = float('inf')
        self._reported_joint_velocity_max = None
        self._joint_history = deque()
        self._joint_history_clock = None
        self._goal_stable_since = None
        self.joint_names = None
        self.arm_index = None
        self.q_meas = None
        self.q_pub = None
        self.q_interp_start = None
        self.q_interp_goal = None
        self.interp_elapsed = 0.0
        self.dt = 1.0 / publish_rate
        self._stop = threading.Event()
        self._motion_done = threading.Event()
        self._motion_done.set()
        self._motion_finished_at = 0.0
        self._finished = threading.Event()
        self._worker_error = None
        self._last_motion_failure = None
        self._homed = False
        self._steps = 0
        self._waiting = 0
        self._run_t0 = time.monotonic()
        self._log_path, self._log_fp, self._latest_fp = self._open_run_log()

        self.create_timer(self.dt, self._publish_command)
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
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
            f'force_home=False sequential_motion=True clip_actions={self.clip_actions} '
            f'joint_tolerance_rad={self.joint_tolerance} settle_s={self.settle_duration} '
            f'settle_velocity_rad_s={self.settle_velocity} settle_velocity_source=position_window '
            f'elbow_branch=positive'
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
            self._last_image_at = time.monotonic()

    def _on_joint_states(self, msg: JointState):
        if not msg.name or not msg.position:
            return
        count = min(len(msg.name), len(msg.position))
        names = list(msg.name[:count])
        pos = list(msg.position[:count])
        arm_index, q = self._arm_q(names, pos)
        if q is None or not np.isfinite(q).all():
            return
        locked = False
        with self._lock:
            stamp = msg.header.stamp
            stamp_ns = int(stamp.sec) * 1000000000 + int(stamp.nanosec)
            if stamp_ns > 0 and stamp_ns <= self._last_joint_stamp_ns:
                return  # Both QoS subscriptions can deliver the same or an older sample.
            now = time.monotonic()
            velocity = None
            if len(msg.velocity) >= count:
                _, velocity = self._arm_q(names, list(msg.velocity[:count]))
            if velocity is not None and np.isfinite(velocity).all():
                self._reported_joint_velocity_max = float(np.max(np.abs(velocity)))
            else:
                self._reported_joint_velocity_max = None
            # The simulation can keep a nonzero velocity field while all
            # reported positions remain constant for seconds. Measure settling
            # from a full position window; retain raw velocity for diagnostics.
            clock = 'stamp' if stamp_ns > 0 else 'arrival'
            sample_time = stamp_ns / 1e9 if stamp_ns > 0 else now
            if (clock != self._joint_history_clock or self._joint_history
                    and sample_time <= self._joint_history[-1][0]):
                self._joint_history.clear()
            self._joint_history_clock = clock
            self._joint_history.append((sample_time, q.copy()))
            cutoff = sample_time - self.settle_duration
            while len(self._joint_history) > 2 and self._joint_history[1][0] <= cutoff:
                self._joint_history.popleft()
            span = sample_time - self._joint_history[0][0]
            if span + 1e-9 >= self.settle_duration:
                positions = np.asarray([sample[1] for sample in self._joint_history])
                self._joint_velocity_max = float(np.max(np.ptp(positions, axis=0)) / span)
            else:
                self._joint_velocity_max = float('inf')
            if stamp_ns > 0:
                self._last_joint_stamp_ns = stamp_ns
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
            self._last_joint_at = now
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
        self._motion_done.clear()
        self._goal_stable_since = None

    def _plan_and_submit(self, goal):
        """Keep one Cartesian target; retry IK seeds if feedback changes slightly."""
        for attempt in range(3):
            with self._lock:
                if time.monotonic() - self._last_joint_at > self.max_observation_age:
                    raise RuntimeError('Joint feedback became stale before IK')
                seed = self.q_meas.copy()
            started = time.monotonic()
            q_goal = ik(
                seed, goal,
                position_limit=self.ik_position_tol,
                rotation_limit=self.ik_rotation_tol,
                require_rotation=self.ik_require_rotation,
            )
            if np.asarray(q_goal).shape != (6,) or not np.isfinite(q_goal).all():
                raise ValueError('IK returned invalid joints')
            actual = fk(q_goal)
            rotation_error = np.arccos(np.clip((np.trace(actual[:3, :3] @ goal[:3, :3].T) - 1) / 2, -1, 1))
            position_error = float(np.linalg.norm(actual[:3, 3] - goal[:3, 3]))
            joint_jump = float(np.max(np.abs(q_goal - seed)))
            self._flog('ik_candidate ' + json.dumps({
                'position_error_mm': position_error * 1000,
                'rotation_error_deg': float(np.degrees(rotation_error)),
                'joint_jump_deg': float(np.degrees(joint_jump)),
                'seed_rad': seed.tolist(), 'candidate_rad': np.asarray(q_goal).tolist(),
            }))
            if position_error > self.ik_position_tol or joint_jump > np.radians(20):
                raise RuntimeError(
                    'IK position error or joint jump exceeds limit'
                    + f'; position={position_error * 1000:.3f}mm'
                    + f'; rotation={np.degrees(rotation_error):.3f}deg'
                    + f'; joint_step={np.degrees(joint_jump):.3f}deg'
                )
            if self.ik_require_rotation and rotation_error > self.ik_rotation_tol:
                raise RuntimeError(
                    'IK rotation error exceeds limit'
                    + f'; position={position_error * 1000:.3f}mm'
                    + f'; rotation={np.degrees(rotation_error):.3f}deg'
                    + f'; joint_step={np.degrees(joint_jump):.3f}deg'
                )
            with self._lock:
                if time.monotonic() - self._last_joint_at > self.max_observation_age:
                    raise RuntimeError('Joint feedback became stale during IK')
                drift = float(np.max(np.abs(self.q_meas - seed)))
                if drift <= self.joint_tolerance:
                    self.interp_duration = self._step_duration()
                    self._set_interp_goal(q_goal)
                    self._steps += 1
                    return seed, q_goal, self._steps
            self._flog(f'ik_replan attempt={attempt+1} seed_drift_rad={drift:.7f} '
                       f'ik_seconds={time.monotonic()-started:.4f}')
        raise RuntimeError(f'Joint state kept changing during 3 IK attempts; last drift={drift:.7f}rad')

    def _step_duration(self):
        return self.interp_duration

    def _request_action(self, image):
        response = requests.post(
            self.server_url,
            data=dumps_numpy({'image': image, 'instruction': self.instruction, 'unnorm_key': self.unnorm_key}),
            headers={'Content-Type': 'application/json'}, timeout=120,
        )
        response.raise_for_status()
        return np.asarray(loads_numpy(response.text), dtype=np.float64).reshape(-1)

    def _motion_status(self):
        with self._lock:
            now = time.monotonic()
            age = now - self._last_joint_at
            error = (None if self.q_meas is None or self.q_interp_goal is None
                     else np.abs(self.q_meas - self.q_interp_goal))
            max_error = None if error is None else float(np.max(error))
            settled = (0.0 if self._goal_stable_since is None
                       else max(0.0, now - self._goal_stable_since))
            blocked = []
            if self.interp_elapsed < self.interp_duration:
                blocked.append('interpolation_incomplete')
            if age > self.max_observation_age:
                blocked.append('joint_feedback_stale')
            if max_error is None or max_error > self.joint_tolerance:
                blocked.append('joint_position_error')
            if not np.isfinite(self._joint_velocity_max):
                blocked.append('joint_position_window_incomplete')
            elif self._joint_velocity_max > self.settle_velocity:
                blocked.append('joint_velocity')
            if settled < self.settle_duration:
                blocked.append('settling_duration')
            return {
                'step': self._steps, 'blocked_by': blocked,
                'interp_elapsed_s': self.interp_elapsed, 'interp_duration_s': self.interp_duration,
                'joint_feedback_age_s': age, 'max_observation_age_s': self.max_observation_age,
                'target_joint_rad': None if self.q_interp_goal is None else self.q_interp_goal.tolist(),
                'measured_joint_rad': None if self.q_meas is None else self.q_meas.tolist(),
                'joint_error_rad': None if error is None else error.tolist(),
                'max_joint_error_rad': max_error, 'joint_tolerance_rad': self.joint_tolerance,
                'max_joint_velocity_rad_s': (float(self._joint_velocity_max)
                                            if np.isfinite(self._joint_velocity_max) else None),
                'settle_velocity_source': 'position_window_range_over_time',
                'reported_joint_velocity_max_rad_s': self._reported_joint_velocity_max,
                'settle_velocity_rad_s': self.settle_velocity,
                'settled_s': settled, 'required_settle_s': self.settle_duration,
            }

    def _wait_for_motion(self):
        deadline = time.monotonic() + self.interp_duration + self.motion_timeout
        next_status_at = time.monotonic() + 1.0
        while not self._stop.is_set() and rclpy.ok():
            if self._motion_done.wait(.02):
                return True
            now = time.monotonic()
            if now >= next_status_at:
                self._flog('motion_wait ' + json.dumps(self._motion_status()))
                next_status_at = now + 1.0
            if now > deadline:
                status = self._motion_status()
                self._last_motion_failure = status
                self._flog('motion_timeout ' + json.dumps(status))
                raise RuntimeError('Motion timeout: ' + ','.join(status['blocked_by'])
                                   + f'; joint_error={status["max_joint_error_rad"]}rad'
                                   + f'; velocity={status["max_joint_velocity_rad_s"]}rad/s'
                                   + f'; feedback_age={status["joint_feedback_age_s"]:.3f}s')
        return False

    def _run_worker(self):
        try:
            self._infer_loop()
        except Exception as error:
            self._worker_error = str(error)
            self.get_logger().error('Control stopped: ' + str(error))
            self._flog('control_stopped ' + str(error))
            with self._lock:
                if self.q_meas is not None:
                    self.q_interp_start = self.q_meas.copy()
                    self.q_interp_goal = self.q_meas.copy()
                    self.q_pub = self.q_meas.copy()
                    self.interp_elapsed = self.interp_duration
            self._stop.set()
        finally:
            self._finished.set()

    def _infer_loop(self):
        while not self._stop.is_set() and rclpy.ok():
            if not self._wait_for_motion():
                return
            if self._steps >= self.max_steps:
                self.get_logger().info('Max steps completed and measured target reached; holding last command.')
                self._flog('max_steps completed; final measured target reached')
                return
            with self._lock:
                homed = self._homed
                now = time.monotonic()
                fresh = (now - self._last_joint_at <= self.max_observation_age
                         and now - self._last_image_at <= self.max_observation_age
                         and self._last_image_at >= self._motion_finished_at)
                image = None if self._latest_image is None or not fresh else self._latest_image.copy()
                q = None if self.q_meas is None or not fresh else self.q_meas.copy()
            if not homed:
                time.sleep(0.05)
                continue
            if image is None or q is None:
                self._waiting += 1
                if self._waiting % 50 == 1:
                    self.get_logger().info(
                        'Waiting for /rgb and /joint_states. Is Isaac playing and OpenVLA up?'
                    )
                time.sleep(0.02)
                continue
            self._waiting = 0
            request_start = time.monotonic()
            try:
                action = np.asarray(self._request_action(image), dtype=float).reshape(-1)
            except requests.RequestException as exc:
                self.get_logger().error(f'OpenVLA request failed: {exc}')
                self._flog(f'error request {exc}')
                time.sleep(1.0)
                continue
            if self._stop.is_set() or not rclpy.ok():
                return
            if action.size != 7 or not np.isfinite(action).all():
                raise ValueError(f'Expected 7 finite action values, got {action.tolist()}')
            # Re-read measured joints after inference; never seed IK with pre-request state.
            with self._lock:
                if time.monotonic() - self._last_joint_at > self.max_observation_age:
                    raise RuntimeError('Joint feedback became stale during inference')
                q = self.q_meas.copy()
            clipped_dp = _clip_vec(action[:3], self.max_trans_m) if self.clip_actions else action[:3].copy()
            clipped_drpy = np.clip(action[3:6], -self.max_rot_rad, self.max_rot_rad) if self.clip_actions else action[3:6].copy()
            T = fk(q)
            self._flog('model_action ' + json.dumps({
                'step': self._steps + 1, 'action': action.tolist(), 'seed_rad': q.tolist(),
            }))
            # Each candidate is a fraction of the same action from the same pose.
            # Failed plans never publish a motion or accumulate a Cartesian delta.
            for backtrack in range(self.max_ik_backtracks + 1):
                action_scale = 0.5 ** backtrack
                dp, drpy = clipped_dp * action_scale, clipped_drpy * action_scale
                dp_world = T[:3, :3] @ dp if self.delta_frame == 'ee' else dp
                T_goal = T.copy()
                T_goal[:3, 3] = T[:3, 3] + dp_world
                T_goal[:3, :3] = _rpy(drpy[0], drpy[1], drpy[2])[:3, :3] @ T[:3, :3]
                try:
                    q, q_goal, step = self._plan_and_submit(T_goal)
                except RuntimeError as exc:
                    ik_failure = (str(exc).startswith('IK failed on preferred elbow branch')
                                  or str(exc).startswith('IK rotation error exceeds')
                                  or str(exc).startswith('IK position error'))
                    if ik_failure and backtrack < self.max_ik_backtracks:
                        detail = {'step': self._steps + 1, 'failed_scale': action_scale,
                                  'next_scale': action_scale / 2, 'reason': str(exc)}
                        self.get_logger().warning('IK backtrack: ' + json.dumps(detail))
                        self._flog('ik_backtrack ' + json.dumps(detail))
                        continue
                    self.get_logger().warning(f'IK skip: {exc}')
                    self._flog(f'ik_stop {exc} raw_xyz={np.round(action[:3], 4).tolist()}')
                    raise RuntimeError(f'Action {self._steps+1} planning stopped: {exc}') from exc
                if action_scale < 1:
                    self._flog('ik_action_scaled ' + json.dumps({'step': step, 'scale': action_scale}))
                break
            p0, p1 = fk(q)[:3, 3], fk(q_goal)[:3, 3]
            line = (
                f'step {step}/{self.max_steps} '
                f'action_scale={action_scale:.3f} '
                f'request_s={time.monotonic() - request_start:.3f} interp_s={self.interp_duration:.3f} '
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

    def _publish_command(self):
        with self._lock:
            if self.joint_names is None or self.q_interp_goal is None:
                return
            self.interp_elapsed = min(self.interp_elapsed + self.dt, self.interp_duration)
            alpha = self.interp_elapsed / self.interp_duration
            q = (1.0 - alpha) * self.q_interp_start + alpha * self.q_interp_goal
            self.q_pub = q
            now = time.monotonic()
            if not self._motion_done.is_set():
                stable = (alpha >= 1.0 and self.q_meas is not None
                          and now - self._last_joint_at <= self.max_observation_age
                          and np.max(np.abs(self.q_meas - self.q_interp_goal)) <= self.joint_tolerance
                          and self._joint_velocity_max <= self.settle_velocity)
                if stable:
                    if self._goal_stable_since is None:
                        self._goal_stable_since = now
                    if now - self._goal_stable_since >= self.settle_duration:
                        self._motion_finished_at = now
                        self._motion_done.set()
                else:
                    self._goal_stable_since = None
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
