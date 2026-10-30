"""Exercise production sequencing methods without importing ROS or publishing."""
import ast
import json
from collections import deque
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest

import numpy as np


class Message:
    def __init__(self):
        self.header = SimpleNamespace(stamp=None)


class ControlSequenceTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "scripts/ros/openvla_closed_loop.py"
        tree = ast.parse(path.read_text())
        controller = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ClosedLoop")
        wanted = {"_set_interp_goal", "_step_duration", "_publish_command", "_wait_for_motion", "_infer_loop", "_run_worker", "_plan_and_submit", "_on_joint_states", "_arm_q", "_motion_status"}
        methods = [node for node in controller.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        self.assertEqual({node.name for node in methods}, wanted)
        test_class = ast.ClassDef(name="Controller", bases=[], keywords=[], body=methods, decorator_list=[])
        self.now = 100.0
        self.logs = []
        self.commands = []
        self.seeds = []

        def fk(q):
            result = np.eye(4)
            result[:3, 3] = q[:3]
            return result

        def ik(seed, goal, **kwargs):
            self.seeds.append(seed.copy())
            q = seed.copy()
            q[:3] = goal[:3, 3]
            return q

        self.namespace = {"np": np, "json": json, "time": SimpleNamespace(monotonic=lambda: self.now, sleep=lambda _: None),
                          "rclpy": SimpleNamespace(ok=lambda: True), "JointState": Message,
                          "requests": SimpleNamespace(RequestException=ConnectionError),
                          "ARM_JOINTS": list("abcdef"), "_joint_key": lambda name: name,
                          "fk": fk, "ik": ik, "_rpy": lambda *args: np.eye(4)}
        module = ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[]))
        exec(compile(module, str(path), "exec"), self.namespace)
        self.h = self.namespace["Controller"]()
        h = self.h
        h._lock = threading.Lock()
        h._stop, h._motion_done, h._finished = (threading.Event() for _ in range(3))
        h._motion_done.set()
        h._worker_error = None
        h._last_motion_failure = None
        h.q_meas, h.q_pub = np.zeros(6), np.zeros(6)
        h.q_interp_start, h.q_interp_goal = np.zeros(6), np.ones(6)*.01
        h.interp_elapsed, h.interp_duration, h.dt = 0.0, 1.0, .5
        h.joint_tolerance, h.motion_timeout, h.max_observation_age = .001, 10.0, 1.0
        h.settle_duration, h.settle_velocity = .1, .01
        h._goal_stable_since, h._joint_velocity_max = None, 0.
        h._reported_joint_velocity_max = None
        h._joint_history, h._joint_history_clock = deque(), None
        h._last_joint_at, h._last_image_at, h._motion_finished_at = self.now, self.now, 0.0
        h._last_joint_stamp_ns = -1
        h.joint_names, h.arm_index = list("abcdef"), list(range(6))
        h._steps, h.max_steps, h._waiting, h._homed = 0, 1, 0, True
        h._latest_image = np.zeros((1, 1, 3), dtype=np.uint8)
        h.clip_actions, h.delta_frame = False, "ee"
        h.max_ik_backtracks = 0
        h.ik_position_tol = .003
        h.ik_rotation_tol = np.radians(20)
        h.ik_require_rotation = False
        h.command_pubs = [SimpleNamespace(publish=lambda message: self.commands.append(message))]
        h.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: None))
        h.get_logger = lambda: SimpleNamespace(info=self.logs.append, error=self.logs.append, warning=self.logs.append)
        h._flog = self.logs.append
        h._request_action = lambda image: np.array([.001, 0, 0, 0, 0, 0, 1])

    def test_interpolation_end_alone_does_not_complete_motion(self):
        h = self.h
        h._motion_done.clear()
        h._publish_command()
        self.assertFalse(h._motion_done.is_set())
        h._publish_command()
        self.assertFalse(h._motion_done.is_set())
        np.testing.assert_allclose(self.commands[-1].position, h.q_interp_goal)
        h.q_meas = h.q_interp_goal.copy()
        h._publish_command()
        self.assertFalse(h._motion_done.is_set())
        self.now += .11
        h._last_joint_at = self.now
        h._publish_command()
        self.assertTrue(h._motion_done.is_set())
        self.assertEqual(h._motion_finished_at, self.now)

    def test_stale_measured_target_cannot_complete_motion(self):
        h = self.h
        h._motion_done.clear()
        h.interp_elapsed = h.interp_duration
        h.q_meas = h.q_interp_goal.copy()
        h._last_joint_at = self.now - 2
        h._publish_command()
        self.assertFalse(h._motion_done.is_set())

    def test_new_goal_clears_completion_gate(self):
        h = self.h
        h.q_pub = np.arange(6, dtype=float)
        h._set_interp_goal(np.ones(6))
        self.assertFalse(h._motion_done.is_set())
        np.testing.assert_array_equal(h.q_interp_start, np.arange(6))
        self.assertEqual(h.interp_elapsed, 0)

    def test_moving_through_target_does_not_count_as_settled(self):
        h = self.h
        h._motion_done.clear()
        h.interp_elapsed = h.interp_duration
        h.q_meas = h.q_interp_goal.copy()
        h._joint_velocity_max = .02
        h._publish_command()
        self.now += .2
        h._last_joint_at = self.now
        h._publish_command()
        self.assertFalse(h._motion_done.is_set())
        self.assertIsNone(h._goal_stable_since)

    def test_settling_timer_restarts_after_target_excursion(self):
        h = self.h
        h._motion_done.clear()
        h.interp_elapsed = h.interp_duration
        h.q_meas = h.q_interp_goal.copy()
        h._publish_command()
        h.q_meas[0] += .002
        self.now += .05
        h._last_joint_at = self.now
        h._publish_command()
        self.assertIsNone(h._goal_stable_since)
        h.q_meas = h.q_interp_goal.copy()
        self.now += .06
        h._last_joint_at = self.now
        h._publish_command()
        self.assertFalse(h._motion_done.is_set())

    def test_seed_drift_replans_same_cartesian_target_without_advancing_action(self):
        h = self.h
        goals = []
        def ik(seed, goal, **kwargs):
            goals.append(goal.copy())
            if len(goals) == 1:
                h.q_meas[0] += .002
            result = seed.copy()
            result[:3] = goal[:3, 3]
            return result
        self.namespace["ik"] = ik
        goal = np.eye(4)
        goal[0, 3] = .005
        seed, result, step = h._plan_and_submit(goal)
        self.assertEqual(len(goals), 2)
        np.testing.assert_array_equal(goals[0], goals[1])
        self.assertEqual(step, 1)
        self.assertEqual(h._steps, 1)
        self.assertEqual(seed[0], .002)
        self.assertEqual(result[0], .005)

    def test_persistent_seed_drift_stops_after_three_attempts(self):
        h = self.h
        calls = []
        def ik(seed, goal, **kwargs):
            calls.append(1)
            h.q_meas[0] += .002
            result = seed.copy()
            result[:3] = goal[:3, 3]
            return result
        self.namespace["ik"] = ik
        with self.assertRaisesRegex(RuntimeError, "3 IK attempts"):
            h._plan_and_submit(np.eye(4))
        self.assertEqual(len(calls), 3)
        self.assertEqual(h._steps, 0)

    def test_duplicate_and_out_of_order_feedback_is_ignored(self):
        h = self.h
        def message(stamp, value):
            return SimpleNamespace(name=list("abcdef"), position=[value]*6, velocity=[.005]*6,
                                   header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=stamp)))
        h._on_joint_states(message(100, .001))
        h._on_joint_states(message(100, .5))
        h._on_joint_states(message(50, .6))
        np.testing.assert_allclose(h.q_meas, .001)
        self.assertEqual(h._reported_joint_velocity_max, .005)
        h._on_joint_states(message(200, .002))
        np.testing.assert_allclose(h.q_meas, .002)

    def test_feedback_velocity_is_estimated_if_not_reported(self):
        h = self.h
        h._last_joint_stamp_ns = 1000000000
        h._joint_history_clock = 'stamp'
        h._joint_history.append((1., h.q_meas.copy()))
        message = SimpleNamespace(name=list("abcdef"), position=[.001]*6, velocity=[],
                                  header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=100000000)))
        h._on_joint_states(message)
        self.assertAlmostEqual(h._joint_velocity_max, .01)

    def test_constant_positions_settle_despite_persistent_reported_velocity(self):
        h = self.h
        h._motion_done.clear()
        h.interp_elapsed = h.interp_duration
        h.q_interp_goal = np.zeros(6)
        for stamp in (0, 50000000, 100000000):
            m = SimpleNamespace(name=list('abcdef'), position=[0.]*6, velocity=[.0106]*6,
                                header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=stamp)))
            h._on_joint_states(m)
        self.assertEqual(h._joint_velocity_max, 0.)
        self.assertEqual(h._reported_joint_velocity_max, .0106)
        h._publish_command()
        self.assertFalse(h._motion_done.is_set())
        self.now += .11
        h._last_joint_at = self.now
        h._publish_command()
        self.assertTrue(h._motion_done.is_set())

    def test_position_window_detects_oscillation_even_with_equal_endpoints(self):
        h = self.h
        for stamp, position in ((0, 0.), (50000000, .002), (100000000, 0.)):
            m = SimpleNamespace(name=list('abcdef'), position=[position]*6, velocity=[0.]*6,
                                header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=stamp)))
            h._on_joint_states(m)
        self.assertGreater(h._joint_velocity_max, h.settle_velocity)

    def test_position_window_requires_full_duration_and_one_clock(self):
        h = self.h
        for sec, ns in ((0, 0), (1, 0), (1, 50000000)):
            m = SimpleNamespace(name=list('abcdef'), position=[0.]*6, velocity=[0.]*6,
                                header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=ns)))
            h._on_joint_states(m)
            self.assertTrue(np.isinf(h._joint_velocity_max))
        self.assertEqual(len(h._joint_history), 2)

    def test_incomplete_position_window_has_valid_json_diagnostics(self):
        self.h._joint_velocity_max = float('inf')
        status = self.h._motion_status()
        self.assertIn('joint_position_window_incomplete', status['blocked_by'])
        self.assertIsNone(status['max_joint_velocity_rad_s'])
        json.dumps(status, allow_nan=False)

    def test_timeout_preserves_failed_target_before_hold_overwrites_it(self):
        h = self.h
        h._motion_done.clear()
        h.interp_elapsed = h.interp_duration
        target = h.q_interp_goal.copy()
        def wait(_):
            self.now += 20
            h._last_joint_at = self.now
            return False
        h._motion_done = SimpleNamespace(wait=wait)
        with self.assertRaisesRegex(RuntimeError, 'joint_position_error'):
            h._wait_for_motion()
        h.q_interp_goal = h.q_meas.copy()
        np.testing.assert_array_equal(h._last_motion_failure['target_joint_rad'], target)
        self.assertEqual(h._last_motion_failure['max_joint_error_rad'], .01)

    def test_diagnostics_separate_velocity_gate_from_target_error(self):
        h = self.h
        h.interp_elapsed = h.interp_duration
        h.q_meas = h.q_interp_goal.copy()
        h._joint_velocity_max = .02
        status = h._motion_status()
        self.assertIn('joint_velocity', status['blocked_by'])
        self.assertNotIn('joint_position_error', status['blocked_by'])
        self.assertEqual(status['max_joint_error_rad'], 0)

    def test_ik_uses_measured_state_after_action_request(self):
        h = self.h
        latest = np.ones(6)*.0003
        def request(image):
            h.q_meas = latest.copy()
            return np.array([.001, 0, 0, 0, 0, 0, 1])
        h._request_action = request
        def log(line):
            if line.startswith("step "):
                h._stop.set()
        h._flog = log
        h._infer_loop()
        np.testing.assert_array_equal(self.seeds[0], latest)
        self.assertEqual(h._steps, 1)
        self.assertFalse(h._motion_done.is_set())

    def test_image_from_before_motion_completion_is_not_reused(self):
        h = self.h
        h._motion_finished_at = self.now + .1
        h._request_action = lambda image: self.fail("Old image was sent for inference")
        self.namespace["time"].sleep = lambda _: h._stop.set()
        h._infer_loop()
        self.assertEqual(h._steps, 0)

    def test_unfinished_motion_prevents_next_action_request(self):
        h = self.h
        h._wait_for_motion = lambda: False
        h._request_action = lambda image: self.fail("Action requested before motion finished")
        h._infer_loop()
        self.assertEqual(h._steps, 0)

    def test_nonfinite_action_halts_and_holds_measured_joints(self):
        h = self.h
        h._request_action = lambda image: [np.nan]*7
        h._run_worker()
        self.assertIn("finite", h._worker_error)
        self.assertTrue(h._stop.is_set())
        self.assertTrue(h._finished.is_set())
        np.testing.assert_array_equal(h.q_interp_goal, h.q_meas)

    def test_final_step_is_not_finished_until_measured_target_reached(self):
        h = self.h
        h._steps = h.max_steps
        h._wait_for_motion = lambda: False
        h._infer_loop()
        self.assertFalse(any("max_steps completed" in line for line in self.logs))
        h._wait_for_motion = lambda: True
        h._infer_loop()
        self.assertTrue(any("max_steps completed" in line for line in self.logs))

    def test_ik_backtracking_scales_translation_and_rotation_without_accumulation(self):
        h = self.h
        h.max_ik_backtracks = 2
        h._wait_for_motion = lambda: True
        action = np.array([.002, -.001, .0008, .03, -.02, .01, 1.])
        h._request_action = lambda image: action.copy()
        goals, rotations = [], []
        def rpy(*angles):
            rotations.append(angles)
            return np.eye(4)
        def ik(seed, goal, **kwargs):
            goals.append(goal.copy())
            if np.linalg.norm(goal[:3, 3]) > .0012:
                raise RuntimeError("IK failed on preferred elbow branch 1: test goal")
            result = seed.copy()
            result[:3] = goal[:3, 3]
            return result
        self.namespace["ik"], self.namespace["_rpy"] = ik, rpy
        h._run_worker()
        self.assertIsNone(h._worker_error)
        self.assertEqual(h._steps, 1)
        self.assertEqual(len(goals), 2)
        np.testing.assert_allclose(goals[0][:3, 3], action[:3])
        np.testing.assert_allclose(goals[1][:3, 3], action[:3] * .5)
        np.testing.assert_allclose(rotations[0], action[3:6])
        np.testing.assert_allclose(rotations[1], action[3:6] * .5)
        np.testing.assert_allclose(h.q_interp_goal[:3], action[:3] * .5)

    def test_ik_backtracking_exhaustion_stops_without_a_motion(self):
        h = self.h
        h.max_ik_backtracks = 2
        calls = []
        def ik(seed, goal, **kwargs):
            calls.append(goal.copy())
            raise RuntimeError("IK failed on preferred elbow branch 1: infeasible")
        self.namespace["ik"] = ik
        h._run_worker()
        self.assertEqual(len(calls), 3)
        self.assertEqual(h._steps, 0)
        self.assertTrue(h._stop.is_set())
        self.assertEqual(self.commands, [])

    def test_ik_backtracking_does_not_retry_stale_feedback(self):
        h = self.h
        h.max_ik_backtracks = 2
        calls = []
        def ik(seed, goal, **kwargs):
            calls.append(goal.copy())
            raise RuntimeError("Joint feedback became stale during IK")
        self.namespace["ik"] = ik
        h._run_worker()
        self.assertEqual(len(calls), 1)
        self.assertEqual(h._steps, 0)
        self.assertIn("stale", h._worker_error)

    def test_default_ik_planning_never_scales_recorded_actions(self):
        h = self.h
        self.assertEqual(h.max_ik_backtracks, 0)
        calls = []
        def ik(seed, goal, **kwargs):
            calls.append(goal.copy())
            raise RuntimeError("IK failed on preferred elbow branch 1: infeasible")
        self.namespace["ik"] = ik
        h._run_worker()
        self.assertEqual(len(calls), 1)
        self.assertEqual(h._steps, 0)



if __name__ == "__main__":
    unittest.main(verbosity=2)
