import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

path = Path(__file__).resolve().parents[1] / "scripts/isaac/replay_action_episode.py"
spec = importlib.util.spec_from_file_location("action_replay", path)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def rpy(roll, pitch, yaw):
    result = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    result[:2, :2] = [[c, -s], [s, c]]
    return result


def fk(q):
    result = rpy(0, 0, q[5])
    result[:3, 3] = q[:3]
    return result


def ik(seed, goal):
    result = seed.copy()
    result[:3] = goal[:3, 3]
    result[5] = np.arctan2(goal[1, 0], goal[0, 0])
    return result


KINEMATICS = {"fk": fk, "ik": ik, "_rpy": rpy}


class ActionReplayTests(unittest.TestCase):
    def test_ee_translation_uses_sequential_solved_orientation(self):
        initial = np.zeros(6)
        initial[5] = np.pi / 2
        actions = np.array([[.001, 0, 0, 0, 0, .1, 1], [.001, 0, 0, 0, 0, 0, 1]])
        q, report = replay.solve_actions(actions, initial, KINEMATICS)
        np.testing.assert_allclose(q[1, :3], [0, .001, 0], atol=1e-12)
        np.testing.assert_allclose(q[2, :3], [-.001*np.sin(.1), .001*(1+np.cos(.1)), 0], atol=1e-12)
        self.assertEqual(report["actions_solved"], 2)

    def test_future_recorded_joints_do_not_seed_solver(self):
        actions = np.array([[.001, 0, 0, 0, 0, 0, 1], [.001, 0, 0, 0, 0, 0, 1]])
        reference, _ = replay.solve_actions(actions, np.zeros(6), KINEMATICS)
        meta = {"pose_convention": replay.CONVENTION, "actions_pose_convention": replay.CONVENTION,
                "action_frame": "ee", "rotation_representation": "rpy_left_multiply",
                "action_units": ["m", "m", "m", "rad", "rad", "rad", "binary"],
                "gripper_open_value": 1}
        with tempfile.TemporaryDirectory() as temporary:
            np.save(Path(temporary) / "actions.npy", actions)
            planned, _ = replay.build_trajectory(temporary, meta, reference, np.arange(3), kinematics=KINEMATICS)
            changed = reference.copy()
            changed[1:, :3] += .0001
            replanned, _ = replay.build_trajectory(temporary, meta, changed, np.arange(3), kinematics=KINEMATICS)
            np.testing.assert_array_equal(planned, replanned)
            with self.assertRaisesRegex(ValueError, "raw_pose_v2"):
                replay.build_trajectory(temporary, {**meta, "actions_pose_convention": "legacy"}, reference, np.arange(3))

    def test_position_only_fallback_cannot_hide_orientation_error(self):
        def position_only(seed, goal):
            result = seed.copy()
            result[:3] = goal[:3, 3]
            return result
        with self.assertRaisesRegex(ValueError, "rejected IK"):
            replay.solve_actions([[0, 0, 0, 0, 0, .1, 1]], np.zeros(6), {**KINEMATICS, "ik": position_only})

    def test_alternate_joint_branch_is_rejected(self):
        def distant_branch(seed, goal):
            result = ik(seed, goal)
            result[4] += 2 * np.pi
            return result
        with self.assertRaisesRegex(ValueError, "joint_jump"):
            replay.solve_actions([[.001, 0, 0, 0, 0, 0, 1]], np.zeros(6), {**KINEMATICS, "ik": distant_branch})

    def test_nonfinite_actions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            replay.solve_actions([[np.nan, 0, 0, 0, 0, 0, 1]], np.zeros(6), KINEMATICS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
