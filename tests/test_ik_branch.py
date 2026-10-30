"""Verify the production IK's posture preference without importing ROS."""
import ast
import math
from pathlib import Path
import unittest

import numpy as np


path = Path(__file__).resolve().parents[1] / "scripts/ros/ee_nudge.py"
wanted = {"_xyz", "_rpy", "_rotz", "fk", "_rot_log", "_jacobian", "ik"}
nodes = [node for node in ast.parse(path.read_text()).body if
         isinstance(node, ast.FunctionDef) and node.name in wanted or
         isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_UR10E_ORIGIN" for t in node.targets)]
kinematics = {"np": np, "math": math}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), kinematics)


class IKBranchTests(unittest.TestCase):
    def setUp(self):
        self.home = np.array([0., -np.pi/2, 0., 0., 0., 0.])
        self.goal = kinematics["fk"](self.home)
        self.goal[:3, 3] += self.goal[:3, :3] @ np.array([.0004, -.0056, -.0116])
        self.goal[:3, :3] = kinematics["_rpy"](.0141, .0116, -.105)[:3, :3] @ self.goal[:3, :3]

    def test_noisy_straight_seeds_keep_positive_elbow(self):
        for offset in (-.0003, 0., .0003):
            with self.subTest(offset=offset):
                seed = self.home.copy()
                seed[2] += offset
                solution = kinematics["ik"](seed, self.goal)
                self.assertGreater(np.sin(solution[2]), 0)
                actual = kinematics["fk"](solution)
                self.assertLess(np.linalg.norm(actual[:3, 3] - self.goal[:3, 3]), .002)
                error = kinematics["_rot_log"](actual[:3, :3].T @ self.goal[:3, :3])
                self.assertLess(np.linalg.norm(error), np.radians(1))

    def test_opposite_established_branch_requires_reset(self):
        seed = self.home.copy()
        seed[2] = -.5
        with self.assertRaisesRegex(RuntimeError, "opposite IK branch"):
            kinematics["ik"](seed, kinematics["fk"](seed))

    def test_invalid_branch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "elbow_branch"):
            kinematics["ik"](self.home, self.goal, elbow_branch=2)

    def test_explicit_negative_branch_preserves_existing_posture(self):
        seed = self.home.copy()
        seed[2] = -.5
        solution = kinematics["ik"](seed, kinematics["fk"](seed), elbow_branch=-1)
        np.testing.assert_allclose(solution, seed)

    def test_legacy_opt_out_does_not_force_positive_elbow(self):
        seed = self.home.copy()
        seed[2] = -.5
        solution = kinematics["ik"](seed, kinematics["fk"](seed), elbow_branch=0)
        np.testing.assert_allclose(solution, seed)


class IKPoseAcceptanceTests(unittest.TestCase):
    def test_model_start_action_meets_both_pose_limits_with_noisy_home(self):
        action = np.array([
            -0.0019045901257252577, -0.00006618764905222743, -0.008632117407725138,
            0.030013042810614947, -0.03411674526165806, -0.04758950081401889,
        ])
        for offset_deg in (-0.02, 0.0, 0.02):
            with self.subTest(offset_deg=offset_deg):
                seed = np.deg2rad([0, -89.99 + offset_deg, 0, 0, 0, 0])
                before = kinematics["fk"](seed)
                goal = before.copy()
                goal[:3, 3] += before[:3, :3] @ action[:3]
                goal[:3, :3] = kinematics["_rpy"](*action[3:6])[:3, :3] @ before[:3, :3]
                solution = kinematics["ik"](seed, goal)
                actual = kinematics["fk"](solution)
                self.assertLessEqual(np.linalg.norm(actual[:3, 3] - goal[:3, 3]), .002)
                self.assertLessEqual(np.linalg.norm(kinematics["_rot_log"](actual[:3, :3].T @ goal[:3, :3])),
                                     np.radians(1))
                self.assertLessEqual(np.max(np.abs(solution - seed)), np.radians(20))
                self.assertGreaterEqual(np.sin(solution[2]), -1e-4)

    def test_position_only_success_cannot_hide_unreachable_rotation(self):
        old_fk, old_jacobian = kinematics["fk"], kinematics["_jacobian"]
        kinematics["fk"] = lambda q: np.eye(4)
        kinematics["_jacobian"] = lambda q: np.zeros((6, 6))
        goal = kinematics["_rpy"](0, 0, np.radians(30))
        try:
            with self.assertRaisesRegex(RuntimeError, "rotation="):
                kinematics["ik"](np.zeros(6), goal)
        finally:
            kinematics["fk"], kinematics["_jacobian"] = old_fk, old_jacobian

    def test_soft_rotation_accepts_position_when_rotation_unmet(self):
        old_fk, old_jacobian = kinematics["fk"], kinematics["_jacobian"]
        kinematics["fk"] = lambda q: np.eye(4)
        kinematics["_jacobian"] = lambda q: np.zeros((6, 6))
        goal = kinematics["_rpy"](0, 0, np.radians(30))
        try:
            solution = kinematics["ik"](np.zeros(6), goal, require_rotation=False)
            actual = kinematics["fk"](solution)
            self.assertLessEqual(np.linalg.norm(actual[:3, 3] - goal[:3, 3]), .003)
        finally:
            kinematics["fk"], kinematics["_jacobian"] = old_fk, old_jacobian


if __name__ == "__main__":
    unittest.main(verbosity=2)
