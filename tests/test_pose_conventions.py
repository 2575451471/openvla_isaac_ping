"""Regression tests for USD rotation conventions and non-destructive dataset repair."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dataset_raw.episode_recorder import get_world_pose7
from dataset_raw.make_actions import make_actions_for_episode
from dataset_raw.pose_math import (
    POSE_CONVENTION, R_ISAAC_TO_FK, conjugate_quaternion, delta_action_closed_loop,
    matrix_to_quat_wxyz, quat_wxyz_to_matrix, rpy_to_matrix, usd_rotation_to_column,
)
from dataset_raw.repair_pose_dataset import repair_dataset
from dataset_raw.verify_pose_dataset import verify_dataset


def reference_fk(q):
    matrix = np.eye(4)
    matrix[:3, 3] = q[:3]
    matrix[:3, :3] = rpy_to_matrix(*q[3:])
    return matrix


class PoseTests(unittest.TestCase):
    def test_usd_positive_z_rotation_maps_x_to_y(self):
        row_matrix = np.array([[0., 1., 0., 0.], [-1., 0., 0., 0.],
                               [0., 0., 1., 0.], [1., 2., 3., 1.]])
        rotation = usd_rotation_to_column(row_matrix)
        np.testing.assert_allclose(rotation @ [1., 0., 0.], [0., 1., 0.])
        np.testing.assert_allclose(rotation @ [0., 1., 0.], [-1., 0., 0.])

    def test_recorder_getter_emits_active_wxyz_and_preserves_translation(self):
        class Matrix:
            def __getitem__(self, index):
                return [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]][index]

            def ExtractTranslation(self):
                return (1., 2., 3.)

        usd = types.ModuleType("omni.usd")
        usd.get_context = lambda: types.SimpleNamespace(get_stage=lambda: types.SimpleNamespace(
            GetPrimAtPath=lambda path: types.SimpleNamespace(IsValid=lambda: True)))
        usd.get_world_transform_matrix = lambda prim: Matrix()
        omni = types.ModuleType("omni")
        omni.usd = usd
        with patch.dict(sys.modules, {"omni": omni, "omni.usd": usd}):
            pose = get_world_pose7("/example")
        np.testing.assert_allclose(pose[:3], [1., 2., 3.])
        np.testing.assert_allclose(quat_wxyz_to_matrix(pose[3:]) @ [1., 0., 0.], [0., 1., 0.], atol=1e-14)

    def test_shared_isaac_helper_uses_same_rotation_convention(self):
        path = ROOT / "scripts/isaac/arm_motion.py"
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "gf_matrix_to_np_rotation")
        namespace = {"usd_rotation_to_column": usd_rotation_to_column, "np": np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        row_matrix = [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]
        np.testing.assert_allclose(namespace["gf_matrix_to_np_rotation"](row_matrix) @ [1., 0., 0.], [0., 1., 0.])

    def test_ee_actions_reconstruct_position_and_left_rotation(self):
        start = rpy_to_matrix(.4, -.3, 1.2)
        delta_rotation = rpy_to_matrix(.02, -.03, .04)
        position = np.array([.1, .2, .3])
        following = position + np.array([.01, -.02, .03])
        action = delta_action_closed_loop(position, start, following, delta_rotation @ start)
        np.testing.assert_allclose(position + start @ action[:3], following, atol=1e-8)
        np.testing.assert_allclose(rpy_to_matrix(*action[3:6]) @ start, delta_rotation @ start, atol=1e-8)

    def test_zero_quaternion_is_rejected(self):
        with self.assertRaises(ValueError):
            conjugate_quaternion(np.zeros(4))


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "raw"
        self.destination = self.root / "raw_pose_v2"
        self.episode = self.source / "episode_000001"
        (self.episode / "images").mkdir(parents=True)
        q = np.array([[.1, .2, .3, .1, .2, .3], [.11, .22, .33, .12, .17, .34],
                      [.09, .25, .35, .15, .19, .38]])
        base_position = np.array([.4, -.3, .2])
        base_rotation = rpy_to_matrix(.2, -.3, .5)
        poses = []
        for row in q:
            fk = reference_fk(row)
            world_position = base_position + base_rotation @ R_ISAAC_TO_FK.T @ fk[:3, 3]
            world_rotation = base_rotation @ R_ISAAC_TO_FK.T @ fk[:3, :3]
            poses.append(np.r_[world_position, matrix_to_quat_wxyz(world_rotation.T)])
        np.save(self.episode / "ee_poses_world.npy", np.array(poses))
        np.save(self.episode / "joint_positions.npy", q)
        np.save(self.episode / "cube_poses_world.npy", np.full((3, 7), np.nan))
        np.save(self.episode / "timestamps.npy", np.array([0., .2, .4]))
        np.save(self.episode / "actions.npy", np.zeros((2, 7), dtype=np.float32))
        meta = dict(success=True, quaternion_order="wxyz", action_frame="ee",
                    rotation_representation="rpy_left_multiply", gripper_open_value=1.,
                    base_world_xyz=base_position.tolist(),
                    base_world_quat_wxyz=matrix_to_quat_wxyz(base_rotation.T).tolist())
        (self.episode / "meta.json").write_text(json.dumps(meta))
        for index in range(3):
            (self.episode / "images" / f"{index:06d}.png").write_bytes(b"unchanged image fixture")
        failed = self.source / "episode_000002"
        failed.mkdir()
        (failed / "meta.json").write_text(json.dumps({"success": False}))
        self.validator = lambda root: verify_dataset(root, reference_fk=reference_fk)

    def source_hashes(self):
        return {str(path.relative_to(self.source)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.source.rglob("*") if path.is_file()}

    def test_repair_preserves_sources_and_handles_nonidentity_base(self):
        before = self.source_hashes()
        report = repair_dataset(self.source, self.destination, validator=self.validator)
        self.assertTrue(report["passed"])
        self.assertEqual(report["episodes"], 1)
        self.assertEqual(report["actions"], 2)
        self.assertLess(report["metrics"]["action_position_mm"]["max"], .001)
        self.assertEqual(self.source_hashes(), before)
        repaired = self.destination / self.episode.name
        self.assertTrue((repaired / "images").is_symlink())
        self.assertEqual((repaired / "images/000000.png").read_bytes(), b"unchanged image fixture")
        meta = json.loads((repaired / "meta.json").read_text())
        self.assertEqual(meta["pose_convention"], POSE_CONVENTION)
        self.assertEqual(meta["actions_pose_convention"], POSE_CONVENTION)
        self.assertFalse((self.destination / "episode_000002").exists())

    def test_refuses_existing_destination_and_double_correction(self):
        repair_dataset(self.source, self.destination, validator=self.validator)
        with self.assertRaisesRegex(ValueError, "Destination exists"):
            repair_dataset(self.source, self.destination, validator=self.validator)
        with self.assertRaisesRegex(ValueError, "double correction"):
            repair_dataset(self.destination, self.root / "third", validator=self.validator)

    def test_old_labels_cannot_be_overwritten_without_repair(self):
        before = self.source_hashes()
        with self.assertRaisesRegex(ValueError, "legacy or unknown"):
            make_actions_for_episode(self.episode, overwrite=True)
        self.assertEqual(self.source_hashes(), before)

    def test_failed_validation_does_not_publish_dataset(self):
        before = self.source_hashes()
        with self.assertRaisesRegex(ValueError, "failed execution-FK"):
            repair_dataset(self.source, self.destination, validator=lambda root: {"passed": False})
        self.assertFalse(self.destination.exists())
        self.assertFalse(list(self.root.glob(".raw_pose_v2.repair-*")))
        self.assertEqual(self.source_hashes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
