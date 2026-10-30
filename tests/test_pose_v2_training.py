import importlib.util
from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts/dataset_raw"))
from pose_v2_training_utils import busy_gpu_processes, learning_rate_multiplier, validation_indices


class TrainingScheduleTests(unittest.TestCase):
    def test_gpu_guard_allows_desktop_but_blocks_other_training(self):
        training = "123, python, 4694"
        rows = ["456, /usr/libexec/gnome-remote-desktop-daemon, 463", training, "789, python, 10"]
        self.assertEqual(busy_gpu_processes(rows, 789), [training])

    def test_warmup_then_decay_and_exact_update_range(self):
        rates = [learning_rate_multiplier(i, 200, 20) for i in range(200)]
        self.assertAlmostEqual(rates[0], 0.05)
        self.assertAlmostEqual(rates[19], 1.0)
        self.assertAlmostEqual(rates[-1], 0.1)
        self.assertTrue(all(a <= b for a, b in zip(rates[:19], rates[1:20])))
        self.assertTrue(all(a >= b for a, b in zip(rates[20:-1], rates[21:])))
        with self.assertRaises(ValueError):
            learning_rate_multiplier(200, 200, 20)

    def test_fixed_validation_samples_cover_entire_split(self):
        indices = validation_indices(567, 64)
        self.assertEqual(len(set(indices)), 64)
        self.assertEqual((indices[0], indices[-1]), (0, 566))
        self.assertEqual(validation_indices(3, 64), [0, 1, 2])


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "Requires OpenVLA training environment")
class NormalizationMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import dlimp as dl
        import numpy as np
        import tensorflow as tf
        from prismatic.vla.datasets.rlds.utils.data_utils import (
            NormalizationType, normalize_action_and_proprio,
        )
        cls.tf, cls.np = tf, np
        cls.normalization_type = NormalizationType
        cls.normalize = staticmethod(normalize_action_and_proprio)

    def check_constant_gripper(self, gripper, method):
        tf, np = self.tf, self.np
        metadata = {"action": {
            "mean": np.array([0.0, gripper]), "std": np.array([0.0, 0.0]),
            "min": np.array([0.0, gripper]), "max": np.array([0.0, gripper]),
            "q01": np.array([0.0, gripper]), "q99": np.array([0.0, gripper]),
            "mask": np.array([True, False]),
        }}
        metadata["proprio"] = dict(metadata["action"])
        trajectory = {"action": tf.constant([[0.0, gripper]], dtype=tf.float64), "observation": {}}
        result = self.normalize(trajectory, metadata, method)["action"].numpy()
        np.testing.assert_array_equal(result, [[0.0, gripper]])

    def test_q99_preserves_constant_open_gripper(self):
        self.check_constant_gripper(1.0, self.normalization_type.BOUNDS_Q99)

    def test_bounds_preserves_constant_open_gripper(self):
        self.check_constant_gripper(1.0, self.normalization_type.BOUNDS)

    def test_q99_preserves_closed_gripper(self):
        self.check_constant_gripper(0.0, self.normalization_type.BOUNDS_Q99)


if __name__ == "__main__":
    unittest.main()
