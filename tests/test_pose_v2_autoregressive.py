import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/temp/evaluate_pose_v2_autoregressive.py"
spec = importlib.util.spec_from_file_location("pose_v2_autoregressive", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AutoregressiveMetricsTests(unittest.TestCase):
    def test_normalization_preserves_constant_unmasked_gripper(self):
        stats = {"q01": [-1] * 6 + [1], "q99": [1] * 7,
                 "min": [-1] * 6 + [1], "max": [1] * 7, "mask": [True] * 6 + [False]}
        result = module.normalized_reference([0] * 6 + [1], stats)
        np.testing.assert_allclose(result, [0] * 6 + [1], atol=1e-8)

    def test_constant_masked_dimension_is_zero(self):
        stats = {"q01": [2] * 7, "q99": [2] * 7, "min": [2] * 7,
                 "max": [2] * 7, "mask": [True] * 7}
        np.testing.assert_array_equal(module.normalized_reference([2] * 7, stats), [0] * 7)

    def test_identical_actions_and_opposite_direction_are_detected(self):
        rows = [{"prediction": [-0.01, 0, 0, 0, 0, 0, 1],
                 "reference": [0.01, 0, 0, 0, 0, 0, 1],
                 "normalized_prediction": [-1, 0, 0, 0, 0, 0, 1],
                 "normalized_reference": [1, 0, 0, 0, 0, 0, 1],
                 "generated_tokens": [31800] * 7, "invalid_action_tokens": 0} for _ in range(3)]
        result = module.summarize(rows)
        self.assertEqual(result["unique_actions"], 1)
        self.assertEqual(result["dominant_action_fraction"], 1)
        self.assertEqual(result["opposite_translation_direction_fraction"], 1)
        self.assertAlmostEqual(result["xyz_error_norm_mean_mm"], 20)


if __name__ == "__main__":
    unittest.main()
