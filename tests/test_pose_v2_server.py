import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "pose_v2_server", Path(__file__).resolve().parents[1] / "scripts/temp/serve_pose_v2.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ServerProtocolTests(unittest.TestCase):
    def test_camera_bytes_roundtrip_and_instruction(self):
        image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        payload = json.loads(json.dumps({"image": image, "instruction": "Push the red cube"}, default=module.numpy_default))
        actual, instruction, double = module.parse_payload(payload)
        np.testing.assert_array_equal(actual, image)
        self.assertEqual(instruction, "push the red cube")
        self.assertFalse(double)

    def test_double_encoded_payload_and_response(self):
        image = np.zeros((2, 3, 3), np.uint8)
        encoded = json.dumps({"image": image, "instruction": "push"}, default=module.numpy_default)
        actual, _, double = module.parse_payload({"encoded": encoded})
        np.testing.assert_array_equal(actual, image)
        self.assertTrue(double)
        action = np.arange(7, dtype=float)
        np.testing.assert_array_equal(json.loads(json.dumps(action, default=module.numpy_default),
                                                object_hook=module.numpy_object_hook), action)

    def test_invalid_camera_and_wrong_statistics_key_are_rejected(self):
        for payload in [{"image": [[0]], "instruction": "push"},
                        {"image": [[[0, 0, 256]]], "instruction": "push"},
                        {"image": [[[0, 0, 0]]], "instruction": "push", "unnorm_key": "old"}]:
            with self.assertRaises(ValueError):
                module.parse_payload(payload)


if __name__ == "__main__":
    unittest.main()
