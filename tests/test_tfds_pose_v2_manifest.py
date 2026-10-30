import importlib.util
import json
from pathlib import Path
import tempfile
import struct
import unittest
import zlib

import numpy as np

path = Path(__file__).resolve().parents[1] / "scripts/dataset_raw/tfds_pose_v2_manifest.py"
spec = importlib.util.spec_from_file_location("pose_v2_manifest", path)
manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest)


def write_png(path, width=1280, height=720, color=(0, 0, 0)):
    def chunk(name, data):
        return struct.pack('>I', len(data)) + name + data + struct.pack('>I', zlib.crc32(name + data))
    pixels = (b'\0' + bytes(color) * width) * height
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n'
                          + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
                          + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))


class ManifestTests(unittest.TestCase):
    def create_episode(self, root, number):
        ep = root / f"episode_{number:06d}"
        (ep / "images").mkdir(parents=True)
        meta = {"episode_id": ep.name, "success": True,
                "pose_convention": manifest.CONVENTION, "actions_pose_convention": manifest.CONVENTION,
                "action_frame": "ee", "rotation_representation": "rpy_left_multiply",
                "image_resolution": [1280, 720], "quaternion_order": "wxyz",
                "action_units": ["m", "m", "m", "rad", "rad", "rad", "binary"],
                "num_observations": 2, "num_actions": 1, "gripper_open_value": 1.}
        (ep / "meta.json").write_text(json.dumps(meta))
        np.save(ep / "actions.npy", [[.01, 0, 0, 0, 0, 0, 1]])
        np.save(ep / "ee_poses_world.npy", [[0, 0, 0, 1, 0, 0, 0], [.01, 0, 0, 1, 0, 0, 0]])
        np.save(ep / "joint_positions.npy", np.zeros((2, 6)))
        np.save(ep / "timestamps.npy", [1., 1.2])
        for index in range(2):
            write_png(ep / "images" / f"{index:06d}.png", color=(number, index, 0))
        return ep

    def test_split_is_order_independent_and_has_no_overlap(self):
        ids = [f"episode_{i:06d}" for i in range(26)]
        a = manifest.select_splits(ids)
        self.assertEqual(a, manifest.select_splits(list(reversed(ids))))
        self.assertEqual(len(a["train"]), 22)
        self.assertEqual(len(a["val"]), 4)
        self.assertFalse(set(a["train"]) & set(a["val"]))
        self.assertEqual(set(a["train"] + a["val"]), set(ids))

    def test_frozen_manifest_detects_source_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i in range(3):
                self.create_episode(root, i)
            path = root / "plan.json"
            plan = manifest.freeze_manifest(root, path, validation_count=1)
            self.assertEqual(plan, manifest.freeze_manifest(root, path, validation_count=1))
            ep = root / "episode_000000"
            np.save(ep / "actions.npy", [[.02, 0, 0, 0, 0, 0, 1]])
            with self.assertRaisesRegex(ValueError, "Source changed"):
                manifest.freeze_manifest(root, path, validation_count=1)

    def test_manifest_rejects_cross_split_episode_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i in range(3):
                self.create_episode(root, i)
            plan = manifest.make_manifest(root, validation_count=1)
            plan["splits"]["val"] = [plan["splits"]["train"][0]]
            with self.assertRaisesRegex(ValueError, "overlap"):
                manifest.validate_manifest(plan, root, check_source=False)

    def test_legacy_pose_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            ep = self.create_episode(Path(temporary), 0)
            meta = json.loads((ep / "meta.json").read_text())
            meta.pop("actions_pose_convention")
            (ep / "meta.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "uncorrected"):
                manifest.load_episode_inputs(ep)

    def test_missing_link_target_and_wrong_image_size_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_episode(root, 0)
            ep = self.create_episode(root, 1)
            image = ep / "images/000001.png"
            image.unlink()
            image.symlink_to(root / "missing.png")
            with self.assertRaises(FileNotFoundError):
                manifest.inventory(root)
            image.unlink()
            write_png(image, width=10, height=10)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                manifest.inventory(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
