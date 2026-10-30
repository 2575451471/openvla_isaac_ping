import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

path = Path(__file__).resolve().parents[1] / "scripts/isaac/replay_recorded_episode.py"
spec = importlib.util.spec_from_file_location("recorded_replay", path)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ReplayTests(unittest.TestCase):
    def test_planned_trajectory_must_preserve_initial_pose(self):
        q = np.zeros((3, 6))
        def builder(*args):
            return np.ones((3, 6)), {}
        with self.assertRaisesRegex(ValueError, "initial joints"):
            replay.prepare_trajectory(None, {}, q, np.arange(3), builder)

    def test_planned_trajectory_rejects_nonfinite_values(self):
        q = np.zeros((3, 6))
        def builder(*args):
            result = q.copy()
            result[-1, 0] = np.nan
            return result, {}
        with self.assertRaisesRegex(ValueError, "finite"):
            replay.prepare_trajectory(None, {}, q, np.arange(3), builder)

    def test_original_time_interpolation_and_endpoints(self):
        q = np.array([[0., 0.], [2., 4.], [6., 8.]])
        times = np.array([0., .2, .6])
        np.testing.assert_allclose(replay.interpolate_joints(q, times, -.1), q[0])
        np.testing.assert_allclose(replay.interpolate_joints(q, times, .1), [1., 2.])
        np.testing.assert_allclose(replay.interpolate_joints(q, times, .4), [4., 6.])
        np.testing.assert_allclose(replay.interpolate_joints(q, times, 99.), q[-1])

    def test_unwrapped_joint_angles_are_preserved(self):
        q = np.array([[-3.], [-5.]])
        np.testing.assert_allclose(replay.interpolate_joints(q, np.array([0., 1.]), .5), [-4.])

    def test_only_right_edge_counts_as_success(self):
        self.assertTrue(replay.pushed_off_right([-.93, 0., -.05]))
        self.assertTrue(replay.pushed_off_right([-.86, .1, .065]))
        self.assertFalse(replay.pushed_off_right([.93, 0., -.05]))
        self.assertFalse(replay.pushed_off_right([-.93, .5, -.05]))
        self.assertFalse(replay.pushed_off_right([-.81, 0., .065]))

    def test_loading_rejects_legacy_and_nonmonotonic_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            meta = {"success": True}
            (root / "meta.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "raw_pose_v2"):
                replay.load_episode(root)
            meta["pose_convention"] = "world_from_local_column_v2"
            (root / "meta.json").write_text(json.dumps(meta))
            np.save(root / "joint_positions.npy", np.zeros((3, 6)))
            np.save(root / "timestamps.npy", [10., 10.2, 10.4])
            np.save(root / "cube_poses_world.npy", np.tile([0., 0., .065, 1., 0., 0., 0.], (3, 1)))
            _, _, times, _ = replay.load_episode(root)
            np.testing.assert_allclose(times, [0., .2, .4])
            np.save(root / "timestamps.npy", [10., 10., 10.4])
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                replay.load_episode(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
