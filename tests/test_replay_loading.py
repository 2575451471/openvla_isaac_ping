import importlib.util
import inspect
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


path = Path(__file__).resolve().parents[1] / "scripts/isaac/replay_action_episode.py"
namespace = {"__file__": str(path), "__name__": "replay_loading_test"}
exec(compile(path.read_bytes(), str(path), "exec"), namespace)
replay = types.SimpleNamespace(**namespace)


class ReplayLoadingTests(unittest.TestCase):
    def test_source_loader_cache_cannot_hide_current_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "sample.py"
            source.write_text('def version(): return "old"\n')
            stat = source.stat()
            spec = importlib.util.spec_from_file_location("cached_replay", source)
            cached = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cached)
            self.assertEqual(cached.version(), "old")
            # Same size and timestamp deliberately preserve a valid stale pyc.
            source.write_text('def version(): return "new"\n')
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            with patch("importlib.machinery.SourceFileLoader.get_code", side_effect=AssertionError("Cached loader used")):
                fresh = replay.load_module(source, "fresh_replay")
            self.assertEqual(fresh.version(), "new")
            self.assertEqual(fresh.__file__, str(source))

    def test_actual_reset_keyword_binds_without_running_simulation(self):
        baseline = replay.baseline_module()
        function = baseline.replay_recorded_episode
        self.assertIn("reset_only", inspect.signature(function).parameters)
        coroutine = function(reset_only=True)
        coroutine.close()

    def test_reset_entry_passes_flag_to_fresh_baseline(self):
        baseline = types.SimpleNamespace(start_replay=Mock(return_value="task"))
        with patch.dict(namespace, baseline_module=lambda: baseline):
            task = namespace["start_reset"]()
        self.assertEqual(task, "task")
        self.assertTrue(baseline.start_replay.call_args.kwargs["reset_only"])
        self.assertEqual(baseline.start_replay.call_args.kwargs["mode"], "ros_replay_scene_reset")


if __name__ == "__main__":
    unittest.main(verbosity=2)
