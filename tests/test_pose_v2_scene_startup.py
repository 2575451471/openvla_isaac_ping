"""CPU-only sequencing, failure propagation, and detached-launcher checks."""

import asyncio
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bootstrap = load("pose_scene_bootstrap_test", ROOT / "scripts/isaac/prepare_pose_v2_scene.py")
launcher = load("pose_scene_launcher_test", ROOT / "scripts/isaac/launch_pose_v2_webrtc.py")


class FakeRuntime:
    def __init__(self, root, *, load_error=False, reset_error=False, silent_error=False, pause=False):
        self.root, self.calls, self.active = root, [], False
        self.load_error, self.reset_error = load_error, reset_error
        self.silent_error, self.pause_during_warmup = silent_error, pause

    async def open_stage(self, path):
        self.calls.append("open")
        if self.load_error:
            raise RuntimeError("load failed")

    async def update(self):
        self.calls.append("update")
        if self.pause_during_warmup:
            self.active = False
        await asyncio.sleep(0)

    def loading(self):
        return False

    def validate_scene(self, path, prims):
        self.calls.append("validate")

    def play(self):
        self.calls.append("play")
        self.active = True

    def playing(self):
        return self.active

    def pause(self):
        self.calls.append("pause")
        self.active = False

    def start_reset(self, path):
        async def reset():
            self.calls.append("reset")
            if self.reset_error:
                raise RuntimeError("reset failed before report")
            report = {"completed": not self.silent_error, "scene_reset": not self.silent_error}
            if self.silent_error:
                report["error"] = "Robot did not hold initial pose"
            (self.root / "logs/replay_reset_new.json").write_text(json.dumps(report))
        return asyncio.ensure_future(reset())

    def run_collect(self, path):
        async def collect():
            self.calls.append("collect")
            return 2
        return asyncio.ensure_future(collect())


class SceneStartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "project"
        episode = self.root / "datasets/raw_pose_v2/episode_000001"
        episode.mkdir(parents=True)
        (episode / "meta.json").write_text(json.dumps({"base_prim": "/World/robot/base",
            "ee_prim": "/World/robot/ee", "cube_prim": "/World/RedCube"}))
        (self.root / "logs").mkdir()
        self.status_path = self.root / "status.json"
        self.stage = self.root / "assets/sence/sence1.usd"
        self.stage.parent.mkdir(parents=True)
        self.stage.write_bytes(b"fixture USD; must not be saved or overwritten")

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, mode="reset", **options):
        runtime = FakeRuntime(self.root, **options)
        with contextlib.redirect_stdout(io.StringIO()):
            passed = asyncio.run(bootstrap.prepare(runtime, self.root, self.stage, self.status_path,
                                                    timeout=1, warmup_frames=2, mode=mode))
        return runtime, passed, json.loads(self.status_path.read_text())

    def test_collect_mode_skips_reset(self):
        runtime, passed, status = self.prepare(mode="collect")
        self.assertTrue(passed)
        self.assertEqual(status["phase"], "READY")
        self.assertEqual(status.get("collect_successes"), 2)
        self.assertNotIn("reset", runtime.calls)
        self.assertIn("collect", runtime.calls)

    def test_load_play_then_reset_and_verify_report(self):
        original = self.stage.read_bytes()
        runtime, passed, status = self.prepare()
        self.assertTrue(passed)
        self.assertEqual(status["phase"], "READY")
        self.assertLess(runtime.calls.index("open"), runtime.calls.index("play"))
        self.assertLess(runtime.calls.index("play"), runtime.calls.index("reset"))
        self.assertTrue(runtime.playing())
        self.assertEqual(self.stage.read_bytes(), original)

    def test_load_failure_never_plays_or_resets(self):
        runtime, passed, status = self.prepare(load_error=True)
        self.assertFalse(passed)
        self.assertEqual(status["phase"], "ERROR")
        self.assertNotIn("play", runtime.calls)
        self.assertNotIn("reset", runtime.calls)

    def test_reset_start_error_is_reported_and_pauses(self):
        runtime, passed, status = self.prepare(reset_error=True)
        self.assertFalse(passed)
        self.assertFalse(runtime.playing())
        self.assertIn("reset failed", status["error"])

    def test_swallowed_legacy_reset_failure_is_not_marked_ready(self):
        runtime, passed, status = self.prepare(silent_error=True)
        self.assertFalse(passed)
        self.assertEqual(status["phase"], "ERROR")
        self.assertIn("Robot did not hold", status["error"])
        self.assertFalse(runtime.playing())

    def test_paused_simulation_aborts_before_reset(self):
        runtime, passed, status = self.prepare(pause=True)
        self.assertFalse(passed)
        self.assertNotIn("reset", runtime.calls)
        self.assertEqual(status["phase"], "ERROR")

    def test_old_success_report_cannot_mask_failed_reset(self):
        (self.root / "logs/replay_reset_old.json").write_text(json.dumps(
            {"completed": True, "scene_reset": True}))
        _, passed, status = self.prepare(silent_error=True)
        self.assertFalse(passed)
        self.assertEqual(status["phase"], "ERROR")

    def test_command_uses_existing_streaming_launcher_and_project_bootstrap(self):
        import argparse
        args = argparse.Namespace(isaac_dir=Path("/home/dt/isaacsim5.1"), address="100.64.61.8", port=49100)
        command = launcher.build_command(args, self.root)
        self.assertEqual(command[0], "/home/dt/isaacsim5.1/isaac-sim.streaming.sh")
        self.assertIn("--/app/livestream/port=49100", command)
        self.assertIn("--/app/file/ignoreUnsavedOnExit=true", command)
        self.assertEqual(command[-1], str(self.root / "scripts/isaac/prepare_pose_v2_scene.py"))

    def test_detached_launch_status_and_stop_without_kit_or_gpu(self):
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
        except OSError as error:
            self.skipTest("Local TCP sockets unavailable: " + str(error))
        scripts = self.root / "scripts/isaac"
        scripts.mkdir(parents=True)
        for name in ("launch_pose_v2_webrtc.py", "prepare_pose_v2_scene.py"):
            shutil.copyfile(ROOT / "scripts/isaac" / name, scripts / name)
        for name in ("replay_action_episode.py", "replay_recorded_episode.py"):
            (scripts / name).write_text("# fixture dependency\n")
        episode = self.root / "datasets/raw_pose_v2/episode_000001"
        for name in ("joint_positions.npy", "timestamps.npy", "cube_poses_world.npy"):
            (episode / name).write_bytes(b"fixture; fake Kit does not read arrays")
        isaac = Path(self.temp.name) / "fake-isaac"
        isaac.mkdir()
        fake = isaac / "isaac-sim.streaming.sh"
        fake.write_text("#!/usr/bin/python3\n" +
            "import json,os,socket,sys,time\n" +
            "from pathlib import Path\n" +
            "port=int(next(a.split('=')[1] for a in sys.argv if a.startswith('--/app/livestream/port=')))\n" +
            "listener=socket.socket();listener.bind(('127.0.0.1',port));listener.listen()\n" +
            "path=Path(os.environ['SHENLE_WEBRTC_STATUS'])\n" +
            "tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps({'phase':'READY'}));tmp.replace(path)\n" +
            "while True: time.sleep(0.2)\n")
        fake.chmod(0o755)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        entry = scripts / "launch_pose_v2_webrtc.py"
        # Isolate process lifecycle testing from unrelated real Isaac users/ports.
        # The fixture still exercises the real launcher and TCP readiness check.
        wrapper = self.root / "test_launcher_wrapper.py"
        wrapper.write_text("import importlib.util,sys,socket\n" +
            "spec=importlib.util.spec_from_file_location('fixture_launcher'," + repr(str(entry)) + ")\n" +
            "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n" +
            "module.existing_processes=lambda:[]\n" +
            "original=module.check_port\n" +
            "module.check_port=lambda port,kind: original(port,kind) if kind==socket.SOCK_STREAM else None\n" +
            "sys.exit(module.main())\n")
        original = self.stage.read_bytes()
        command = [sys.executable, str(wrapper), "--isaac-dir", str(isaac), "--address", "127.0.0.1",
                   "--port", str(port), "--timeout", "10"]
        try:
            started = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(started.returncode, 0, started.stderr)
            self.assertIn("READY: scene loaded", started.stdout)
            result = subprocess.run([sys.executable, str(entry), "--status"], capture_output=True, text=True, timeout=5)
            self.assertTrue(json.loads(result.stdout)["running"])
            self.assertEqual(self.stage.read_bytes(), original)
        finally:
            stopped = subprocess.run([sys.executable, str(entry), "--stop"], capture_output=True, text=True, timeout=25)
        self.assertEqual(stopped.returncode, 0, stopped.stderr)
        result = subprocess.run([sys.executable, str(entry), "--status"], capture_output=True, text=True, timeout=5)
        self.assertFalse(json.loads(result.stdout)["running"])
        self.assertEqual(self.stage.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
