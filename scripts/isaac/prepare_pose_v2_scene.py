"""Kit --exec bootstrap: load the stage, Play, and await the existing reset."""

import asyncio
import builtins
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import time
import traceback


def write_status(path, phase, **details):
    value = {"phase": phase, "pid": os.getpid(),
             "updated_at": datetime.now(timezone.utc).isoformat(), **details}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
    print("[pose-v2-start] " + phase + " " + json.dumps(details, ensure_ascii=False), flush=True)


async def await_updates(runtime, predicate, timeout, description):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(description)
        await runtime.update()


DEFAULT_PRIMS = ("/World/ur10e/base_link", "/World/ur10e/wrist_3_link", "/World/RedCube")


def scene_prims(root):
    meta_path = Path(root) / "datasets/raw_pose_v2/episode_000001/meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        return [meta[key] for key in ("base_prim", "ee_prim", "cube_prim")]
    return list(DEFAULT_PRIMS)


async def prepare(runtime, root, stage_path, status_path, timeout=180, warmup_frames=30, mode="reset"):
    """Dependency-injected orchestration, also exercised without Kit or a GPU."""
    root, stage_path, status_path = map(Path, (root, stage_path, status_path))
    try:
        write_status(status_path, "LOADING_SCENE", stage=str(stage_path))
        await asyncio.wait_for(runtime.open_stage(stage_path), timeout=timeout)
        await await_updates(runtime, lambda: not runtime.loading(), timeout,
                            "USD assets did not finish loading")
        runtime.validate_scene(stage_path, scene_prims(root))
        write_status(status_path, "PLAYING", stage=str(stage_path))
        runtime.play()
        await await_updates(runtime, runtime.playing, 15, "Timeline did not enter Play")
        for _ in range(warmup_frames):
            await runtime.update()
            if not runtime.playing():
                raise RuntimeError("Simulation was paused during initialization")
        if mode == "collect":
            write_status(status_path, "COLLECTING", stage=str(stage_path))
            successes = await asyncio.wait_for(
                runtime.run_collect(root / "scripts/isaac/collect_push_batch.py"), timeout=timeout)
            if not runtime.playing():
                raise RuntimeError("Collection finished but simulation is no longer playing")
            write_status(status_path, "READY", stage=str(stage_path), collect_successes=int(successes or 0),
                         scope="Batch collection; no closed-loop control started")
            return True
        if mode == "play":
            write_status(status_path, "HOMING", stage=str(stage_path))
            await asyncio.wait_for(runtime.hold_home(), timeout=timeout)
            if not runtime.playing():
                raise RuntimeError("Home pose finished but simulation is no longer playing")
            write_status(status_path, "READY", stage=str(stage_path),
                         scope="Scene playing at collection HOME; model not started")
            return True
        write_status(status_path, "RESETTING", stage=str(stage_path))
        logs = root / "logs"
        before = set(logs.glob("replay_reset_*.json"))
        task = runtime.start_reset(root / "scripts/isaac/replay_action_episode.py")
        await asyncio.wait_for(task, timeout=timeout)
        reports = set(logs.glob("replay_reset_*.json")) - before
        if len(reports) != 1:
            raise RuntimeError("Expected exactly one new reset report; found " + str(len(reports)))
        report_path = next(iter(reports))
        report = json.loads(report_path.read_text())
        # The legacy coroutine catches some failures; awaiting it alone is insufficient.
        if report.get("completed") is not True or report.get("scene_reset") is not True or report.get("error"):
            raise RuntimeError("Reset did not pass: " + json.dumps(report, ensure_ascii=False))
        if not runtime.playing():
            raise RuntimeError("Reset finished but simulation is no longer playing")
        write_status(status_path, "READY", stage=str(stage_path), reset_report=str(report_path),
                     scope="Scene reset only; no model actions or trajectory replay started")
        return True
    except asyncio.CancelledError:
        runtime.pause()
        write_status(status_path, "ERROR", error="Scene preparation cancelled")
        raise
    except Exception as error:
        runtime.pause()
        write_status(status_path, "ERROR", error=str(error), traceback=traceback.format_exc())
        return False


class KitRuntime:
    def __init__(self):
        import omni.kit.app
        import omni.timeline
        import omni.usd
        self.app = omni.kit.app.get_app()
        self.context = omni.usd.get_context()
        self.timeline = omni.timeline.get_timeline_interface()

    async def update(self):
        await self.app.next_update_async()

    async def open_stage(self, path):
        result = await self.context.open_stage_async(str(path))
        if not result[0]:
            raise RuntimeError("Opening USD failed: " + str(result))

    def loading(self):
        return self.context.get_stage_loading_status()[2] > 0

    def validate_scene(self, path, prims):
        stage = self.context.get_stage()
        if stage is None:
            raise RuntimeError("USD stage is absent")
        if Path(stage.GetRootLayer().realPath).resolve() != path.resolve():
            raise RuntimeError("Unexpected stage is open; refusing to reset it")
        for name in prims:
            if not stage.GetPrimAtPath(name).IsValid():
                raise RuntimeError("Required scene prim missing: " + name)

    def play(self):
        self.timeline.play()

    def playing(self):
        return self.timeline.is_playing()

    def pause(self):
        self.timeline.pause()

    def start_reset(self, path):
        module = runpy.run_path(str(path))
        return module["start_reset"](int(os.environ.get("SHENLE_WEBRTC_RESET_STEP", "0")))

    def run_collect(self, path):
        os.environ["SHENLE_SKIP_COLLECT_AUTORUN"] = "1"
        module = runpy.run_path(str(path))
        return asyncio.ensure_future(module["run_batch"]())

    def hold_home(self):
        import os
        import sys
        import numpy as np
        scripts = os.path.join(os.environ.get("SHENLE_WEBRTC_ROOT", ""), "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from isaac.arm_motion import (
            ROBOT_PRIM, find_arm_indices, move_arm_to_q, resolve_cube_prim,
            set_prim_world_translation,
        )
        from isaacsim.core.prims import SingleArticulation

        async def move():
            robot = SingleArticulation(prim_path=ROBOT_PRIM)
            robot.initialize()
            arm_idx = find_arm_indices(list(robot.dof_names))
            home = np.radians([0.0, -88.4, 80.1, 0.0, 254.2, 0.0])
            cube = resolve_cube_prim(None)
            set_prim_world_translation(cube, np.array([0.0, 0.0, 0.040], dtype=float))
            await move_arm_to_q(robot, arm_idx, home, app=self.app, frames=150)

        return asyncio.ensure_future(move())


def schedule():
    status = Path(os.environ["SHENLE_WEBRTC_STATUS"])
    previous = getattr(builtins, "_shenle_scene_start_task", None)
    if previous is not None and not previous.done():
        raise RuntimeError("Scene preparation is already running")

    async def bootstrap():
        try:
            runtime = KitRuntime()
            # Yield to Kit so --exec never blocks extension startup.
            await runtime.update()
            await prepare(runtime, Path(os.environ["SHENLE_WEBRTC_ROOT"]),
                          Path(os.environ["SHENLE_WEBRTC_STAGE"]), status,
                          timeout=float(os.environ.get("SHENLE_WEBRTC_TIMEOUT", "180")),
                          mode=os.environ.get("SHENLE_WEBRTC_MODE", "reset"))
        except Exception as error:
            write_status(status, "ERROR", error=str(error), traceback=traceback.format_exc())

    builtins._shenle_scene_start_task = asyncio.ensure_future(bootstrap())


# Kit versions can use different execution namespaces for --exec.
if __name__ == "__main__" or os.environ.get("SHENLE_WEBRTC_STATUS"):
    schedule()
