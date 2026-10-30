"""Run start_replay() in Isaac Script Editor to check recorded-joint playback.

This baseline uses measured joint positions, not OpenVLA or IK. It resets the
simulated robot/cube, follows original simulation timestamps, and writes a report.
"""
from __future__ import annotations

import asyncio
import builtins
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/home/dt/openvla/shenle_projects")
EPISODE = ROOT / "datasets/raw_pose_v2/episode_000001"
PLAYBACK_SPEED = 1.0
MAX_TRACK_ERROR_RAD = 0.15
MAX_BAD_TRACK_FRAMES = 10


def load_episode(path):
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if meta.get("pose_convention") != "world_from_local_column_v2":
        raise ValueError("Use a verified raw_pose_v2 episode")
    if meta.get("success") is not True:
        raise ValueError("Select a successful recorded episode")
    q = np.load(path / "joint_positions.npy", allow_pickle=False)
    timestamps = np.load(path / "timestamps.npy", allow_pickle=False)
    cube = np.load(path / "cube_poses_world.npy", allow_pickle=False)
    if q.ndim != 2 or q.shape[1] != 6 or len(q) < 2:
        raise ValueError("Expected (N, 6) measured joints")
    if timestamps.shape != (len(q),) or cube.shape != (len(q), 7):
        raise ValueError("Joint, cube, and timestamp counts differ")
    if not all(np.isfinite(value).all() for value in (q, timestamps, cube)):
        raise ValueError("Non-finite episode data")
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("Timestamps must be strictly increasing")
    return meta, q, timestamps - timestamps[0], cube


def interpolate_joints(q, times, elapsed):
    if elapsed <= times[0]:
        return q[0].copy()
    if elapsed >= times[-1]:
        return q[-1].copy()
    index = int(np.searchsorted(times, elapsed, side="right") - 1)
    alpha = (elapsed - times[index]) / (times[index + 1] - times[index])
    # Preserve unwrapped recorded angles; do not wrap joints through +/-pi.
    return (1.0 - alpha) * q[index] + alpha * q[index + 1]


def pushed_off_right(position):
    x, y, z = map(float, position[:3])
    return x < -0.80 and abs(y) <= 0.35 and (z < 0.03 or x < -0.85)


def prepare_trajectory(episode, meta, recorded_q, times, trajectory_builder=None):
    if trajectory_builder is None:
        return recorded_q, {}
    q, details = trajectory_builder(episode, meta, recorded_q.copy(), times.copy())
    q = np.asarray(q, dtype=float)
    if q.shape != recorded_q.shape or not np.isfinite(q).all():
        raise ValueError("Planned trajectory must be finite and match recorded joint shape")
    if not np.allclose(q[0], recorded_q[0], atol=1e-9, rtol=0):
        raise ValueError("Planned trajectory must start at the recorded initial joints")
    return q, details


async def replay_recorded_episode(*, trajectory_builder=None,
                                  mode="recorded_joint_baseline",
                                  report_prefix="replay_joints",
                                  reset_only=False,
                                  reset_step=0,
                                  scope="Joint/scene/physics baseline; does not validate action-to-IK control or model quality."):
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim

    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from isaac.arm_motion import find_arm_indices, read_arm_q, send_arm_target
    from dataset_raw.pose_math import quat_wxyz_to_matrix, usd_rotation_to_column

    meta, q, times, cube_poses = load_episode(EPISODE)
    q, plan = prepare_trajectory(EPISODE, meta, q, times, trajectory_builder)
    if not np.isfinite(PLAYBACK_SPEED) or PLAYBACK_SPEED <= 0:
        raise ValueError("PLAYBACK_SPEED must be positive")
    timeline = omni.timeline.get_timeline_interface()
    app = omni.kit.app.get_app()
    stage = omni.usd.get_context().get_stage()
    if not timeline.is_playing():
        raise RuntimeError("Open sence1.usd and press Play before running replay")
    for name in ("base_prim", "ee_prim", "cube_prim"):
        if not stage.GetPrimAtPath(meta[name]).IsValid():
            raise RuntimeError("Scene prim missing: " + meta[name])
    base_matrix = omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(meta["base_prim"]))
    base_position = np.array(base_matrix.ExtractTranslation(), dtype=float)
    if not np.allclose(base_position, meta["base_world_xyz"], atol=.001):
        raise RuntimeError("Robot base position differs from the recording; use the original scene")
    if not np.allclose(usd_rotation_to_column(base_matrix),
                       quat_wxyz_to_matrix(meta["base_world_quat_wxyz"]), atol=1e-4):
        raise RuntimeError("Robot base orientation differs from the recording")

    robot = SingleArticulation(prim_path=str(Path(meta["base_prim"]).parent))
    robot.initialize()
    indices = np.asarray(find_arm_indices(list(robot.dof_names)), dtype=np.int32)
    cube = SingleRigidPrim(prim_path=meta["cube_prim"])
    cube.initialize()
    report = {"mode": mode, "episode": str(EPISODE),
              "playback_speed": PLAYBACK_SPEED, "recorded_sim_seconds": float(times[-1]),
              "completed": False, "task_success": False,
              "scope": scope, "plan": plan}
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    output = logs / (report_prefix + "_" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json")
    errors = []
    last_time = timeline.get_current_time()

    async def next_frame():
        nonlocal last_time
        await app.next_update_async()
        if not timeline.is_playing():
            raise RuntimeError("Replay stopped because simulation was paused/stopped")
        current = timeline.get_current_time()
        if current < last_time - 1e-8:
            raise RuntimeError("Replay stopped because simulation time was rewound")
        last_time = current
        return current

    async def hold(target, duration):
        end = timeline.get_current_time() + duration
        while timeline.get_current_time() < end:
            send_arm_target(robot, indices, target)
            await next_frame()

    k = int(reset_step)
    if not 0 <= k < len(q) or (k and not reset_only):
        raise ValueError("reset_step must index a recorded frame and is only valid with reset_only")
    try:
        print(f"[replay] Resetting simulated robot and cube to recorded frame {k}")
        robot.set_joint_positions(q[k], joint_indices=indices)
        robot.set_joint_velocities(np.zeros(6), joint_indices=indices)
        send_arm_target(robot, indices, q[k])
        cube.set_world_pose(position=cube_poses[k, :3], orientation=cube_poses[k, 3:])
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))
        await hold(q[k], .5)
        if np.max(np.abs(read_arm_q(robot, indices) - q[k])) > .02:
            raise RuntimeError("Robot did not hold the reset pose; check competing controllers")
        if reset_only:
            report.update(completed=True, task_success=None, scene_reset=True, reset_step=k,
                          reset_joints_deg=np.degrees(q[k]).round(3).tolist(),
                          reset_cube_world=cube_poses[k, :3].tolist())
            print(f"[replay] RESET READY at frame {k}: start the ROS controller now")
            return
        start = timeline.get_current_time()
        progress = -1
        bad_frames = 0
        while True:
            elapsed = (timeline.get_current_time() - start) * PLAYBACK_SPEED
            target = interpolate_joints(q, times, elapsed)
            error = float(np.max(np.abs(read_arm_q(robot, indices) - target)))
            errors.append(error)
            bad_frames = bad_frames + 1 if error > MAX_TRACK_ERROR_RAD else 0
            if bad_frames >= MAX_BAD_TRACK_FRAMES:
                raise RuntimeError(f"Tracking error {error:.3f} rad persists; replay aborted")
            send_arm_target(robot, indices, target)
            percent = min(100, int(100 * elapsed / times[-1]))
            if percent // 10 != progress:
                progress = percent // 10
                print(f"[replay] {percent:3d}% sim={elapsed:.2f}/{times[-1]:.2f}s tracking={error:.4f}rad")
            if elapsed >= times[-1]:
                break
            await next_frame()
        await hold(q[-1], 1.5)
        cube_position, _ = cube.get_world_pose()
        report.update(completed=True, task_success=pushed_off_right(cube_position),
                      cube_final_world=np.asarray(cube_position).tolist(),
                      recorded_cube_final_world=cube_poses[-1, :3].tolist(),
                      joint_final_error_rad=float(np.max(np.abs(read_arm_q(robot, indices) - q[-1]))))
        print("[replay] DONE task_success=" + str(report["task_success"]))
        print("[replay] Cube final world:", report["cube_final_world"])
    except asyncio.CancelledError:
        report["error"] = "Replay cancelled"
        raise
    except Exception as error:
        report["error"] = str(error)
        print("[replay] STOPPED:", error)
    finally:
        if errors:
            report["tracking_max_rad"] = max(errors)
            report["tracking_mean_rad"] = float(np.mean(errors))
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("[replay] Report:", output)
        if timeline.is_playing():
            send_arm_target(robot, indices, read_arm_q(robot, indices))


def start_replay(**options):
    previous = getattr(builtins, "_shenle_joint_replay_task", None)
    if previous is not None and not previous.done():
        raise RuntimeError("A replay is already running; pause simulation to stop it first")
    builtins._shenle_joint_replay_task = asyncio.ensure_future(replay_recorded_episode(**options))

    def report_start_failure(task):
        if not task.cancelled() and task.exception() is not None:
            print("[replay] START FAILED:", task.exception())

    builtins._shenle_joint_replay_task.add_done_callback(report_start_failure)
    return builtins._shenle_joint_replay_task
