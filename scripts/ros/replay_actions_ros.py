#!/usr/bin/env python3
"""Feed corrected recorded actions through the real ROS closed-loop controller.

Reset the scene with replay_action_episode.start_reset() in Isaac first.
No model server is needed. This command moves the simulated robot over ROS.
"""
from __future__ import annotations

from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from rclpy.parameter import Parameter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openvla_closed_loop import ClosedLoop, fk


class RecordedActions(ClosedLoop):
    def __init__(self):
        path = ROOT / "scripts/isaac/replay_action_episode.py"
        spec = importlib.util.spec_from_file_location("replay_action_plan", path)
        plan_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plan_module)
        self.episode = plan_module.EPISODE
        meta, self.reference_q, times, _ = plan_module.baseline_module().load_episode(self.episode)
        _, self.preflight = plan_module.build_trajectory(self.episode, meta, self.reference_q, times)
        self.actions = np.load(self.episode / "actions.npy", allow_pickle=False)
        self.durations = np.diff(times)
        self.reference_errors = []
        self._last_reference_step = -1
        overrides = [Parameter(name, value=value) for name, value in {
            "instruction": meta["instruction"], "unnorm_key": "ur10e_push_sim",
            "max_steps": len(self.actions), "interp_duration": float(np.median(self.durations)),
            "delta_frame": "ee", "clip_actions": False,
        }.items()]
        super().__init__(node_name="replay_actions_ros", parameter_overrides=overrides)
        if self.max_steps != len(self.actions) or self.delta_frame != "ee" or self.clip_actions:
            raise ValueError("Recorded replay requires every original ee-frame action without clipping")
        self._flog("recorded_action_source=" + str(self.episode) + " no_model=True")

    def _check_reference(self, step):
        if self._last_reference_step == step:
            return
        with self._lock:
            q = self.q_meas.copy()
        if step == 0 and np.max(np.abs(q - self.reference_q[0])) > .02:
            raise RuntimeError("Robot is not at the recorded initial pose; run start_reset() in Isaac first")
        actual, reference = fk(q), fk(self.reference_q[step])
        position_mm = float(np.linalg.norm(actual[:3, 3] - reference[:3, 3]) * 1000)
        rotation_deg = float(np.degrees(np.arccos(np.clip(
            (np.trace(actual[:3, :3] @ reference[:3, :3].T) - 1) / 2, -1, 1))))
        joint_delta = (q - self.reference_q[step] + np.pi) % (2 * np.pi) - np.pi
        joint_deg = float(np.degrees(np.max(np.abs(joint_delta))))
        error = {"step": step, "position_mm": position_mm, "rotation_deg": rotation_deg,
                 "max_joint_error_deg": joint_deg}
        self.reference_errors.append(error)
        self._last_reference_step = step
        if position_mm > 10 or rotation_deg > 2 or joint_deg > 15:
            raise RuntimeError("Measured ROS trajectory diverged from recording: " + json.dumps(error))

    def _request_action(self, image):
        # Same worker, IK, interpolation, feedback gate and publishers as live inference.
        # Only the action provider changes. Future recorded joints never seed IK.
        self._check_reference(self._steps)
        return self.actions[self._steps].copy()

    def _step_duration(self):
        return float(self.durations[self._steps])


def main():
    rclpy.init()
    node = None
    completed = False
    error = None
    started = time.monotonic()
    output = ROOT / "logs" / ("replay_ros_" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json")
    try:
        node = RecordedActions()
        print("[ros-replay] Waiting for /rgb and /joint_states; model server is not used", flush=True)
        while rclpy.ok() and not node._finished.is_set():
            rclpy.spin_once(node, timeout_sec=.05)
            if node._steps == 0 and time.monotonic() - started > 30:
                raise RuntimeError("No first action after 30s; check Play, ROS bridge, /rgb and /joint_states")
            if node._steps > 0 and time.monotonic() - node._last_image_at > 10:
                raise RuntimeError("No fresh camera frames for 10s; check Play and ROS camera bridge")
        if node._worker_error:
            raise RuntimeError(node._worker_error)
        completed = node._steps == len(node.actions) and node._motion_done.is_set()
        if completed:
            node._check_reference(len(node.actions))
            print("[ros-replay] DONE actions_completed=True; inspect the cube in Isaac", flush=True)
    except KeyboardInterrupt:
        error = "Cancelled by user"
    except Exception as exc:
        error = str(exc)
        completed = False
        print("[ros-replay] STOPPED:", error, flush=True)
    finally:
        report = {"mode": "corrected_actions_ros_measured_feedback", "actions_completed": completed,
                  "task_success": None, "error": error, "wall_seconds": time.monotonic() - started,
                  "scope": "ROS action execution and measured tracking; cube success requires Isaac observation."}
        if node is not None:
            node._stop.set()
            node._worker.join(timeout=1)
            report.update(episode=str(node.episode), actions_submitted=node._steps,
                          preflight=node.preflight, reference_errors=node.reference_errors,
                          motion_failure=node._last_motion_failure,
                          controller_log=str(node._log_path))
            node.destroy_node()
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("[ros-replay] Report:", output, flush=True)
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
