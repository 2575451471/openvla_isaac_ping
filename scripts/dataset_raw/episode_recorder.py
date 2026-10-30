"""Bypass raw episode recorder for Isaac Sim (no ROS / OpenVLA / TensorFlow).

Records absolute observations during motion. Does NOT write actions.npy —
run make_actions.py offline after a successful episode.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dataset_raw.pose_math import (  # noqa: E402
    POSE_CONVENTION,
    matrix_to_quat_wxyz,
    usd_rotation_to_column,
)

DEFAULT_ROOT = Path("/home/dt/openvla/shenle_projects/datasets/raw_pose_v2")
DEFAULT_CAMERA = "/World/Realsense/RSD455/Camera_OmniVision_OV9782_Color"
DEFAULT_EE = "/World/ur10e/wrist_3_link"
DEFAULT_BASE = "/World/ur10e/base_link"
CUBE_CANDIDATES = ("/World/RedCube", "/World/red_cup", "/World/Red_Cup")


def get_world_pose7(prim_path: str) -> np.ndarray:
    """Return [x,y,z, qw,qx,qy,qz] in World."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError("invalid prim: " + prim_path)
    T = omni.usd.get_world_transform_matrix(prim)
    p = T.ExtractTranslation()
    R = usd_rotation_to_column(T)
    q = matrix_to_quat_wxyz(R)
    return np.array([p[0], p[1], p[2], q[0], q[1], q[2], q[3]], dtype=float)


def _next_episode_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        ep = root / f"episode_{idx:06d}"
        if not ep.exists():
            return ep
        idx += 1


def _resolve_cube_prim(preferred: str | None) -> str | None:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(CUBE_CANDIDATES)
    for path in candidates:
        if stage.GetPrimAtPath(path).IsValid():
            return path
    return None


class EpisodeRecorder:
    """Time-based sampler: absolute poses + RGB. Skip END_HOLD yourself."""

    def __init__(
        self,
        *,
        instruction: str,
        sample_hz: float = 5.0,
        out_root: str | Path = DEFAULT_ROOT,
        ee_prim: str = DEFAULT_EE,
        base_prim: str = DEFAULT_BASE,
        camera_prim: str = DEFAULT_CAMERA,
        cube_prim: str | None = None,
        robot_name: str = "ur10e",
        image_wh: tuple[int, int] = (1280, 720),
        action_frame: str = "ee",
        rotation_representation: str = "rpy_left_multiply",
        gripper_open_value: float = 1.0,
        episode_name: str | None = None,
    ):
        if sample_hz <= 0:
            raise ValueError("sample_hz must be positive")
        self.instruction = instruction
        self.sample_hz = float(sample_hz)
        self.dt = 1.0 / self.sample_hz
        self.ee_prim = ee_prim
        self.base_prim = base_prim
        self.camera_prim = camera_prim
        self.cube_prim = _resolve_cube_prim(cube_prim)
        self.robot_name = robot_name
        self.image_wh = (int(image_wh[0]), int(image_wh[1]))
        self.action_frame = action_frame
        self.rotation_representation = rotation_representation
        self.gripper_open_value = float(gripper_open_value)

        if episode_name:
            self.episode_dir = Path(out_root) / episode_name
            if self.episode_dir.exists():
                shutil.rmtree(self.episode_dir)
        else:
            self.episode_dir = _next_episode_dir(Path(out_root))
        self.images_dir = self.episode_dir / "images"
        self.images_dir.mkdir(parents=True)

        self._times: list[float] = []
        self._ee: list[np.ndarray] = []
        self._q: list[np.ndarray] = []
        self._cube: list[np.ndarray] = []
        self._last_sample_t: float | None = None
        self._active = True
        self._camera = None
        self._base_xyz = None
        self._base_quat = None

        self._init_camera()
        pose_b = get_world_pose7(self.base_prim)
        self._base_xyz = pose_b[:3].copy()
        self._base_quat = pose_b[3:].copy()
        print("[recorder] writing to", self.episode_dir)
        if self.cube_prim is None:
            print("[recorder] WARNING: no cube prim; cube_poses will be NaN")
        else:
            print("[recorder] cube_prim =", self.cube_prim)

    def _init_camera(self):
        try:
            from isaacsim.sensors.camera import Camera
        except ImportError:
            from omni.isaac.sensor import Camera  # type: ignore

        w, h = self.image_wh
        self._camera = Camera(prim_path=self.camera_prim, resolution=(w, h))
        self._camera.initialize()

    def maybe_sample(self, sim_time: float, q_arm: np.ndarray) -> bool:
        if not self._active:
            return False
        if self._last_sample_t is not None and (sim_time - self._last_sample_t) < self.dt - 1e-9:
            return False
        self._sample(sim_time, q_arm)
        self._last_sample_t = sim_time
        return True

    def _sample(self, sim_time: float, q_arm: np.ndarray):
        from PIL import Image

        ee = get_world_pose7(self.ee_prim)
        if self.cube_prim is not None:
            cube = get_world_pose7(self.cube_prim)
        else:
            cube = np.full(7, np.nan, dtype=float)

        rgba = self._camera.get_rgba()
        if rgba is None or getattr(rgba, "size", 0) == 0:
            raise RuntimeError("empty camera frame; is the scene Playing?")

        idx = len(self._times)
        Image.fromarray(np.asarray(rgba)[:, :, :3]).save(
            self.images_dir / f"{idx:06d}.png"
        )

        self._times.append(float(sim_time))
        self._ee.append(np.asarray(ee, dtype=np.float64))
        self._q.append(np.asarray(q_arm, dtype=np.float64).reshape(6))
        self._cube.append(np.asarray(cube, dtype=np.float64))

    def stop_sampling(self):
        self._active = False

    def finalize(self, success: bool = True) -> Path:
        self.stop_sampling()
        n = len(self._times)
        if n < 2:
            raise RuntimeError(f"need >=2 observations, got {n}")

        np.save(self.episode_dir / "timestamps.npy", np.asarray(self._times, dtype=np.float64))
        np.save(self.episode_dir / "ee_poses_world.npy", np.stack(self._ee, axis=0))
        np.save(self.episode_dir / "joint_positions.npy", np.stack(self._q, axis=0))
        np.save(self.episode_dir / "cube_poses_world.npy", np.stack(self._cube, axis=0))

        w = h = None
        first = self.images_dir / "000000.png"
        if first.is_file():
            from PIL import Image

            im = Image.open(first)
            w, h = im.size

        meta = {
            "episode_id": self.episode_dir.name,
            "instruction": self.instruction,
            "success": bool(success),
            "robot": self.robot_name,
            "ee_prim": self.ee_prim,
            "base_prim": self.base_prim,
            "camera_prim": self.camera_prim,
            "cube_prim": self.cube_prim,
            "sample_rate_hz": self.sample_hz,
            "action_frame": self.action_frame,
            "rotation_representation": self.rotation_representation,
            "action_units": ["m", "m", "m", "rad", "rad", "rad", "binary"],
            "gripper_open_value": self.gripper_open_value,
            "quaternion_order": "wxyz",
            "pose_convention": POSE_CONVENTION,
            "image_resolution": [w, h] if w is not None else list(self.image_wh),
            "num_observations": n,
            "num_actions": n - 1,
            "actions_file": None,
            "base_world_xyz": self._base_xyz.tolist(),
            "base_world_quat_wxyz": self._base_quat.tolist(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "note": "actions.npy is produced offline by make_actions.py",
        }
        with open(self.episode_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(
            "[recorder] finalized {} obs={} success={}".format(
                self.episode_dir, n, success
            )
        )
        return self.episode_dir
