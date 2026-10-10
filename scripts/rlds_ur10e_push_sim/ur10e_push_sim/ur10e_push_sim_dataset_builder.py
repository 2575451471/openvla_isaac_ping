"""TFDS / RLDS builder: shenle_projects raw episodes -> ur10e_push_sim.

Raw layout (per episode):
  images/000000.png ...
  actions.npy          (N-1, 7)
  ee_poses_world.npy   (N, 7) xyz + quat wxyz
  joint_positions.npy  (N, 6)
  meta.json

Build (in an env with tensorflow_datasets, from THIS package directory):
  export UR10E_PUSH_RAW=/home/dt/openvla/shenle_projects/datasets/raw
  export TFDS_DATA_DIR=$HOME/tensorflow_datasets
  cd .../scripts/rlds_ur10e_push_sim/ur10e_push_sim
  tfds build --overwrite

Then OpenVLA finetune:
  --data_root_dir $TFDS_DATA_DIR --dataset_name ur10e_push_sim
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator, Tuple

import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

# Local-only dataset: skip GCS checksum / dataset_info lookups (hangs offline).
tfds.core.utils.gcs_utils._is_gcs_disabled = True  # noqa: SLF001


def _raw_root() -> Path:
    env = os.environ.get("UR10E_PUSH_RAW")
    if env:
        return Path(env)
    return Path("/home/dt/openvla/shenle_projects/datasets/raw")


class Ur10ePushSim(tfds.core.GeneratorBasedBuilder):
    """Single third-person RGB, 7-DoF EEF delta actions (OpenVLA EEF_POS)."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Raw Isaac Sim UR10e push-cube demos from shenle_projects.",
    }

    # Fixed image size from current capture (H, W, C). Rebuild if you change resolution.
    IMAGE_SHAPE = (720, 1280, 3)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "UR10e push red cube in Isaac Sim. Actions are EE-frame "
                "delta xyz + RPY (left-multiply) + gripper, matching "
                "scripts/ros/openvla_closed_loop.py."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=self.IMAGE_SHAPE,
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Third-person RGB (primary).",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(8,),
                                        dtype=np.float32,
                                        doc=(
                                            "EEF XYZ (3) + quat wxyz (4) + gripper (1). "
                                            "Matches StateEncoding.POS_QUAT packing."
                                        ),
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc=(
                                    "dx,dy,dz [m] in EE frame; droll,dpitch,dyaw [rad] "
                                    "as RPY left-multiply; gripper open≈1."
                                ),
                            ),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                            "language_embedding": tfds.features.Tensor(
                                shape=(512,),
                                dtype=np.float32,
                                doc="Unused placeholder zeros (OpenVLA uses text).",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(),
                            "episode_id": tfds.features.Text(),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage="file:///home/dt/openvla/shenle_projects",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        root = _raw_root()
        return {
            "train": self._generate_examples(root),
        }

    def _generate_examples(self, root: Path) -> Iterator[Tuple[str, Any]]:
        root = Path(root)
        episodes = sorted(p for p in root.glob("episode_*") if p.is_dir())
        if not episodes:
            raise FileNotFoundError(
                f"No episode_* under {root}. Set UR10E_PUSH_RAW or record first."
            )

        for ep in episodes:
            sample = self._parse_episode(ep)
            if sample is None:
                continue
            yield ep.name, sample

    def _parse_episode(self, ep: Path):
        meta_path = ep / "meta.json"
        actions_path = ep / "actions.npy"
        poses_path = ep / "ee_poses_world.npy"
        images_dir = ep / "images"
        if not (meta_path.is_file() and actions_path.is_file() and poses_path.is_file()):
            print(f"[skip] incomplete episode: {ep}")
            return None

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("success") is False:
            print(f"[skip] success=false: {ep}")
            return None

        actions = np.load(actions_path).astype(np.float32)
        poses = np.load(poses_path).astype(np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"{ep}: actions shape {actions.shape}")
        if poses.ndim != 2 or poses.shape[1] != 7:
            raise ValueError(f"{ep}: poses shape {poses.shape}")

        n_act = actions.shape[0]
        # Need image_0 .. image_{n_act-1} (and optionally image_n_act unused for labels)
        instruction = str(meta.get("instruction", "push the red cube to the right"))
        gripper = float(meta.get("gripper_open_value", 1.0))
        lang_emb = np.zeros((512,), dtype=np.float32)

        steps = []
        for i in range(n_act):
            img_path = images_dir / f"{i:06d}.png"
            if not img_path.is_file():
                raise FileNotFoundError(img_path)
            img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            if img.shape != self.IMAGE_SHAPE:
                raise ValueError(
                    f"{img_path}: got {img.shape}, builder expects {self.IMAGE_SHAPE}. "
                    "Update IMAGE_SHAPE or resize images."
                )

            pose = poses[i]
            state = np.concatenate(
                [pose[:3], pose[3:7], np.array([gripper], dtype=np.float32)]
            ).astype(np.float32)

            steps.append(
                {
                    "observation": {
                        "image": img,
                        "state": state,
                    },
                    "action": actions[i],
                    "discount": np.float32(1.0),
                    "reward": np.float32(1.0 if i == n_act - 1 else 0.0),
                    "is_first": i == 0,
                    "is_last": i == n_act - 1,
                    "is_terminal": i == n_act - 1,
                    "language_instruction": instruction,
                    "language_embedding": lang_emb,
                }
            )

        return {
            "steps": steps,
            "episode_metadata": {
                "file_path": str(ep),
                "episode_id": str(meta.get("episode_id", ep.name)),
            },
        }
