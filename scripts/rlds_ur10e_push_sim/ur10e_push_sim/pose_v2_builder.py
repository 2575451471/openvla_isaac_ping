"""Additive TFDS 2.0.0 builder; the legacy builder and prepared 1.0.0 stay intact."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import tensorflow_datasets as tfds

SCRIPTS = Path(__file__).resolve().parents[2]
for path in (SCRIPTS, SCRIPTS / "rlds_ur10e_push_sim"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from dataset_raw.tfds_pose_v2_manifest import load_episode_inputs, validate_manifest
from ur10e_push_sim.ur10e_push_sim_dataset_builder import Ur10ePushSim


class Ur10ePushSimPoseV2(Ur10ePushSim, skip_registration=True):
    name = "ur10e_push_sim"
    VERSION = tfds.core.Version("2.0.0")
    RELEASE_NOTES = {"2.0.0": "Corrected column-convention poses/actions with frozen episode-level train/val split."}

    def _split_generators(self, dl_manager):
        raw = Path(os.environ["UR10E_PUSH_RAW"])
        manifest = json.loads(Path(os.environ["UR10E_PUSH_SPLIT_MANIFEST"]).read_text())
        validate_manifest(manifest, raw)
        return {split: self._generate_split(raw, ids) for split, ids in manifest["splits"].items()}

    def _generate_split(self, raw, episode_ids):
        for episode_id in episode_ids:
            yield episode_id, self._parse_episode(raw / episode_id)

    def _parse_episode(self, episode):
        meta, actions, poses = load_episode_inputs(episode)
        gripper = np.float32(meta["gripper_open_value"])
        steps = []
        for index, action in enumerate(actions):
            image = episode / "images" / f"{index:06d}.png"
            if not image.is_file():
                raise FileNotFoundError(image)
            state = np.concatenate([poses[index], [gripper]]).astype(np.float32)
            last = index == len(actions) - 1
            steps.append({
                "observation": {"image": str(image), "state": state},
                "action": action, "discount": np.float32(1), "reward": np.float32(last),
                "is_first": index == 0, "is_last": last, "is_terminal": last,
                "language_instruction": str(meta["instruction"]),
                "language_embedding": np.zeros(512, dtype=np.float32),
            })
        return {"steps": steps, "episode_metadata": {"file_path": str(episode), "episode_id": episode.name}}
