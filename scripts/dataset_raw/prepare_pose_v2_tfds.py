#!/usr/bin/env python3
"""Plan, pilot, build, and verify corrected TFDS data without GPU or training."""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = ROOT / "datasets/pose_v2_preparation"
DEFAULT_OUTPUT = Path("/home/dt/tensorflow_datasets_pose_v2")


def configure_resources():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name, value in {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                        "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1",
                        "NO_GCE_CHECK": "true"}.items():
        os.environ[name] = value
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed[:2])
    sys.path.insert(0, str(ROOT / "scripts"))


def tensorflow_runtime():
    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.config.set_visible_devices([], "GPU")
    if tf.config.get_visible_devices("GPU"):
        raise RuntimeError("GPU must remain disabled for dataset preparation")
    tfds.core.utils.gcs_utils._is_gcs_disabled = True
    return tf, tfds


def load_builder_class():
    path = ROOT / "scripts/rlds_ur10e_push_sim/ur10e_push_sim/pose_v2_builder.py"
    spec = importlib.util.spec_from_file_location("ur10e_pose_v2_builder", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Ur10ePushSimPoseV2


def verify_prepared(output_root, manifest, *, pilot=False):
    import numpy as np
    from PIL import Image
    from dataset_raw.tfds_pose_v2_manifest import load_episode_inputs
    tf, tfds = tensorflow_runtime()
    version_dir = output_root / "ur10e_push_sim/2.0.0"
    provenance = json.loads((version_dir / "pose_v2_preparation.json").read_text())
    if provenance != {"pilot": pilot, "manifest": manifest}:
        raise ValueError("Prepared dataset provenance does not match the frozen source/split")
    builder = tfds.builder_from_directory(str(version_dir))
    expected = {split: ids[:1] if pilot else ids for split, ids in manifest["splits"].items()}
    if set(builder.info.splits) != {"train", "val"}:
        raise ValueError("Prepared dataset must contain train and val")
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.autotune.enabled = False
    counts, seen_all = {}, set()
    for split, episode_ids in expected.items():
        if builder.info.splits[split].num_examples != len(episode_ids):
            raise ValueError(f"{split}: wrong episode count")
        data = builder.as_dataset(split=split, shuffle_files=False,
            decoders={"steps": {"observation": {"image": tfds.decode.SkipDecoding()}}})
        seen, total, image_bytes = set(), 0, 0
        for record in data.with_options(options):
            episode_id = record["episode_metadata"]["episode_id"].numpy().decode()
            if episode_id not in episode_ids or episode_id in seen_all:
                raise ValueError(f"Unexpected, duplicate, or overlapping episode: {episode_id}")
            episode = Path(manifest["raw_root"]) / episode_id
            meta, actions, poses = load_episode_inputs(episode)
            count = 0
            for index, step in enumerate(record["steps"].with_options(options).as_numpy_iterator()):
                if index >= len(actions):
                    raise ValueError(f"{episode_id}: too many steps")
                np.testing.assert_array_equal(step["action"], actions[index])
                state = np.concatenate([poses[index], [np.float32(meta["gripper_open_value"])]]).astype(np.float32)
                np.testing.assert_array_equal(step["observation"]["state"], state)
                last = index == len(actions) - 1
                if (bool(step["is_first"]) != (index == 0) or bool(step["is_last"]) != last
                        or bool(step["is_terminal"]) != last or float(step["reward"]) != float(last)
                        or float(step["discount"]) != 1.0):
                    raise ValueError(f"{episode_id}/{index}: wrong RLDS flags")
                if step["language_instruction"].decode() != meta["instruction"]:
                    raise ValueError(f"{episode_id}: wrong instruction")
                encoded = step["observation"]["image"]
                if not isinstance(encoded, bytes):
                    encoded = encoded.item()
                original = (episode / "images" / f"{index:06d}.png").read_bytes()
                if encoded != original:
                    raise ValueError(f"{episode_id}/{index}: image bytes changed")
                if index in {0, len(actions) // 2, len(actions) - 1}:
                    with Image.open(io.BytesIO(encoded)) as image:
                        if image.convert("RGB").size != (1280, 720):
                            raise ValueError("Decoded image dimensions changed")
                count += 1
                image_bytes += len(encoded)
            if count != len(actions):
                raise ValueError(f"{episode_id}: step count differs from raw actions")
            total += count
            seen.add(episode_id)
            seen_all.add(episode_id)
        if seen != set(episode_ids):
            raise ValueError(f"{split}: missing episodes")
        counts[split] = {"episodes": len(seen), "actions": total, "image_bytes": image_bytes}
    return {"passed": True, "pilot": pilot, "dataset_dir": str(version_dir), "counts": counts,
            "checks": ["episode membership/disjointness", "every action/state/flag/instruction",
                       "every image byte-for-byte", "first/middle/last image decoding per episode"],
            "scope": "TFDS serialization and episode split; not model accuracy or task generalization"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("plan", "pilot", "build", "verify"):
        modes.add_argument("--" + name, action="store_true")
    parser.add_argument("--raw", type=Path, default=ROOT / "datasets/raw_pose_v2")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    configure_resources()
    started = time.monotonic()
    from dataset_raw.tfds_pose_v2_manifest import freeze_manifest, validate_manifest
    state = args.state_dir.expanduser().resolve()
    state.mkdir(parents=True, exist_ok=True)
    manifest_path = state / "split_manifest.json"
    manifest = freeze_manifest(args.raw, manifest_path)
    print("MANIFEST", manifest_path, flush=True)
    print("SPLITS", json.dumps(manifest["counts"]), flush=True)
    print("RESOURCES", json.dumps({"cpu_affinity": sorted(os.sched_getaffinity(0)),
                                   "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"]}), flush=True)
    if args.plan:
        from dataset_raw.verify_pose_dataset import verify_dataset
        report = verify_dataset(args.raw)
        if not report["passed"]:
            raise RuntimeError("Corrected raw poses/actions failed FK consistency checks")
        report["source_fingerprint"] = manifest["source_fingerprint"]
        output = state / "pose_preflight_report.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
        summary = {"manifest": str(manifest_path), "splits": manifest["splits"], "counts": manifest["counts"],
                   "raw_image_gib": sum(ep["image_bytes"] for ep in manifest["episodes"]) / 1024**3,
                   "free_disk_gib": shutil.disk_usage(state).free / 1024**3, "pose_check_passed": True,
                   "production_output": str(args.data_dir / "ur10e_push_sim/2.0.0")}
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    preflight = json.loads((state / "pose_preflight_report.json").read_text())
    if not preflight.get("passed") or preflight.get("source_fingerprint") != manifest["source_fingerprint"]:
        raise ValueError("Run --plan to validate this exact source before preparing TFDS")
    output_root = (state / "pilot_tfds") if args.pilot else args.data_dir.expanduser().resolve()
    version_dir = output_root / "ur10e_push_sim/2.0.0"
    if args.build or args.pilot:
        if version_dir.exists():
            if os.environ.get("SHENLE_TFDS_OVERWRITE") == "1":
                shutil.rmtree(version_dir)
            else:
                raise FileExistsError(f"Refusing to overwrite {version_dir}; use --verify or SHENLE_TFDS_OVERWRITE=1")
        tensorflow_runtime()
        os.environ["UR10E_PUSH_RAW"] = manifest["raw_root"]
        os.environ["UR10E_PUSH_SPLIT_MANIFEST"] = str(manifest_path)
        builder = load_builder_class()(data_dir=str(output_root))
        import tensorflow_datasets as tfds
        builder.download_and_prepare(download_config=tfds.download.DownloadConfig(
            try_download_gcs=False, max_examples_per_split=1 if args.pilot else None))
        (version_dir / "pose_v2_preparation.json").write_text(
            json.dumps({"pilot": args.pilot, "manifest": manifest}, indent=2) + "\n")
        validate_manifest(manifest, args.raw)
    report = verify_prepared(output_root, manifest, pilot=args.pilot)
    report.update(elapsed_seconds=time.monotonic() - started,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  cpu_affinity=sorted(os.sched_getaffinity(0)), gpu_visible_devices=[])
    name = ("pilot" if args.pilot else "build" if args.build else "verify")
    report_path = state / (name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print("REPORT", report_path, flush=True)
    print("TFDS_CHECK_PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
