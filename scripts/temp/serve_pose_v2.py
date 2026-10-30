#!/usr/bin/env python3
"""Serve a pose-v2 LoRA checkpoint with the training image preprocessing."""

import argparse
from base64 import b64decode, b64encode
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = PROJECT / "runs/ft_pose_v2/trial_20261018-163831-005902"
sys.path.insert(0, str(PROJECT.parent))
sys.path.insert(0, str(PROJECT / "scripts/dataset_raw"))


def numpy_object_hook(value):
    if "__numpy__" not in value:
        return value
    dtype = np.dtype(value["dtype"])
    if dtype.hasobject:
        raise ValueError("Object arrays are unsupported")
    array = np.frombuffer(b64decode(value["__numpy__"], validate=True), dtype=dtype)
    return array.reshape(value["shape"])


def numpy_default(value):
    if not isinstance(value, np.ndarray):
        raise TypeError(type(value).__name__)
    value = np.ascontiguousarray(value)
    return {"__numpy__": b64encode(value.tobytes()).decode("ascii"),
            "dtype": value.dtype.str, "shape": list(value.shape)}


def parse_payload(payload):
    double_encoded = "encoded" in payload
    if double_encoded:
        if set(payload) != {"encoded"}:
            raise ValueError("encoded must be the only payload field")
        payload = json.loads(payload["encoded"], object_hook=numpy_object_hook)
    else:
        payload = json.loads(json.dumps(payload), object_hook=numpy_object_hook)
    image = np.asarray(payload["image"])
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
        raise ValueError("Expected a nonempty HxWx3 RGB image")
    if image.dtype.kind not in "ui" or image.min() < 0 or image.max() > 255:
        raise ValueError("Expected RGB integer values in [0,255]")
    instruction = payload["instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A nonempty instruction is required")
    if payload.get("unnorm_key", "ur10e_push_sim") != "ur10e_push_sim":
        raise ValueError("Expected unnorm_key=ur10e_push_sim")
    return image.astype(np.uint8), instruction.strip().lower(), double_encoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run / "config.json").read_text())
    best = json.loads((run / "best.json").read_text())
    if best.get("adapter") is None or config.get("statistics_split") != "train":
        raise ValueError("Expected an improved pose-v2 checkpoint with train-only statistics")
    adapter = (run / best["adapter"]).resolve()
    if adapter.parent != run or not (adapter / "adapter_model.safetensors").is_file():
        raise ValueError("Missing or invalid adapter")
    statistics = json.loads((adapter / "dataset_statistics.json").read_text())
    manifest = json.loads((run / "split_manifest.json").read_text())
    if manifest != json.loads((PROJECT / "datasets/pose_v2_preparation/split_manifest.json").read_text()):
        raise ValueError("Dataset manifest mismatch")
    if statistics["ur10e_push_sim"]["num_trajectories"] != len(manifest["splits"]["train"]):
        raise ValueError("Statistics include unexpected episodes")
    with socket.socket() as probe:
        probe.bind((args.host, args.port))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    from pose_v2_training_utils import busy_gpu_processes
    gpu_rows = subprocess.run(
        ["nvidia-smi", "-i", "0", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    busy = busy_gpu_processes(gpu_rows, os.getpid())
    if busy:
        raise RuntimeError("Start the model service before Isaac Sim; GPU jobs already running: " + "; ".join(busy))

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import torch
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor
    import uvicorn
    from prismatic.vla.datasets.rlds.obs_transforms import decode_and_resize

    if not torch.cuda.is_available() or torch.cuda.mem_get_info(0)[0] < 16000 * 1024**2:
        raise RuntimeError("CUDA and at least 16000 MiB free GPU memory are required")
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(config["base_model"], trust_remote_code=True, local_files_only=True)
    core = AutoModelForVision2Seq.from_pretrained(
        config["base_model"], torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, local_files_only=True, attn_implementation="sdpa",
    ).to(device)
    model = PeftModel.from_pretrained(core, str(adapter), is_trainable=False)
    model.eval()
    core.norm_stats = statistics
    lock = threading.Lock()
    log_path = PROJECT / "logs" / datetime.now().strftime("pose_v2_server_%Y%m%d-%H%M%S-%f.jsonl")

    @torch.inference_mode()
    def predict(image, instruction):
        with lock:
            image224 = decode_and_resize({"image_primary": tf.convert_to_tensor(image, dtype=tf.uint8)},
                                         resize_size=(224, 224), depth_resize_size={})["image_primary"].numpy()
            prompt = f"In: What action should the robot take to {instruction}?\nOut:"
            inputs = processor(prompt, Image.fromarray(image224)).to(device, dtype=torch.bfloat16)
            action = np.asarray(core.predict_action(**inputs, unnorm_key="ur10e_push_sim", do_sample=False))
            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError("Model returned an invalid action")
            return action

    # Warm up using a recorded image; no ROS messages or robot commands are sent.
    first_episode = Path(manifest["raw_root"]) / manifest["splits"]["train"][0]
    warmup_image = np.asarray(Image.open(first_episode / "images/000000.png").convert("RGB"))
    instruction = json.loads((first_episode / "meta.json").read_text())["instruction"].lower()
    predict(warmup_image, instruction)
    health = {"ready": True, "pid": os.getpid(), "run": str(run), "adapter": str(adapter),
              "best_step": best["step"], "unnorm_key": "ur10e_push_sim", "statistics_split": "train",
              "preprocessing": "training_rlds_resize_224_then_processor", "action_frame": "ee",
              "rotation_representation": "rpy_left_multiply", "request_log": str(log_path)}
    app = FastAPI()

    @app.get("/health")
    def get_health():
        return health

    @app.post("/act")
    def get_action(payload: dict):
        start = time.monotonic()
        try:
            image, text, double_encoded = parse_payload(payload)
        except (ValueError, TypeError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        action = predict(image, text)
        with log_path.open("a") as stream:
            stream.write(json.dumps({"time": datetime.now().isoformat(), "image_shape": list(image.shape),
                                     "instruction": text, "action": action.tolist(),
                                     "request_seconds": time.monotonic() - start}) + "\n")
        if double_encoded:
            return JSONResponse(content=json.dumps(action, default=numpy_default))
        return JSONResponse(content=action.tolist())

    print("POSE_V2_MODEL_READY " + json.dumps(health), flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
