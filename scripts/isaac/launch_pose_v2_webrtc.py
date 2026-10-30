#!/usr/bin/env python3
"""Start a private WebRTC Isaac instance and wait for scene reset readiness."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs/pose_v2_webrtc"
LATEST = LOGS / "latest.json"


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def process_birth(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def owned_alive(record):
    return (process_birth(record["pid"]) == record["process_birth"]
            and Path("/proc/sys/kernel/random/boot_id").read_text().strip() == record["boot_id"])


def existing_processes():
    result = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = [value for value in path.read_bytes().decode(errors="replace").split("\0") if value]
            if not argv or "-c" in argv[:3]:
                continue
            kit = Path(argv[0]).name == "kit" and any("isaacsim" in a for a in argv[:5])
            controller = any(Path(a).name in {"openvla_closed_loop.py", "replay_actions_ros.py"}
                             for a in argv[:4])
            if kit or controller:
                result.append({"pid": int(path.parent.name), "kind": "Isaac" if kit else "controller"})
        except OSError:
            continue
    return result


def check_port(port, kind):
    with socket.socket(socket.AF_INET, kind) as connection:
        connection.bind(("0.0.0.0", port))


def build_command(args, root=ROOT):
    return [str(args.isaac_dir / "isaac-sim.streaming.sh"),
            "--/app/livestream/publicEndpointAddress=" + args.address,
            "--/app/livestream/port=" + str(args.port),
            "--/app/file/ignoreUnsavedOnExit=true",
            "--enable", "isaacsim.ros2.bridge",
            "--exec", str(root / "scripts/isaac/prepare_pose_v2_scene.py")]


def preflight(args):
    required = [args.isaac_dir / "isaac-sim.streaming.sh", args.stage,
                ROOT / "scripts/isaac/prepare_pose_v2_scene.py",
                ROOT / "scripts/isaac/replay_action_episode.py",
                ROOT / "scripts/isaac/replay_recorded_episode.py"]
    if getattr(args, "mode", "reset") == "collect":
        required.append(ROOT / "scripts/isaac/collect_push_batch.py")
    elif getattr(args, "mode", "reset") == "play":
        required.append(ROOT / "scripts/isaac/arm_motion.py")
    else:
        required += [ROOT / "datasets/raw_pose_v2/episode_000001" / name for name in
                     ("meta.json", "joint_positions.npy", "timestamps.npy", "cube_poses_world.npy")]
    for path in required:
        if not path.is_file():
            raise RuntimeError("Required file missing: " + str(path))
    if not os.access(required[0], os.X_OK):
        raise RuntimeError("Isaac streaming launcher is not executable")
    if any(character in args.address for character in "\r\n\0"):
        raise RuntimeError("Invalid streaming address")
    current = existing_processes()
    if current:
        raise RuntimeError("Existing Isaac/controller processes; close your old session normally first: " + json.dumps(current))
    check_port(args.port, socket.SOCK_STREAM)
    check_port(47998, socket.SOCK_DGRAM)
    return build_command(args)


def state():
    if not LATEST.exists():
        return {"running": False, "message": "No instance launched by this entry point yet"}
    record = json.loads(LATEST.read_text())
    record["running"] = owned_alive(record)
    status = Path(record["status"])
    if status.exists():
        record["preparation"] = json.loads(status.read_text())
    return record


def terminate(record):
    if not owned_alive(record):
        return
    argv = Path(f"/proc/{record['pid']}/cmdline").read_bytes().split(b"\0")
    if str(ROOT / "scripts/isaac/prepare_pose_v2_scene.py").encode() not in argv:
        raise RuntimeError("Bootstrap argument differs; refusing to signal this process")
    # Only the new session created here is signalled; no pkill or shared service.
    if os.getpgid(record["pid"]) != record["pid"]:
        raise RuntimeError("Process group identity differs; refusing to signal it")
    os.killpg(record["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 20
    while owned_alive(record) and time.monotonic() < deadline:
        time.sleep(0.2)
    if owned_alive(record):
        raise RuntimeError("Shutdown requested; instance is still stopping, inspect --status")


def stop():
    if not LATEST.exists():
        print("No instance launched by this entry point")
        return
    record = json.loads(LATEST.read_text())
    terminate(record)
    print("Stopped this launcher instance; stage file and shared Isaac installations preserved")


def launch(args, command):
    LOGS.mkdir(parents=True, exist_ok=True)
    lock = (LOGS / "launch.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("This launcher is already preparing an instance")
    record, child = None, None
    ready = False
    try:
        # Repeat under the lock to avoid simultaneous starts of this entry point.
        preflight(args)
        run = LOGS / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run.mkdir()
        status_path = run / "status.json"
        kit_log = run / "kit.log"
        save(status_path, {"phase": "STARTING_ISAAC"})
        env = dict(os.environ, SHENLE_WEBRTC_ROOT=str(ROOT), SHENLE_WEBRTC_STAGE=str(args.stage),
                   SHENLE_WEBRTC_STATUS=str(status_path),
                   SHENLE_WEBRTC_TIMEOUT=str(int(args.timeout if args.mode == "collect" else 180)),
                   SHENLE_WEBRTC_RESET_STEP=str(args.reset_step),
                   SHENLE_WEBRTC_MODE=args.mode)
        with kit_log.open("wb") as output:
            child = subprocess.Popen(command, cwd=args.isaac_dir, env=env,
                                     stdin=subprocess.DEVNULL, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        birth = process_birth(child.pid)
        if birth is None:
            raise RuntimeError("Isaac exited immediately; see " + str(kit_log))
        record = {"pid": child.pid, "process_birth": birth, "status": str(status_path),
                  "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                  "log": str(kit_log), "command": command, "stage": str(args.stage),
                  "address": args.address, "port": args.port, "reset_step": args.reset_step,
                  "mode": args.mode}
        save(LATEST, record)
        save(run / "launch.json", record)
        print("Kit PID:", child.pid, flush=True)
        print("Log:", kit_log, flush=True)
        deadline, heartbeat, last_phase = time.monotonic() + args.timeout, 0, None
        waiting_webrtc = False
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError("Isaac exited with code " + str(child.returncode) + "; see " + str(kit_log))
            current = json.loads(status_path.read_text())
            phase = current["phase"]
            if phase != last_phase:
                print("[pose-v2-start] " + phase, flush=True)
                last_phase = phase
            if phase == "ERROR":
                raise RuntimeError(current.get("error", "Scene preparation failed"))
            if phase == "READY":
                try:
                    with socket.create_connection((args.address, args.port), timeout=0.5):
                        pass
                except OSError:
                    if not waiting_webrtc:
                        print("Scene reset passed; waiting for the WebRTC signaling port", flush=True)
                        waiting_webrtc = True
                    time.sleep(0.5)
                    continue
                ready = True
                if args.mode == "collect":
                    print("READY: collection finished. Isaac stays running. No OpenVLA.", flush=True)
                elif args.mode == "play":
                    print("READY: gripper scene is playing at HOME. No OpenVLA yet.", flush=True)
                else:
                    print("READY: scene loaded, Play active, reset verified.", flush=True)
                    print("Isaac stays running after this command returns. No OpenVLA actions started.", flush=True)
                print("Connect your WebRTC client to " + args.address + ":" + str(args.port), flush=True)
                return
            if time.monotonic() - heartbeat > 30:
                print("Waiting for Isaac; details in " + str(kit_log), flush=True)
                heartbeat = time.monotonic()
            time.sleep(0.5)
        raise TimeoutError("Startup deadline exceeded; see " + str(kit_log))
    finally:
        try:
            if not ready and child is not None:
                if record:
                    terminate(record)
                elif child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=25)
        finally:
            lock.close()


def interrupted(signum, frame):
    raise KeyboardInterrupt("Received signal " + str(signum))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Preflight only; no Kit/GPU/scene startup")
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--stop", action="store_true", help="Stop only the instance started here")
    parser.add_argument("--isaac-dir", type=Path, default=Path("/home/dt/isaacsim5.1"))
    parser.add_argument("--stage", type=Path, default=ROOT / "assets/sence/sence1_gripper.usd")
    parser.add_argument("--address", default="100.64.61.8")
    parser.add_argument("--port", type=int, default=49100)
    parser.add_argument("--timeout", type=float, default=0,
                        help="Seconds to wait for READY; 0 uses 600 for reset, 5400 for collect")
    parser.add_argument("--reset-step", type=int, default=0,
                        help="Recorded frame of episode_000001 to reset robot and cube to")
    parser.add_argument("--mode", choices=("reset", "collect", "play"), default="reset")
    args = parser.parse_args()
    if args.timeout <= 0:
        args.timeout = 5400 if args.mode == "collect" else 600
    if not 1 <= args.port <= 65535 or args.timeout <= 0 or args.reset_step < 0:
        parser.error("Invalid port, timeout, or reset step")
    args.isaac_dir, args.stage = args.isaac_dir.resolve(), args.stage.resolve()
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.status:
            print(json.dumps(state(), ensure_ascii=False, indent=2))
        elif args.stop:
            stop()
        else:
            command = preflight(args)
            print("Command:", shlex.join(command), flush=True)
            if args.check:
                print("CHECK_PASSED: no Isaac process started and no scene reset performed")
            else:
                launch(args, command)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print("[pose-v2-start] ERROR:", str(error), file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    sys.exit(main())
