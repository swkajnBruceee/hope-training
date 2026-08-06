#!/usr/bin/env python3
"""Measure the official A3 MOTION controller's lateral-speed behavior in SIL.

This diagnostic publishes only the vendor-defined high-level locomotion
velocity channel.  It never publishes leg or waist joint commands.  A zero
velocity command is always sent before the process exits.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


MC_ENDPOINT = "http://127.0.0.1:56322"
SIM_ENDPOINT = "http://127.0.0.1:8001"
LOCOMOTION_CHANNEL = (
    "/channel/%2Fmotion%2Fcontrol%2Flocomotion_velocity/"
    "pb%3Aaimdk.protocol.MotionControlLocomotionVelocityChannel"
)
GET_ACTION_PATH = "/rpc/aimdk.protocol.MotionControlActionService/GetAction"


def post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=3) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def get_json(url: str) -> dict:
    with urlopen(url, timeout=3) as response:
        return json.loads(response.read().decode("utf-8"))


def header(sequence: int) -> dict:
    timestamp = time.time()
    seconds = int(timestamp)
    return {
        "seq": sequence,
        "timestamp": {
            "seconds": seconds,
            "nanos": int((timestamp - seconds) * 1_000_000_000),
            "ms_since_epoch": int(timestamp * 1000),
        },
        "frame_id": "hope_a3_lateral_speed_probe",
        "control_source": "ControlSource_SAFE",
        "uuid": "",
    }


def velocity_payload(sequence: int, lateral_velocity_mps: float, mode: int) -> dict:
    return {
        "header": header(sequence),
        "data": {
            "mode": mode,
            "forward_velocity": 0.0,
            "lateral_velocity": lateral_velocity_mps,
            "angular_velocity": 0.0,
        },
    }


def base_yaw(rotation: dict) -> float:
    """Extract yaw from the simulator's w/x/y/z base orientation."""
    w, x, y, z = (float(rotation[key]) for key in ("w", "x", "y", "z"))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def action_state(endpoint: str) -> str:
    return str(post_json(f"{endpoint}{GET_ACTION_PATH}", {}).get("info", {}).get("current_action", ""))


def pose_sample(simulator_endpoint: str, elapsed_s: float, initial_yaw_rad: float) -> dict:
    base = get_json(f"{simulator_endpoint}/baselink_position")
    imu = get_json(f"{simulator_endpoint}/imu")
    position = [float(value) for value in base["position"]]
    cos_yaw, sin_yaw = math.cos(initial_yaw_rad), math.sin(initial_yaw_rad)
    # Project the world x/y displacement into the initial base horizontal frame.
    base_x = cos_yaw * position[0] + sin_yaw * position[1]
    base_y = -sin_yaw * position[0] + cos_yaw * position[1]
    pelvis = imu["imu"]["pelvis"]
    angular_speed = math.sqrt(sum(float(value) ** 2 for value in pelvis["angular_velocity"]))
    return {
        "elapsed_s": elapsed_s,
        "position_world_m": position,
        "position_initial_base_xy_m": [base_x, base_y],
        "pelvis_angular_speed_rad_s": angular_speed,
    }


def summarize(samples: list[dict], command_duration_s: float) -> dict:
    if len(samples) < 2:
        raise RuntimeError("need at least two simulator samples")
    start = samples[0]["position_initial_base_xy_m"]
    end = samples[-1]["position_initial_base_xy_m"]
    command_samples = [sample for sample in samples if sample["elapsed_s"] <= command_duration_s]
    lateral = [sample["position_initial_base_xy_m"][1] for sample in command_samples]
    peak_speed = 0.0
    for previous, current in zip(command_samples, command_samples[1:]):
        dt = current["elapsed_s"] - previous["elapsed_s"]
        if dt > 1.0e-4:
            peak_speed = max(
                peak_speed,
                abs((current["position_initial_base_xy_m"][1] - previous["position_initial_base_xy_m"][1]) / dt),
            )
    return {
        "net_displacement_initial_base_m": [end[0] - start[0], end[1] - start[1]],
        "command_phase_lateral_displacement_m": lateral[-1] - lateral[0],
        "command_phase_peak_lateral_speed_mps": peak_speed,
        "max_pelvis_angular_speed_rad_s": max(sample["pelvis_angular_speed_rad_s"] for sample in samples),
        "sample_count": len(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lateral-velocity-mps", type=float, default=0.3)
    parser.add_argument("--mode", type=int, default=20)
    parser.add_argument("--command-duration-s", type=float, default=4.0)
    parser.add_argument("--settle-s", type=float, default=3.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--mc-endpoint", default=MC_ENDPOINT)
    parser.add_argument("--simulator-endpoint", default=SIM_ENDPOINT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 0.0 < abs(args.lateral_velocity_mps) <= 0.3:
        parser.error("lateral velocity must be in [-0.3, 0.3] m/s and non-zero")
    if min(args.command_duration_s, args.settle_s, args.rate_hz) <= 0.0:
        parser.error("duration and rate values must be positive")
    if action_state(args.mc_endpoint) != "MotionControlAction_MOTION":
        raise RuntimeError("refusing lateral probe unless official MOTION is active")

    initial_base = get_json(f"{args.simulator_endpoint}/baselink_position")
    initial_yaw_rad = base_yaw(initial_base["rotation"])
    samples: list[dict] = []
    period_s = 1.0 / args.rate_hz
    sequence = 0
    start_s = time.monotonic()
    exception: Exception | None = None
    try:
        while True:
            elapsed_s = time.monotonic() - start_s
            if elapsed_s >= args.command_duration_s + args.settle_s:
                break
            command = args.lateral_velocity_mps if elapsed_s < args.command_duration_s else 0.0
            post_json(f"{args.mc_endpoint}{LOCOMOTION_CHANNEL}", velocity_payload(sequence, command, args.mode))
            samples.append(pose_sample(args.simulator_endpoint, elapsed_s, initial_yaw_rad))
            sequence += 1
            time.sleep(max(0.0, start_s + sequence * period_s - time.monotonic()))
    except Exception as error:
        exception = error
    finally:
        # Send several zero messages so a failed HTTP poll cannot leave a stale
        # maximum-velocity command in the official controller.
        for _ in range(3):
            try:
                post_json(f"{args.mc_endpoint}{LOCOMOTION_CHANNEL}", velocity_payload(sequence, 0.0, args.mode))
            except Exception:
                pass
            sequence += 1
            time.sleep(0.05)

    if exception is not None:
        raise exception
    final_action = action_state(args.mc_endpoint)
    report = {
        "artifact_status": "local_sil_diagnostic_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": {
            "lateral_velocity_mps": args.lateral_velocity_mps,
            "duration_s": args.command_duration_s,
            "settle_s": args.settle_s,
            "rate_hz": args.rate_hz,
            "mode": args.mode,
            "channel": "/motion/control/locomotion_velocity",
        },
        "initial_base_yaw_rad": initial_yaw_rad,
        "final_action": final_action,
        "summary": summarize(samples, args.command_duration_s),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if final_action != "MotionControlAction_MOTION":
        raise RuntimeError(f"official action changed during probe: {final_action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
