#!/usr/bin/env python3
"""Send the waist portion of an accepted NPZ through official A3 MOTION.

This is a SIL command-path probe. It does not replace the official motion
controller and does not claim real-robot calibration. NPZ waist columns are
explicitly mapped from Isaac articulation order to the public A3 waist names.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests


WAIST_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
NPZ_WAIST_INDICES = [2, 5, 8]
SET_WAIST_CHANNEL_URL = (
    "http://127.0.0.1:56322/channel/"
    "%2Fmotion%2Fcontrol%2Fwaist_joint_command/"
    "ros2%3Asensor_msgs%2Fmsg%2FJointState"
)
GET_JOINTS_URL = "http://127.0.0.1:8001/joint_states"
GET_IMU_URL = "http://127.0.0.1:8001/imu"
GET_ACTION_URL = "http://127.0.0.1:56322/rpc/aimdk.protocol.MotionControlActionService/GetAction"
SET_ACTION_URL = "http://127.0.0.1:56322/rpc/aimdk.protocol.MotionControlActionService/SetAction"
USE_EXT_CMD = "MotionControlAction_USE_EXT_CMD"


def header(sequence: int) -> dict:
    now = time.time()
    return {
        "seq": sequence,
        "timestamp": {
            "seconds": int(now),
            "nanos": int((now - int(now)) * 1e9),
            "ms_since_epoch": int(now * 1000),
        },
        "control_source": "ControlSource_SAFE",
        "uuid": "",
        "frame_id": "official_waist_reference_probe",
        "trace_id": "official_waist_reference_probe",
        "domin": "",
    }


def read_joints(session: requests.Session) -> dict[str, float]:
    payload = session.get(GET_JOINTS_URL, timeout=5).json()
    values = {}
    for group in payload["joints"]:
        for joint in group:
            values[joint["name"]] = float(joint["position"])
    missing = [name for name in WAIST_NAMES if name not in values]
    if missing:
        raise RuntimeError(f"official simulator did not publish waist joints: {missing}")
    return values


def read_action(session: requests.Session) -> str:
    response = session.post(GET_ACTION_URL, json={}, timeout=5)
    response.raise_for_status()
    return str(response.json().get("info", {}).get("current_action", ""))


def set_action(session: requests.Session, action_name: str) -> dict:
    """Use the protobuf JSON shape required by the official A3 service."""
    response = session.post(
        SET_ACTION_URL,
        headers={"Content-Type": "application/json"},
        # The local A3 protobuf JSON adapter accepts the command envelope;
        # its generated header schema differs from JointState headers.
        json={"command": {"action": USE_EXT_CMD, "ext_action": action_name}},
        timeout=5,
    )
    response.raise_for_status()
    return {"status_code": response.status_code, "body": response.text}


def read_imu(session: requests.Session) -> dict:
    response = session.get(GET_IMU_URL, timeout=5)
    response.raise_for_status()
    return response.json()


def send_waist(session: requests.Session, positions: np.ndarray, sequence: int) -> dict:
    response = session.post(
        SET_WAIST_CHANNEL_URL,
        headers={"Content-Type": "application/json"},
        json={
            "header": header(sequence),
            "name": WAIST_NAMES,
            "position": [float(x) for x in positions],
            "velocity": [0.0] * len(WAIST_NAMES),
            "effort": [0.0] * len(WAIST_NAMES),
        },
        timeout=5,
    )
    response.raise_for_status()
    return {"status_code": response.status_code, "body": response.text}


def summarize_tracking(
    samples: list[dict], joint_count: int, active_action: str
) -> dict:
    """Separate HTTP acceptance from observed actuator/state movement."""
    if not samples:
        return {"status": "no_samples"}
    baseline = np.asarray(samples[0]["actual"], dtype=np.float64)
    actual = np.asarray([sample["actual"] for sample in samples], dtype=np.float64)
    command = np.asarray([sample["command"] for sample in samples], dtype=np.float64)
    error = np.asarray([sample["error"] for sample in samples], dtype=np.float64)
    actual_span = np.max(np.abs(actual - baseline), axis=0)
    target_span = np.max(np.abs(command - baseline), axis=0)
    error_abs = np.abs(error)
    # A command is considered observed only when a meaningful fraction of the
    # requested excursion appears in the official simulator state. A 200
    # response alone only proves that the HTTP channel parsed the message.
    observed = actual_span >= np.maximum(0.02, 0.25 * target_span)
    return {
        "status": "observed" if bool(np.any(observed)) else "accepted_but_not_observed",
        "joint_observed": [bool(value) for value in observed],
        "baseline_actual_rad": baseline.tolist(),
        "actual_span_from_baseline_rad": actual_span.tolist(),
        "requested_target_span_from_baseline_rad": target_span.tolist(),
        "mean_abs_tracking_error_rad": np.mean(error_abs, axis=0).tolist(),
        "max_abs_tracking_error_rad": np.max(error_abs, axis=0).tolist(),
        "interpretation": (
            f"HTTP channel accepted the messages, but no meaningful waist state "
            f"movement was observed while {active_action or 'the requested action'} "
            "was active; the external target is overridden or not consumed by "
            "this official SIL path."
            if not bool(np.any(observed))
            else "At least one waist joint state followed a meaningful fraction of the requested target."
        ),
        "joint_count": joint_count,
    }


def resample_reference(reference: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Resample the NPZ trajectory without changing its physical duration."""
    if reference.ndim != 2 or reference.shape[0] < 2:
        raise ValueError(f"reference must be [T,J] with T>=2, got {reference.shape}")
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"source_fps must be positive, got {source_fps}")
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if abs(source_fps - target_fps) < 1.0e-9:
        return reference.copy()
    source_time = np.arange(reference.shape[0], dtype=np.float64) / source_fps
    step = 1.0 / target_fps
    target_time = np.arange(0.0, source_time[-1] + 0.5 * step, step, dtype=np.float64)
    return np.column_stack(
        [np.interp(target_time, source_time, reference[:, joint]) for joint in range(reference.shape[1])]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-npz", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--blend-seconds", type=float, default=1.0)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument(
        "--action",
        choices=["MOTION", "PD_STAND"],
        default=None,
        help="optionally switch to this official action before probing",
    )
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    args = parser.parse_args()

    data = np.load(args.motion_npz, allow_pickle=False)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 31:
        raise ValueError(f"expected joint_pos [T,31], got {joint_pos.shape}")
    reference = resample_reference(joint_pos[:, NPZ_WAIST_INDICES], fps, args.rate_hz)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    action_at_start = read_action(session)
    action_transition = None
    if args.action:
        action_transition = {
            "requested": args.action,
            "response": set_action(session, args.action),
        }
        time.sleep(max(0.0, args.settle_seconds))
    action_after_request = read_action(session)
    current_map = read_joints(session)
    current = np.asarray([current_map[name] for name in WAIST_NAMES], dtype=np.float64)

    samples = []
    sequence = 0
    dt = 1.0 / float(args.rate_hz)

    def publish(positions: np.ndarray, phase: str, frame: int) -> None:
        nonlocal sequence
        response = send_waist(session, positions, sequence)
        sequence += 1
        actual_map = read_joints(session)
        actual = np.asarray([actual_map[name] for name in WAIST_NAMES], dtype=np.float64)
        samples.append(
            {
                "time": time.time(),
                "phase": phase,
                "frame": int(frame),
                "command": positions.tolist(),
                "actual": actual.tolist(),
                "error": (actual - positions).tolist(),
                "response": response,
                "imu": read_imu(session),
            }
        )

    blend_steps = max(1, int(round(args.blend_seconds * args.rate_hz)))
    for step in range(blend_steps):
        alpha = (step + 1) / blend_steps
        publish((1.0 - alpha) * current + alpha * reference[0], "blend", 0)
        time.sleep(dt)

    for frame, positions in enumerate(reference):
        publish(positions, "reference", frame)
        time.sleep(dt)

    hold_steps = max(0, int(round(args.hold_seconds * args.rate_hz)))
    for step in range(hold_steps):
        publish(reference[-1], "hold", step)
        time.sleep(dt)

    output = {
        "motion_npz": str(args.motion_npz),
        "fps": fps,
        "official_rate_hz": args.rate_hz,
        "resampled_frame_count": int(reference.shape[0]),
        "duration_s": float((reference.shape[0] - 1) / args.rate_hz),
        "transport": "official_http_channel_ros2_sensor_msgs_JointState",
        "requested_action": args.action or "unchanged",
        "action_at_start": action_at_start,
        "action_request": action_transition,
        "action_after_request": action_after_request,
        "joint_names": WAIST_NAMES,
        "npz_waist_indices": NPZ_WAIST_INDICES,
        "tracking_summary": summarize_tracking(
            samples, len(WAIST_NAMES), action_after_request
        ),
        "samples": samples,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "samples": len(samples)}))


if __name__ == "__main__":
    main()
