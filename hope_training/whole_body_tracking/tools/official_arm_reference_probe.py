#!/usr/bin/env python3
"""Send one accepted arm reference through the official A3 HTTP interface.

Only the official MotionControlJointService is used here. Legs and waist are
left to the native MOTION controller. NPZ columns are the Isaac articulation
order and are mapped explicitly to the official A3 arm joint names.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests


ARM_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# NPZ is saved in the Isaac A3 articulation order, not the public arm API order.
NPZ_ARM_INDICES = [12, 17, 21, 23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30]

# A3 documents this as a continuous JointState channel in MOTION. The RPC
# with a similar name accepts requests but is not the streaming servo path.
SET_ARM_CHANNEL_URL = "http://127.0.0.1:56322/channel/%2Fmotion%2Fcontrol%2Farm_joint_command/ros2%3Asensor_msgs%2Fmsg%2FJointState"
GET_JOINTS_URL = "http://127.0.0.1:8001/joint_states"
GET_IMU_URL = "http://127.0.0.1:8001/imu"
GET_ACTION_URL = "http://127.0.0.1:56322/rpc/aimdk.protocol.MotionControlActionService/GetAction"


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
        "frame_id": "official_arm_reference_probe",
        "trace_id": "official_arm_reference_probe",
        "domin": "",
    }


def read_current_arm(session: requests.Session) -> np.ndarray:
    payload = session.get(GET_JOINTS_URL, timeout=5).json()
    values = {}
    for group in payload["joints"]:
        for joint in group:
            values[joint["name"]] = float(joint["position"])
    missing = [name for name in ARM_NAMES if name not in values]
    if missing:
        raise RuntimeError(f"official simulator did not publish arm joints: {missing}")
    return np.asarray([values[name] for name in ARM_NAMES], dtype=np.float64)


def read_action(session: requests.Session) -> str:
    response = session.post(GET_ACTION_URL, json={}, timeout=5)
    response.raise_for_status()
    body = response.json()
    return str(body.get("info", {}).get("current_action", ""))


def send_arm(
    session: requests.Session, positions: np.ndarray, sequence: int
) -> dict:
    response = session.post(
        SET_ARM_CHANNEL_URL,
        headers={"Content-Type": "application/json"},
        json={
            "header": header(sequence),
            "name": ARM_NAMES,
            "position": [float(x) for x in positions],
            "velocity": [0.0] * len(ARM_NAMES),
            "effort": [0.0] * len(ARM_NAMES),
        },
        timeout=5,
    )
    response.raise_for_status()
    return {"status_code": response.status_code, "body": response.text}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-npz", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--blend-seconds", type=float, default=1.0)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    args = parser.parse_args()

    data = np.load(args.motion_npz, allow_pickle=False)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 31:
        raise ValueError(f"expected joint_pos [T,31], got {joint_pos.shape}")
    target = joint_pos[:, NPZ_ARM_INDICES]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    action = read_action(session)
    if action != "MotionControlAction_MOTION":
        raise RuntimeError(
            "official A3 arm channel requires MOTION, "
            f"but current action is {action!r}"
        )
    current = read_current_arm(session)
    samples: list[dict] = []

    def publish_and_log(position: np.ndarray, phase: str, frame: int) -> None:
        response = send_arm(session, position, len(samples))
        record = {
            "time": time.time(),
            "phase": phase,
            "frame": frame,
            "command": position.tolist(),
            "response": response,
        }
        # 20 Hz state sampling is enough to evaluate posture without turning
        # the diagnostic HTTP service into the control loop.
        if len(samples) % 5 == 0:
            record["imu"] = session.get(GET_IMU_URL, timeout=5).json().get("imu")
            # Read back the official simulator state at the same cadence. This
            # separates API acceptance from actual arm tracking under MOTION.
            record["actual_arm"] = read_current_arm(session).tolist()
        samples.append(record)

    period = 0.01  # Official arm interface recommendation: 100 Hz.
    blend_steps = max(1, round(args.blend_seconds / period))
    blend_start = time.monotonic()
    for step in range(blend_steps):
        alpha = (step + 1) / blend_steps
        publish_and_log(current + alpha * (target[0] - current), "blend", step)
        time.sleep(max(0.0, blend_start + (step + 1) * period - time.monotonic()))

    # NPZ is 50 Hz; linearly interpolate it to the official 100 Hz command rate.
    reference_steps = max(1, round((len(target) - 1) / fps / period))
    reference = np.linspace(0.0, len(target) - 1, reference_steps + 1)
    reference_start = time.monotonic()
    for step, index in enumerate(reference):
        lo = int(np.floor(index))
        hi = min(lo + 1, len(target) - 1)
        alpha = index - lo
        position = (1.0 - alpha) * target[lo] + alpha * target[hi]
        publish_and_log(position, "reference", step)
        time.sleep(max(0.0, reference_start + (step + 1) * period - time.monotonic()))

    hold_steps = max(1, round(args.hold_seconds / period))
    for step in range(hold_steps):
        publish_and_log(target[-1], "hold", step)
        time.sleep(period)

    args.output.write_text(
        json.dumps(
            {
                "motion_npz": str(args.motion_npz),
                "fps": fps,
                "official_rate_hz": 100,
                "transport": "official_http_channel_ros2_sensor_msgs_JointState",
                "required_action": "MotionControlAction_MOTION",
                "action_at_start": action,
                "joint_names": ARM_NAMES,
                "npz_arm_indices": NPZ_ARM_INDICES,
                "samples": samples,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "samples": len(samples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
