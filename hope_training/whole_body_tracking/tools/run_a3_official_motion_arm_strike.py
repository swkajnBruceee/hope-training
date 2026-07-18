#!/usr/bin/env python3
"""Run a local-SIL A3 strike with native MOTION balance and right-arm control.

The official MotionControl process retains ownership of both legs, waist,
trunk, head, and the non-paddle arm.  This tool only publishes the 14-element
official arm command, holding the left arm at its initial measured posture and
following the canonical command with the right arm.  It refuses to publish a
stroke until MOTION is active and the pelvis has been still for a complete
preflight window.  During the stroke, excessive relative pelvis tilt or body
angular speed immediately stops further arm publication and records a failed
SIL result.

This is strictly local SIL: it refuses any SIM_MODE other than ``sil`` and
uses only loopback HTTP endpoints exposed by the official AimSim stack.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests


ARM_NAMES = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
RIGHT_ARM_NAMES = ARM_NAMES[7:]
MOTION_ACTION = "MotionControlAction_MOTION"
DEFAULT_MOTION_ENDPOINT = "http://127.0.0.1:56322"
DEFAULT_SIM_ENDPOINT = "http://127.0.0.1:8001"


def normalized_quaternion(values: Any) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("IMU orientation must be four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("IMU orientation must have non-zero norm")
    return quaternion / norm


def relative_tilt_deg(reference: np.ndarray, current: np.ndarray) -> float:
    """Quaternion-angle change independent of whether the service uses xyzw/wxyz."""

    dot = abs(float(np.dot(normalized_quaternion(reference), normalized_quaternion(current))))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))


def vector_norm(values: Any, name: str) -> float:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be three finite values")
    return float(np.linalg.norm(vector))


def request_header(sequence: int) -> dict[str, Any]:
    now = datetime.now(timezone.utc).timestamp()
    seconds = int(now)
    return {
        "seq": sequence,
        "timestamp": {"seconds": seconds, "nanos": int((now - seconds) * 1.0e9), "ms_since_epoch": int(now * 1000)},
        "control_source": "ControlSource_SAFE",
        "uuid": "",
        "frame_id": "a3_official_motion_arm_strike",
        "trace_id": "a3_official_motion_arm_strike",
        "domin": "",
    }


def read_action(session: requests.Session, endpoint: str) -> str:
    response = session.post(f"{endpoint}/rpc/aimdk.protocol.MotionControlActionService/GetAction", json={}, timeout=3)
    response.raise_for_status()
    return str(response.json().get("info", {}).get("current_action", ""))


def read_joint_positions(session: requests.Session, simulator_endpoint: str) -> dict[str, float]:
    response = session.get(f"{simulator_endpoint}/joint_states", timeout=3)
    response.raise_for_status()
    values: dict[str, float] = {}
    for group in response.json().get("joints", []):
        for joint in group:
            values[str(joint["name"])] = float(joint["position"])
    missing = [name for name in ARM_NAMES if name not in values]
    if missing:
        raise RuntimeError(f"official simulator omitted arm joint states: {missing}")
    return values


def read_stability_state(session: requests.Session, simulator_endpoint: str) -> dict[str, Any]:
    response = session.get(f"{simulator_endpoint}/imu", timeout=3)
    response.raise_for_status()
    imu = response.json().get("imu", {})
    pelvis = imu.get("pelvis") or {}
    torso = imu.get("torso") or {}
    return {
        "pelvis_orientation": normalized_quaternion(pelvis.get("orientation")),
        "pelvis_angular_speed_rad_s": vector_norm(pelvis.get("angular_velocity"), "pelvis angular velocity"),
        "torso_angular_speed_rad_s": vector_norm(torso.get("angular_velocity"), "torso angular velocity"),
    }


def read_body_pose(session: requests.Session, simulator_endpoint: str, body_name: str) -> dict[str, Any]:
    """Read an arbitrary MuJoCo body's world pose from the official SIL API."""

    response = session.get(
        f"{simulator_endpoint}/baselink_position", params={"body_name": body_name}, timeout=3
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("status"):
        raise RuntimeError(f"simulator did not resolve tracked body {body_name!r}: {payload.get('message')}")
    position = np.asarray(payload.get("position"), dtype=np.float64)
    rotation = payload.get("rotation") or {}
    quaternion_wxyz = np.asarray(
        [rotation.get("w"), rotation.get("x"), rotation.get("y"), rotation.get("z")], dtype=np.float64
    )
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise RuntimeError(f"tracked body {body_name!r} has invalid world position")
    if quaternion_wxyz.shape != (4,) or not np.all(np.isfinite(quaternion_wxyz)):
        raise RuntimeError(f"tracked body {body_name!r} has invalid world quaternion")
    return {"position_w_m": position.tolist(), "quaternion_wxyz": normalized_quaternion(quaternion_wxyz).tolist()}


def send_body_force(session: requests.Session, simulator_endpoint: str, body_name: str, force_world: np.ndarray, steps: int) -> None:
    if force_world.shape != (3,) or not np.all(np.isfinite(force_world)):
        raise ValueError("body launch force must be finite [3]")
    response = session.post(
        f"{simulator_endpoint}/body_apply_force",
        json={"body_name": body_name, "force_world": force_world.tolist(), "torque_world": [0.0, 0.0, 0.0], "steps": steps},
        timeout=3,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("status"):
        raise RuntimeError(f"body force request failed: {payload.get('message')}")


def summarize_ball_tracking(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    paired = [sample for sample in samples if "ball_pose" in sample and "tracked_body_pose" in sample]
    if not paired:
        return None
    distances = [
        float(np.linalg.norm(np.asarray(sample["ball_pose"]["position_w_m"]) - np.asarray(sample["tracked_body_pose"]["position_w_m"])))
        for sample in paired
    ]
    closest_index = int(np.argmin(distances))
    closest = paired[closest_index]
    return {
        "sample_count": len(paired),
        "minimum_ball_to_racket_center_distance_m": distances[closest_index],
        "closest_approach_elapsed_s": closest["elapsed_s"],
        "closest_ball_center_w_m": closest["ball_pose"]["position_w_m"],
        "closest_racket_center_w_m": closest["tracked_body_pose"]["position_w_m"],
        "note": "closest approach is a coordinate/contact diagnostic; it is not a physical-hit verdict by itself",
    }


def send_arm_command(session: requests.Session, endpoint: str, command: np.ndarray, sequence: int) -> None:
    if command.shape != (14,) or not np.all(np.isfinite(command)):
        raise ValueError("arm command must be finite [14]")
    response = session.post(
        f"{endpoint}/channel/%2Fmotion%2Fcontrol%2Farm_joint_command/ros2%3Asensor_msgs%2Fmsg%2FJointState",
        headers={"Content-Type": "application/json"},
        json={
            "header": request_header(sequence), "name": list(ARM_NAMES),
            "position": [float(value) for value in command],
            "velocity": [0.0] * 14, "effort": [0.0] * 14,
        },
        timeout=3,
    )
    response.raise_for_status()


def load_right_arm_reference(command_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(command_path, allow_pickle=False) as data:
        required = {"timestamps_s", "q_des", "joint_names"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{command_path}: missing {', '.join(missing)}")
        timestamps = np.asarray(data["timestamps_s"], dtype=np.float64)
        positions = np.asarray(data["q_des"], dtype=np.float64)
        names = [str(name) for name in np.asarray(data["joint_names"]).tolist()]
    if timestamps.ndim != 1 or len(timestamps) < 2 or not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("timestamps_s must be strictly increasing with at least two samples")
    if positions.shape != (len(timestamps), len(names)) or not np.all(np.isfinite(positions)):
        raise ValueError("q_des must be finite [T, len(joint_names)]")
    indices = []
    for name in RIGHT_ARM_NAMES:
        if names.count(name) != 1:
            raise ValueError(f"canonical command must contain exactly one {name}")
        indices.append(names.index(name))
    return timestamps, positions[:, indices]


def interpolate_reference(timestamps: np.ndarray, values: np.ndarray, at_s: float) -> np.ndarray:
    return np.asarray([np.interp(at_s, timestamps, values[:, column]) for column in range(values.shape[1])], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True, type=Path, help="Canonical 31-DOF NPZ; only right-arm q_des columns are used.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--endpoint", default=DEFAULT_MOTION_ENDPOINT)
    parser.add_argument("--simulator-endpoint", default=DEFAULT_SIM_ENDPOINT)
    parser.add_argument("--pre-stand-s", type=float, default=2.0)
    parser.add_argument("--blend-s", type=float, default=0.5)
    parser.add_argument("--hold-s", type=float, default=0.5)
    parser.add_argument("--command-hz", type=float, default=100.0)
    parser.add_argument("--monitor-hz", type=float, default=50.0)
    parser.add_argument("--tracked-body", default="", help="Optional MuJoCo body whose world pose is recorded with monitor samples.")
    parser.add_argument("--ball-body", default="", help="Optional free ball body whose world pose is recorded.")
    parser.add_argument("--ball-launch-force-world", nargs=3, type=float, metavar=("FX", "FY", "FZ"),
                        help="Optional one-shot world-frame force used to launch --ball-body.")
    parser.add_argument("--ball-launch-steps", type=int, default=1)
    parser.add_argument("--ball-launch-at-s", type=float, default=0.0,
                        help="Seconds after arm replay starts at which to send the ball launch force.")
    parser.add_argument("--max-pre-tilt-deg", type=float, default=3.0,
                        help="Maximum pelvis orientation drift during the stationary preflight window.")
    parser.add_argument("--max-stroke-tilt-deg", type=float, default=25.0,
                        help="Maximum pelvis orientation change from preflight baseline during blend/replay/hold.")
    parser.add_argument("--max-pelvis-angular-speed-rad-s", type=float, default=0.5)
    parser.add_argument("--max-torso-angular-speed-rad-s", type=float, default=0.5)
    args = parser.parse_args()
    if (args.pre_stand_s <= 0.0 or args.blend_s < 0.0 or args.hold_s < 0.0
            or args.command_hz <= 0.0 or args.monitor_hz <= 0.0
            or args.max_pre_tilt_deg < 0.0 or args.max_stroke_tilt_deg < 0.0
            or args.max_pelvis_angular_speed_rad_s <= 0.0 or args.max_torso_angular_speed_rad_s <= 0.0
            or args.ball_launch_steps <= 0 or args.ball_launch_at_s < 0.0):
        parser.error("all durations/rates must be positive and limits must be non-negative")
    if args.ball_launch_force_world is not None and not args.ball_body:
        parser.error("--ball-launch-force-world requires --ball-body")
    return args


def stability_violation(state: dict[str, Any], baseline_orientation: np.ndarray, args: argparse.Namespace, *, preflight: bool) -> str | None:
    tilt = relative_tilt_deg(baseline_orientation, state["pelvis_orientation"])
    tilt_limit = args.max_pre_tilt_deg if preflight else args.max_stroke_tilt_deg
    if tilt > tilt_limit:
        return f"pelvis relative tilt {tilt:.2f} deg exceeds {tilt_limit:.2f} deg"
    if state["pelvis_angular_speed_rad_s"] > args.max_pelvis_angular_speed_rad_s:
        return f"pelvis angular speed {state['pelvis_angular_speed_rad_s']:.3f} rad/s exceeds limit"
    if state["torso_angular_speed_rad_s"] > args.max_torso_angular_speed_rad_s:
        return f"torso angular speed {state['torso_angular_speed_rad_s']:.3f} rad/s exceeds limit"
    return None


def main() -> int:
    args = parse_args()
    if os.environ.get("SIM_MODE", "").lower() != "sil":
        raise SystemExit("Refusing non-SIL operation: set SIM_MODE=sil.")
    command_path = args.command.expanduser().resolve()
    timestamps, right_reference = load_right_arm_reference(command_path)
    session = requests.Session()
    output: dict[str, Any] = {
        "schema_version": 1,
        "scope": "local_official_aimsim_sil_native_motion_right_arm_only",
        "command": str(command_path),
        "endpoint": args.endpoint,
        "simulator_endpoint": args.simulator_endpoint,
        "right_arm_joint_names": list(RIGHT_ARM_NAMES),
        "native_motion_owned_joints": ["legs", "waist", "torso", "head", "left_arm"],
        "limits": {
            "pre_stand_s": args.pre_stand_s, "max_pre_tilt_deg": args.max_pre_tilt_deg,
            "max_stroke_tilt_deg": args.max_stroke_tilt_deg,
            "max_pelvis_angular_speed_rad_s": args.max_pelvis_angular_speed_rad_s,
            "max_torso_angular_speed_rad_s": args.max_torso_angular_speed_rad_s,
        },
        "preflight_samples": [], "stroke_samples": [], "pass": False,
    }
    if args.tracked_body:
        output["tracked_body"] = args.tracked_body
    if args.ball_body:
        output["ball_body"] = args.ball_body
    if args.ball_launch_force_world is not None:
        output["ball_launch"] = {
            "force_world_n": args.ball_launch_force_world,
            "steps": args.ball_launch_steps,
            "requested_at_s": args.ball_launch_at_s,
            "sent": False,
        }
    try:
        action = read_action(session, args.endpoint)
        output["action_at_start"] = action
        if action != MOTION_ACTION:
            raise RuntimeError(f"native MOTION must be active before arm control; observed {action!r}")
        initial = read_stability_state(session, args.simulator_endpoint)
        baseline = initial["pelvis_orientation"]
        preflight_deadline = time.monotonic() + args.pre_stand_s
        while time.monotonic() < preflight_deadline:
            state = read_stability_state(session, args.simulator_endpoint)
            state["relative_pelvis_tilt_deg"] = relative_tilt_deg(baseline, state["pelvis_orientation"])
            state["time_s"] = time.monotonic()
            if args.tracked_body:
                state["tracked_body_pose"] = read_body_pose(session, args.simulator_endpoint, args.tracked_body)
            if args.ball_body:
                state["ball_pose"] = read_body_pose(session, args.simulator_endpoint, args.ball_body)
            state["pelvis_orientation"] = state["pelvis_orientation"].tolist()
            output["preflight_samples"].append(state)
            violation = stability_violation({**state, "pelvis_orientation": np.asarray(state["pelvis_orientation"])}, baseline, args, preflight=True)
            if violation:
                raise RuntimeError(f"preflight standing gate failed: {violation}")
            time.sleep(1.0 / args.monitor_hz)
        joints = read_joint_positions(session, args.simulator_endpoint)
        left_hold = np.asarray([joints[name] for name in ARM_NAMES[:7]], dtype=np.float64)
        current_right = np.asarray([joints[name] for name in RIGHT_ARM_NAMES], dtype=np.float64)
        command_period = 1.0 / args.command_hz
        monitor_period = 1.0 / args.monitor_hz
        started = time.monotonic()
        next_monitor = started
        sequence = 0
        ball_launched = False
        total_s = args.blend_s + float(timestamps[-1]) + args.hold_s
        while True:
            now = time.monotonic()
            elapsed = now - started
            if elapsed > total_s:
                break
            if elapsed < args.blend_s and args.blend_s > 0.0:
                alpha = elapsed / args.blend_s
                right = (1.0 - alpha) * current_right + alpha * right_reference[0]
                phase = "blend"
            elif elapsed < args.blend_s + float(timestamps[-1]):
                right = interpolate_reference(timestamps, right_reference, elapsed - args.blend_s)
                phase = "strike"
            else:
                right = right_reference[-1]
                phase = "hold"
            if (args.ball_launch_force_world is not None and not ball_launched
                    and elapsed >= args.ball_launch_at_s):
                send_body_force(
                    session, args.simulator_endpoint, args.ball_body,
                    np.asarray(args.ball_launch_force_world, dtype=np.float64), args.ball_launch_steps,
                )
                ball_launched = True
                output["ball_launch"]["sent"] = True
                output["ball_launch"]["actual_elapsed_s"] = elapsed
            if now >= next_monitor:
                state = read_stability_state(session, args.simulator_endpoint)
                state["relative_pelvis_tilt_deg"] = relative_tilt_deg(baseline, state["pelvis_orientation"])
                state["elapsed_s"] = elapsed
                state["phase"] = phase
                if args.tracked_body:
                    state["tracked_body_pose"] = read_body_pose(session, args.simulator_endpoint, args.tracked_body)
                if args.ball_body:
                    state["ball_pose"] = read_body_pose(session, args.simulator_endpoint, args.ball_body)
                state["pelvis_orientation"] = state["pelvis_orientation"].tolist()
                output["stroke_samples"].append(state)
                violation = stability_violation({**state, "pelvis_orientation": np.asarray(state["pelvis_orientation"])}, baseline, args, preflight=False)
                if violation:
                    raise RuntimeError(f"stroke stability gate failed; arm publication stopped: {violation}")
                next_monitor += monitor_period
            send_arm_command(session, args.endpoint, np.concatenate([left_hold, right]), sequence)
            sequence += 1
            time.sleep(max(0.0, started + sequence * command_period - time.monotonic()))
        ball_summary = summarize_ball_tracking(output["stroke_samples"])
        if ball_summary is not None:
            output["ball_tracking"] = ball_summary
        output["pass"] = True
        output["status"] = "native_motion_stable_right_arm_strike"
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        output["status"] = "failed"
        output["failure"] = str(exc)
    finally:
        output["action_at_end"] = None
        try:
            output["action_at_end"] = read_action(session, args.endpoint)
        except requests.RequestException:
            pass
        output["output_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pass": output["pass"], "status": output["status"]}, ensure_ascii=False))
    return 0 if output["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
