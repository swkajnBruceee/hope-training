#!/usr/bin/env python3
"""Short, guarded local-SIL command test for the ONNX standing policy.

This test is deliberately bounded: ONNX begins publishing during its own
three-second PD transition; only then is native MotionControl changed to
PASSIVE, giving ONNX exclusive body-drive ownership for three seconds.  Any
IMU violation terminates the ONNX process and restores native PD_STAND/MOTION.
It rejects every non-loopback or non-SIL invocation.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "a3_deploy_example/dist/a3_deploy_x86_64"
RUNTIME_CFG = ROOT / "experiments/a3_onnx_balance_sil/stand_shadow.yaml"
REPORT = ROOT / "experiments/a3_onnx_balance_sil/artifacts/command_stand_report.json"
MOTION_ENDPOINT = "http://127.0.0.1:56322"
SIM_ENDPOINT = "http://127.0.0.1:8001"
HANDOFF_AFTER_S = 3.3
COMMAND_WINDOW_S = 3.0
MAX_TILT_DEG = 15.0
MAX_ANGULAR_SPEED_RAD_S = 1.5
MIN_PELVIS_HEIGHT_M = 0.75


def header() -> dict[str, Any]:
    # Match the vendor HTTP tool's naive-UTC timestamp convention.  The
    # MotionControl HTTP bridge rejects a timezone-aware timestamp here.
    now = datetime.utcnow()
    timestamp = now.timestamp()
    return {
        "timestamp": {"seconds": int(timestamp), "nanos": now.microsecond * 1000, "ms_since_epoch": int(timestamp * 1000)},
        "control_source": "ControlSource_SAFE", "uuid": "", "trace_id": "a3_onnx_sil_branch", "domin": "",
    }


def get_action(session: requests.Session) -> str:
    response = session.post(f"{MOTION_ENDPOINT}/rpc/aimdk.protocol.MotionControlActionService/GetAction", json={"header": header()}, timeout=2)
    response.raise_for_status()
    return str(response.json().get("info", {}).get("current_action", ""))


def set_action(session: requests.Session, action: str) -> None:
    response = session.post(
        f"{MOTION_ENDPOINT}/rpc/aimdk.protocol.MotionControlActionService/SetAction",
        # command is a MotionControlActionCommand message, not a string.  The
        # vendor helper script currently serializes it incorrectly.
        json={"header": header(), "command": {"action": action, "ext_action": ""}}, timeout=2,
    )
    response.raise_for_status()


def state(session: requests.Session) -> tuple[np.ndarray, float, float, float]:
    response = session.get(f"{SIM_ENDPOINT}/imu", timeout=2)
    response.raise_for_status()
    imu = response.json()["imu"]
    pelvis = imu["pelvis"]
    torso = imu["torso"]
    orientation = np.asarray(pelvis["orientation"], dtype=float)
    orientation /= np.linalg.norm(orientation)
    pose_response = session.get(
        f"{SIM_ENDPOINT}/baselink_position", params={"body_name": "pelvis_link"}, timeout=2,
    )
    pose_response.raise_for_status()
    pose = pose_response.json()
    if not pose.get("status"):
        raise RuntimeError(f"simulator did not resolve pelvis_link: {pose.get('message')}")
    height = float(np.asarray(pose["position"], dtype=float)[2])
    return orientation, float(np.linalg.norm(pelvis["angular_velocity"])), float(np.linalg.norm(torso["angular_velocity"])), height


def tilt_deg(reference: np.ndarray, current: np.ndarray) -> float:
    return float(np.degrees(2.0 * np.arccos(np.clip(abs(float(np.dot(reference, current))), -1.0, 1.0))))


def restore(session: requests.Session, original: str) -> list[str]:
    actions: list[str] = []
    # PASSIVE must transition through PD_STAND before returning to MOTION.
    if original == "MotionControlAction_MOTION":
        set_action(session, "MotionControlAction_PD_STAND")
        actions.append("MotionControlAction_PD_STAND")
        time.sleep(3.0)
    if original and original != "MotionControlAction_PD_STAND":
        if original not in {
            "MotionControlAction_MOTION",
            "MotionControlAction_PASSIVE",
            "MotionControlAction_DAMPING",
        }:
            raise RuntimeError(f"unsupported original action for restore: {original}")
        set_action(session, original)
        actions.append(original)
    return actions


def reset_and_restore_motion(session: requests.Session) -> list[str]:
    """Return local SIL to a known upright native-MOTION state after a fall."""

    response = session.post(f"{SIM_ENDPOINT}/reset_simulation", json={}, timeout=3)
    response.raise_for_status()
    time.sleep(0.5)
    set_action(session, "MotionControlAction_PD_STAND")
    time.sleep(3.2)
    set_action(session, "MotionControlAction_MOTION")
    time.sleep(0.3)
    if get_action(session) != "MotionControlAction_MOTION":
        raise RuntimeError("SIL reset did not restore native MOTION")
    return ["reset_simulation", "MotionControlAction_PD_STAND", "MotionControlAction_MOTION"]


def main() -> int:
    if os.environ.get("SIM_MODE", "").lower() != "sil":
        raise SystemExit("Refusing command test outside SIM_MODE=sil")
    session = requests.Session()
    process: subprocess.Popen[str] | None = None
    report: dict[str, Any] = {
        "scope": "local_sil_onnx_full_body_static_stand",
        "command_window_s": COMMAND_WINDOW_S,
        "limits": {"tilt_deg": MAX_TILT_DEG, "pelvis_angular_speed_rad_s": MAX_ANGULAR_SPEED_RAD_S, "torso_angular_speed_rad_s": MAX_ANGULAR_SPEED_RAD_S, "min_pelvis_height_m": MIN_PELVIS_HEIGHT_M},
        "command_publishers": "onnx_only_during_window",
        "pass": False,
    }
    samples: list[dict[str, float]] = []
    try:
        original = get_action(session)
        if original != "MotionControlAction_MOTION":
            raise RuntimeError(f"requires native MOTION preflight; observed {original!r}")
        baseline, _, _, baseline_height = state(session)
        if baseline_height < MIN_PELVIS_HEIGHT_M:
            raise RuntimeError(f"preflight rejected: pelvis height {baseline_height:.3f} m")
        report["preflight_pelvis_height_m"] = baseline_height
        report["native_action_before"] = original
        environment = os.environ | {
            "A3_SOURCE_ROBOT_ENV": "0", "A3_TRANSPORT": "iceoryx",
            "LD_LIBRARY_PATH": f"/workspace/anaconda3/envs/hope_ros/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
        }
        process = subprocess.Popen(
            [str(PACKAGE / "a3_deploy_onnx_ref"), f"--runtime-cfg={RUNTIME_CFG}",
             f"--aimrt-cfg={PACKAGE / 'config/a3_aimrt_config.iceoryx.yaml'}", "--auto-start", "--frame-log-interval=100"],
            cwd=PACKAGE, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True,
        )
        time.sleep(HANDOFF_AFTER_S)
        if process.poll() is not None:
            raise RuntimeError(f"ONNX process exited before handoff ({process.returncode})")
        # The official action graph requires MOTION -> PD_STAND -> PASSIVE.
        # ONNX is already publishing through this short transition, so it is
        # never left without a body-drive command source.
        set_action(session, "MotionControlAction_PD_STAND")
        time.sleep(0.12)
        set_action(session, "MotionControlAction_PASSIVE")
        report["native_action_during_window"] = "MotionControlAction_PASSIVE"
        deadline = time.monotonic() + COMMAND_WINDOW_S
        while time.monotonic() < deadline:
            orientation, pelvis_speed, torso_speed, pelvis_height = state(session)
            tilt = tilt_deg(baseline, orientation)
            sample = {"tilt_deg": tilt, "pelvis_angular_speed_rad_s": pelvis_speed, "torso_angular_speed_rad_s": torso_speed, "pelvis_height_m": pelvis_height}
            samples.append(sample)
            if (tilt > MAX_TILT_DEG or pelvis_speed > MAX_ANGULAR_SPEED_RAD_S
                    or torso_speed > MAX_ANGULAR_SPEED_RAD_S or pelvis_height < MIN_PELVIS_HEIGHT_M):
                raise RuntimeError(f"IMU safety stop: {sample}")
            time.sleep(0.02)
        report["pass"] = True
    except (requests.RequestException, RuntimeError, ValueError) as error:
        report["failure"] = str(error)
    finally:
        if process and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        try:
            report["restore_actions"] = restore(session, report.get("native_action_before", "MotionControlAction_MOTION"))
            time.sleep(0.3)
            report["native_action_after"] = get_action(session)
            if report["native_action_after"] != "MotionControlAction_MOTION":
                report["reset_recovery_actions"] = reset_and_restore_motion(session)
                report["native_action_after"] = get_action(session)
        except (requests.RequestException, RuntimeError) as error:
            report["restore_failure"] = str(error)
        if samples:
            report["samples"] = samples
            report["max_tilt_deg"] = max(item["tilt_deg"] for item in samples)
            report["max_pelvis_angular_speed_rad_s"] = max(item["pelvis_angular_speed_rad_s"] for item in samples)
            report["max_torso_angular_speed_rad_s"] = max(item["torso_angular_speed_rad_s"] for item in samples)
            report["min_pelvis_height_m"] = min(item["pelvis_height_m"] for item in samples)
        report["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "report": str(REPORT), "failure": report.get("failure")}, ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
