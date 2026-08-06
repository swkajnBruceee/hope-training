#!/usr/bin/env python3
"""Guarded SIL test: native PD stand -> stability gate -> ONNX handoff.

This is intentionally a *static-standing* gate, not a motion-performance test.
It refuses to start unless SIM_MODE=sil, requires a sustained native PD_STAND
window, keeps native PD_STAND during the deploy program's own 3 s PD warmup,
and only then transfers body-drive ownership to ONNX.  A safety violation stops
the policy immediately and resets SIL back to native MOTION.
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
PACKAGE = ROOT / "a3_deploy_example" / "dist" / "a3_deploy_x86_64"
RUNTIME_CFG = ROOT / "experiments" / "a3_onnx_balance_sil" / "stand_shadow.yaml"
ARTIFACTS = ROOT / "experiments" / "a3_onnx_balance_sil" / "artifacts"
REPORT = ARTIFACTS / "gated_start_switch_report.json"
ONNX_LOG = ARTIFACTS / "gated_start_switch_onnx.log"
MOTION_ENDPOINT = "http://127.0.0.1:56322"
SIM_ENDPOINT = "http://127.0.0.1:8001"

# All limits are deliberately tighter than the earlier emergency-stop bounds.
STAND_TIMEOUT_S = 10.0
STAND_STABLE_HOLD_S = 3.0
ONNX_PD_WARMUP_S = 3.3  # 150 ticks at 50 Hz, plus scheduling margin.
ONNX_POLICY_WINDOW_S = 1.0
MIN_STAND_HEIGHT_M = 1.00
MAX_STAND_TILT_DEG = 4.0
MAX_STAND_ANGULAR_SPEED_RAD_S = 0.35
MAX_HANDOFF_TILT_DEG = 6.0
MAX_HANDOFF_ANGULAR_SPEED_RAD_S = 0.75
MAX_HANDOFF_HEIGHT_DROP_M = 0.05


def header() -> dict[str, Any]:
    # Match the vendor HTTP bridge's naive-UTC timestamp convention.
    now = datetime.utcnow()
    timestamp = now.timestamp()
    return {
        "timestamp": {
            "seconds": int(timestamp),
            "nanos": now.microsecond * 1000,
            "ms_since_epoch": int(timestamp * 1000),
        },
        "control_source": "ControlSource_SAFE",
        "uuid": "",
        "trace_id": "a3_onnx_gated_sil",
        "domin": "",
    }


def set_action(session: requests.Session, action: str) -> None:
    response = session.post(
        f"{MOTION_ENDPOINT}/rpc/aimdk.protocol.MotionControlActionService/SetAction",
        json={"header": header(), "command": {"action": action, "ext_action": ""}},
        timeout=2,
    )
    response.raise_for_status()


def get_action(session: requests.Session) -> str:
    response = session.post(
        f"{MOTION_ENDPOINT}/rpc/aimdk.protocol.MotionControlActionService/GetAction",
        json={"header": header()}, timeout=2,
    )
    response.raise_for_status()
    return str(response.json().get("info", {}).get("current_action", ""))


def sample_state(session: requests.Session) -> tuple[np.ndarray, float, float, float]:
    imu_response = session.get(f"{SIM_ENDPOINT}/imu", timeout=2)
    imu_response.raise_for_status()
    imu = imu_response.json()["imu"]
    pelvis = imu["pelvis"]
    torso = imu["torso"]
    quat = np.asarray(pelvis["orientation"], dtype=float)
    quat /= np.linalg.norm(quat)
    pose_response = session.get(
        f"{SIM_ENDPOINT}/baselink_position",
        params={"body_name": "pelvis_link"}, timeout=2,
    )
    pose_response.raise_for_status()
    pose = pose_response.json()
    if not pose.get("status"):
        raise RuntimeError(f"pelvis_link unavailable: {pose.get('message')}")
    return (
        quat,
        float(np.linalg.norm(pelvis["angular_velocity"])),
        float(np.linalg.norm(torso["angular_velocity"])),
        float(np.asarray(pose["position"], dtype=float)[2]),
    )


def tilt_deg(reference: np.ndarray, current: np.ndarray) -> float:
    dot = abs(float(np.dot(reference, current)))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))


def restore_native_motion(session: requests.Session) -> list[str]:
    """Use a reset so recovery is deterministic even after a failed handoff."""
    response = session.post(f"{SIM_ENDPOINT}/reset_simulation", json={}, timeout=3)
    response.raise_for_status()
    time.sleep(0.5)
    set_action(session, "MotionControlAction_PD_STAND")
    time.sleep(3.2)
    set_action(session, "MotionControlAction_MOTION")
    time.sleep(0.3)
    if get_action(session) != "MotionControlAction_MOTION":
        raise RuntimeError("recovery did not return to native MOTION")
    return ["reset_simulation", "MotionControlAction_PD_STAND", "MotionControlAction_MOTION"]


def wait_for_native_stand(session: requests.Session) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    """Require an uninterrupted upright/low-rate PD_STAND interval."""
    deadline = time.monotonic() + STAND_TIMEOUT_S
    stable_since: float | None = None
    reference: np.ndarray | None = None
    samples: list[dict[str, float]] = []
    while time.monotonic() < deadline:
        quat, pelvis_rate, torso_rate, height = sample_state(session)
        if reference is None:
            reference = quat
        tilt = tilt_deg(reference, quat)
        ok = (
            height >= MIN_STAND_HEIGHT_M
            and tilt <= MAX_STAND_TILT_DEG
            and pelvis_rate <= MAX_STAND_ANGULAR_SPEED_RAD_S
            and torso_rate <= MAX_STAND_ANGULAR_SPEED_RAD_S
        )
        samples.append({
            "tilt_deg": tilt,
            "pelvis_angular_speed_rad_s": pelvis_rate,
            "torso_angular_speed_rad_s": torso_rate,
            "pelvis_height_m": height,
            "stable": float(ok),
        })
        if ok:
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since >= STAND_STABLE_HOLD_S:
                return reference, height, samples
        else:
            stable_since = None
            reference = quat
        time.sleep(0.02)
    raise RuntimeError("native PD_STAND did not satisfy the 3 s stability gate")


def check_handoff_sample(reference: np.ndarray, stable_height: float, sample: tuple[np.ndarray, float, float, float]) -> dict[str, float]:
    quat, pelvis_rate, torso_rate, height = sample
    data = {
        "tilt_deg": tilt_deg(reference, quat),
        "pelvis_angular_speed_rad_s": pelvis_rate,
        "torso_angular_speed_rad_s": torso_rate,
        "pelvis_height_m": height,
        "height_drop_m": stable_height - height,
    }
    if (
        data["tilt_deg"] > MAX_HANDOFF_TILT_DEG
        or pelvis_rate > MAX_HANDOFF_ANGULAR_SPEED_RAD_S
        or torso_rate > MAX_HANDOFF_ANGULAR_SPEED_RAD_S
        or data["height_drop_m"] > MAX_HANDOFF_HEIGHT_DROP_M
    ):
        raise RuntimeError(f"handoff safety stop: {data}")
    return data


def main() -> int:
    if os.environ.get("SIM_MODE", "").lower() != "sil":
        raise SystemExit("Refusing this command test outside SIM_MODE=sil")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    process: subprocess.Popen[str] | None = None
    log_file = None
    report: dict[str, Any] = {
        "scope": "local_sil_native_pdstand_stability_gate_then_onnx_static_handoff",
        "pass": False,
        "limits": {
            "native_stand_hold_s": STAND_STABLE_HOLD_S,
            "native_min_height_m": MIN_STAND_HEIGHT_M,
            "native_max_tilt_deg": MAX_STAND_TILT_DEG,
            "native_max_angular_speed_rad_s": MAX_STAND_ANGULAR_SPEED_RAD_S,
            "handoff_max_tilt_deg": MAX_HANDOFF_TILT_DEG,
            "handoff_max_angular_speed_rad_s": MAX_HANDOFF_ANGULAR_SPEED_RAD_S,
            "handoff_max_height_drop_m": MAX_HANDOFF_HEIGHT_DROP_M,
        },
        "onnx_policy_window_s": ONNX_POLICY_WINDOW_S,
    }
    handoff_samples: list[dict[str, float]] = []
    try:
        # A reset makes the starting pose/action reproducible for this branch test.
        response = session.post(f"{SIM_ENDPOINT}/reset_simulation", json={}, timeout=3)
        response.raise_for_status()
        time.sleep(0.5)
        set_action(session, "MotionControlAction_PD_STAND")
        report["native_action_before_gate"] = get_action(session)
        reference, stable_height, stand_samples = wait_for_native_stand(session)
        report["native_stand_gate_passed"] = True
        report["native_stand_reference_height_m"] = stable_height
        report["native_stand_samples"] = stand_samples

        environment = os.environ | {
            "A3_SOURCE_ROBOT_ENV": "0",
            "A3_TRANSPORT": "iceoryx",
            "LD_LIBRARY_PATH": f"/workspace/anaconda3/envs/hope_ros/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
        }
        log_file = ONNX_LOG.open("w", encoding="utf-8")
        process = subprocess.Popen(
            [
                str(PACKAGE / "a3_deploy_onnx_ref"),
                f"--runtime-cfg={RUNTIME_CFG}",
                f"--aimrt-cfg={PACKAGE / 'config/a3_aimrt_config.iceoryx.yaml'}",
                "--auto-start",
                "--frame-log-interval=100",
            ],
            cwd=PACKAGE,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # During this window native PD_STAND remains owner, while the ONNX
        # executable performs its documented 150-tick PD warmup.
        warmup_deadline = time.monotonic() + ONNX_PD_WARMUP_S
        while time.monotonic() < warmup_deadline:
            if process.poll() is not None:
                raise RuntimeError(f"ONNX exited during PD warmup ({process.returncode})")
            check_handoff_sample(reference, stable_height, sample_state(session))
            time.sleep(0.02)

        # The only ownership transition happens after both stand gates passed.
        set_action(session, "MotionControlAction_PASSIVE")
        report["native_action_during_onnx_window"] = get_action(session)
        deadline = time.monotonic() + ONNX_POLICY_WINDOW_S
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"ONNX exited during handoff ({process.returncode})")
            handoff_samples.append(check_handoff_sample(reference, stable_height, sample_state(session)))
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
        if log_file is not None:
            log_file.close()
        try:
            report["recovery_actions"] = restore_native_motion(session)
            report["native_action_after"] = get_action(session)
        except (requests.RequestException, RuntimeError) as error:
            report["recovery_failure"] = str(error)
        report["handoff_samples"] = handoff_samples
        if handoff_samples:
            report["handoff_max_tilt_deg"] = max(s["tilt_deg"] for s in handoff_samples)
            report["handoff_max_pelvis_angular_speed_rad_s"] = max(s["pelvis_angular_speed_rad_s"] for s in handoff_samples)
            report["handoff_max_torso_angular_speed_rad_s"] = max(s["torso_angular_speed_rad_s"] for s in handoff_samples)
            report["handoff_min_pelvis_height_m"] = min(s["pelvis_height_m"] for s in handoff_samples)
        report["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "report": str(REPORT), "failure": report.get("failure")}, ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
