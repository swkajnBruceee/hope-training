#!/usr/bin/env python3
"""Check the local official A3 SIL and TA command-path readiness.

This intentionally distinguishes three facts that are often conflated:
  1. the official SIL processes/endpoints are alive;
  2. the TA fusion source/configuration exists;
  3. the active MOTION runtime actually observes a waist command.

It does not publish commands and it does not claim real-robot readiness.
"""

from __future__ import annotations

import argparse
import json
import socket
import urllib.error
import urllib.request
from pathlib import Path


def request_json(url: str, method: str = "GET") -> dict:
    request = urllib.request.Request(
        url,
        method=method,
        headers={"Content-Type": "application/json"},
        data=b"{}" if method == "POST" else None,
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        body = response.read().decode("utf-8")
        return {"ok": True, "status": response.status, "body": json.loads(body)}


def probe(url: str, method: str = "GET") -> dict:
    try:
        return request_json(url, method)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return {"ok": False, "error": str(exc)}


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def contains(root: Path, pattern: str) -> list[str]:
    matches = []
    if not root.exists():
        return matches
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".yaml", ".yml", ".cpp", ".h", ".proto"}:
            try:
                if pattern in path.read_text(errors="replace"):
                    matches.append(str(path))
            except OSError:
                continue
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--motion-port", type=int, default=56322)
    parser.add_argument("--sim-port", type=int, default=8001)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    official_root = args.root / "third_party" / "aimsim_official"
    deployment_root = args.root / "agibot" / "code_deployment"
    motion_root = official_root / "motion_control_humble"
    deployment_ta_matches = contains(
        deployment_root, "/ta/whole_body_command"
    )
    motion_ta_matches = contains(
        motion_root, "/ta/whole_body_command"
    )
    report = {
        "scope": "local_official_a3_aimsim_sil_readiness",
        "publishes_commands": False,
        "ports": {
            "motion_control": {"port": args.motion_port, "open": port_open(args.motion_port)},
            "mujoco_simulator": {"port": args.sim_port, "open": port_open(args.sim_port)},
        },
        "endpoints": {
            "sim_liveness": probe(f"http://127.0.0.1:{args.sim_port}/liveness"),
            "joint_states": probe(f"http://127.0.0.1:{args.sim_port}/joint_states"),
            "imu": probe(f"http://127.0.0.1:{args.sim_port}/imu"),
            "current_action": probe(
                "http://127.0.0.1:%d/rpc/aimdk.protocol.MotionControlActionService/GetAction"
                % args.motion_port,
                "POST",
            ),
        },
        "source_and_config": {
            "ta_whole_body_source_matches": deployment_ta_matches,
            "ta_topic_in_deployment_config": [
                path for path in deployment_ta_matches
                if path.endswith((".yaml", ".yml"))
            ],
            "ta_topic_in_motion_control_config": motion_ta_matches,
            "official_motion_binary": str(motion_root / "bin" / "motion_control"),
            "official_motion_binary_exists": (motion_root / "bin" / "motion_control").exists(),
        },
    }

    action_body = report["endpoints"]["current_action"].get("body", {})
    report["current_action"] = action_body.get("info", {}).get("current_action", "")
    report["official_sil_running"] = bool(
        report["ports"]["motion_control"]["open"]
        and report["ports"]["mujoco_simulator"]["open"]
        and report["endpoints"]["sim_liveness"].get("ok")
    )
    report["ta_source_confirmed"] = bool(
        report["source_and_config"]["ta_whole_body_source_matches"]
    )
    report["ta_runtime_topic_active"] = bool(
        report["source_and_config"]["ta_topic_in_deployment_config"]
    )
    report["waist_external_command_status"] = (
        "not_tested_by_this_readiness_check"
    )
    report["interpretation"] = (
        "Official SIL is online and the deployment TA topic is configured. "
        "The standalone MOTION configuration does not advertise that topic; "
        "TA ownership must therefore be tested through the deployment backend, "
        "not the direct MOTION waist channel."
        if report["official_sil_running"] and report["ta_source_confirmed"]
        else "Official SIL or TA source is not ready according to the checks above."
    )

    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
