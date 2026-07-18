#!/usr/bin/env python3
"""Activate the official A3 AimSim SIL motion-control state.

The official SIL starts in PASSIVE. A valid MOTION validation run must first
request GET_UP, wait for GET_UP_FINISHED, and only then request MOTION.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_ENDPOINT = "http://127.0.0.1:56322/rpc/aimdk.protocol.MotionControlActionService"
USE_EXT_CMD = "MotionControlAction_USE_EXT_CMD"


def make_header() -> dict:
    now = datetime.now(timezone.utc)
    timestamp = now.timestamp()
    seconds = int(timestamp)
    nanos = int((timestamp - seconds) * 1_000_000_000)
    return {
        "timestamp": {
            "seconds": seconds,
            "nanos": nanos,
            "ms_since_epoch": int(timestamp * 1000),
        },
        "control_source": "ControlSource_SAFE",
        "uuid": "",
        "trace_id": "hope_official_aimsim_sil_activation",
        "domin": "",
    }


def call(endpoint: str, method: str, payload: dict) -> dict:
    request = Request(
        f"{endpoint}/{method}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"{method} failed: {exc}") from exc


def get_action(endpoint: str) -> dict:
    return call(endpoint, "GetAction", {"header": make_header()})


def set_external_action(endpoint: str, action: str) -> dict:
    return call(
        endpoint,
        "SetAction",
        {
            "header": make_header(),
            "command": {"action": USE_EXT_CMD, "ext_action": action},
        },
    )


def print_state(prefix: str, response: dict) -> dict:
    info = response.get("info", {})
    state = {
        "current_action": info.get("current_action"),
        "ext_action": info.get("ext_action"),
        "status": info.get("status"),
        "stage": info.get("stage", {}).get("description"),
    }
    print(f"[{prefix}] {json.dumps(state, ensure_ascii=False)}", flush=True)
    return state


def wait_for_action(endpoint: str, action: str, timeout_s: float, poll_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while True:
        state = print_state(action.lower(), get_action(endpoint))
        if state["current_action"] == f"MotionControlAction_{action}":
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{action} did not become active before timeout")
        time.sleep(poll_s)


def wait_for_service_action(endpoint: str, timeout_s: float, poll_s: float) -> dict:
    """Wait until the local controller has a simulator-backed action state."""

    deadline = time.monotonic() + timeout_s
    last_error: RuntimeError | None = None
    while time.monotonic() < deadline:
        try:
            return get_action(endpoint)
        except RuntimeError as error:
            last_error = error
            time.sleep(poll_s)
    raise RuntimeError(
        f"motion-control service did not become ready within {timeout_s:.1f}s: {last_error}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=os.environ.get("A3_MC_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--action", choices=("MOTION", "PD_STAND"), default="MOTION")
    parser.add_argument("--get-up-timeout-s", type=float, default=45.0)
    parser.add_argument("--ready-timeout-s", type=float, default=45.0)
    parser.add_argument(
        "--passive-settle-s",
        type=float,
        default=1.5,
        help="keep the initial PASSIVE (zero-torque) state briefly before DAMPING",
    )
    parser.add_argument("--poll-s", type=float, default=1.0)
    args = parser.parse_args()

    if os.environ.get("SIM_MODE", "sil").lower() != "sil":
        print("Refusing to activate a non-SIL endpoint; set SIM_MODE=sil explicitly.", file=sys.stderr)
        return 2

    if args.get_up_timeout_s <= 0.0 or args.ready_timeout_s <= 0.0 or args.poll_s <= 0.0:
        parser.error("timeouts and poll interval must be positive")
    if args.passive_settle_s < 0.0:
        parser.error("passive settle time must be non-negative")

    initial = print_state(
        "initial", wait_for_service_action(args.endpoint, args.ready_timeout_s, args.poll_s)
    )
    expected = f"MotionControlAction_{args.action}"
    if initial["current_action"] == expected:
        print(f"[final] {expected} is already active")
        return 0

    # Let the official PASSIVE action remain zero-torque briefly so the robot
    # follows the normal fall-and-recovery path before the get-up policy takes
    # over. The table is visual-only during this phase.
    if initial["current_action"] == "MotionControlAction_PASSIVE" and args.passive_settle_s > 0.0:
        time.sleep(args.passive_settle_s)
        initial = print_state("post_passive", get_action(args.endpoint))

    # The official action graph does not allow MOTION -> GET_UP directly.
    # Route through DAMPING whenever the current action is neither DAMPING nor
    # GET_UP. PASSIVE can usually enter GET_UP directly, but DAMPING is safe
    # and makes the transition deterministic across SIL startup states.
    if initial["current_action"] not in (
        "MotionControlAction_DAMPING",
        "MotionControlAction_GET_UP",
    ):
        response = set_external_action(args.endpoint, "DAMPING")
        print(json.dumps(response, ensure_ascii=False))
        wait_for_action(args.endpoint, "DAMPING", args.get_up_timeout_s, args.poll_s)

    if get_action(args.endpoint).get("info", {}).get("current_action") != "MotionControlAction_GET_UP":
        response = set_external_action(args.endpoint, "GET_UP")
        print(json.dumps(response, ensure_ascii=False))

    deadline = time.monotonic() + args.get_up_timeout_s
    while True:
        state = print_state("get_up", get_action(args.endpoint))
        if state["stage"] == "ActionStage::GET_UP_FINISHED":
            break
        if time.monotonic() >= deadline:
            print("GET_UP did not reach GET_UP_FINISHED before timeout.", file=sys.stderr)
            return 1
        time.sleep(args.poll_s)

    response = set_external_action(args.endpoint, args.action)
    print(json.dumps(response, ensure_ascii=False))
    time.sleep(max(args.poll_s, 0.2))
    final = print_state("final", get_action(args.endpoint))
    if final["current_action"] != expected:
        print(f"Expected {expected}, got {final['current_action']}.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
