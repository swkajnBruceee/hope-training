#!/usr/bin/env python3
"""Dependency-free mirror of the official A3 TA command conversion.

The reference implementation is
``a3_deploy_onnx_ref/src/a3_deploy/a3_teleop_reference.cpp``.
This module is a local contract test and JSON adapter only. It does not publish
commands to AimRT or to a robot.

Official command fields:
    leg(12), waist(3), head(2), arm(14)

Official A3 policy-view order:
    waist(3), arm(14), leg(12)

The head command is intentionally omitted from the 29-DOF policy view.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence


POLICY_DOF = 29
LEG_DOF = 12
WAIST_DOF = 3
ARM_DOF = 14
PELVIS_QUAT_DOF = 4


def _finite(values: Sequence[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _exact(values: Sequence[float], size: int, name: str) -> list[float]:
    result = _finite(values, name)
    if len(result) != size:
        raise ValueError(f"{name} must have size {size}, got {len(result)}")
    return result


@dataclass(frozen=True)
class TaWholeBodyCommand:
    pelvis_quat_wxyz: tuple[float, ...]
    leg_angles_rad: tuple[float, ...]
    waist_angles_rad: tuple[float, ...]
    arm_angles_rad: tuple[float, ...]
    joint_velocities_rad_s: tuple[float, ...] = ()
    stamp_ns: int = 1

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "TaWholeBodyCommand":
        return cls(
            pelvis_quat_wxyz=tuple(payload["pelvis_quat_wxyz"]),
            leg_angles_rad=tuple(payload["leg_angles_rad"]),
            waist_angles_rad=tuple(payload["waist_angles_rad"]),
            arm_angles_rad=tuple(payload["arm_angles_rad"]),
            joint_velocities_rad_s=tuple(payload.get("joint_velocities_rad_s", ())),
            stamp_ns=int(payload.get("stamp_ns", 1)),
        )


@dataclass(frozen=True)
class PolicyView:
    """The 29-DOF view consumed by the official A3 deployment tokenizer."""

    pelvis_quat_wxyz: tuple[float, ...]
    q_mujoco: tuple[float, ...]
    dq_mujoco: tuple[float, ...]
    stamp_ns: int


def map_ta_command(command: TaWholeBodyCommand) -> PolicyView:
    """Apply the official TA -> policy-view mapping and validations.

    Velocity layouts mirror the official C++ implementation:
      * 30: leg(12) + waist(3) + head(1) + arm(14)
      * 31: leg(12) + waist(3) + head(2) + arm(14)
      * 29: already in policy-view order
      * empty: zero feed-forward velocity
    """

    pelvis = _exact(command.pelvis_quat_wxyz, PELVIS_QUAT_DOF, "pelvis_quat_wxyz")
    if math.sqrt(sum(value * value for value in pelvis)) <= 1e-12:
        raise ValueError("pelvis_quat_wxyz must be non-zero")
    leg = _exact(command.leg_angles_rad, LEG_DOF, "leg_angles_rad")
    waist = _exact(command.waist_angles_rad, WAIST_DOF, "waist_angles_rad")
    arm = _exact(command.arm_angles_rad, ARM_DOF, "arm_angles_rad")
    stamp_ns = int(command.stamp_ns)
    if stamp_ns <= 0:
        raise ValueError("stamp_ns must be positive")

    # This is intentionally the official policy-view order, not Isaac's 31-DOF
    # articulation order and not the TA protocol's body order.
    q = waist + arm + leg
    velocities = _finite(command.joint_velocities_rad_s, "joint_velocities_rad_s")
    if len(velocities) == 0:
        dq = [0.0] * POLICY_DOF
    elif len(velocities) == 30:
        dq = velocities[12:15] + velocities[16:30] + velocities[0:12]
    elif len(velocities) == 31:
        dq = velocities[12:15] + velocities[17:31] + velocities[0:12]
    elif len(velocities) == POLICY_DOF:
        dq = velocities
    else:
        raise ValueError(
            "joint_velocities_rad_s must have size 0, 29, 30, or 31; "
            f"got {len(velocities)}"
        )

    assert len(q) == POLICY_DOF
    assert len(dq) == POLICY_DOF
    return PolicyView(tuple(pelvis), tuple(q), tuple(dq), stamp_ns)


def load_json(path: Path) -> PolicyView:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("TA fixture must be a JSON object")
    return map_ta_command(TaWholeBodyCommand.from_json(payload))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path, help="local TA command JSON fixture")
    args = parser.parse_args()
    view = load_json(args.fixture)
    print(
        json.dumps(
            {
                "status": "ok",
                "stamp_ns": view.stamp_ns,
                "policy_order": "waist(3)+arm(14)+leg(12)",
                "head_in_policy_view": False,
                "q_mujoco": list(view.q_mujoco),
                "dq_mujoco": list(view.dq_mujoco),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
