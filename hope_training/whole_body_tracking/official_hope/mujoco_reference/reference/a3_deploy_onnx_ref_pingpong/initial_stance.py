"""Optional initial poses for the project MuJoCo verifier.

The right-front values are the project training utility's
``V13B_RIGHT_FRONT_READY`` contract.  They are intentionally an initial-state
override only; the actor's action offset/default pose remains model_21800's
published deploy pose.
"""

from __future__ import annotations

RIGHT_FRONT_JOINTS: dict[str, float] = {
    "left_hip_pitch_joint": -0.1611863932866333,
    "right_hip_pitch_joint": -0.318726339288822,
    "left_knee_joint": 0.48,
    "right_knee_joint": 0.48,
    "left_ankle_pitch_joint": -0.31381360671336667,
    "right_ankle_pitch_joint": -0.156273660711178,
    "left_hip_roll_joint": 0.1462128429019041,
    "right_hip_roll_joint": -0.1462128429019041,
    "left_ankle_roll_joint": -0.07401284290190409,
    "right_ankle_roll_joint": 0.07401284290190409,
}

RIGHT_FRONT_ROOT_HEIGHT_DELTA_M = -0.02047000192530113

LEFT_FRONT_JOINTS: dict[str, float] = {
    "left_hip_pitch_joint": -0.318726339288822,
    "right_hip_pitch_joint": -0.1611863932866333,
    "left_knee_joint": 0.48,
    "right_knee_joint": 0.48,
    "left_ankle_pitch_joint": -0.156273660711178,
    "right_ankle_pitch_joint": -0.31381360671336667,
    "left_hip_roll_joint": 0.1462128429019041,
    "right_hip_roll_joint": -0.1462128429019041,
    "left_ankle_roll_joint": -0.07401284290190409,
    "right_ankle_roll_joint": 0.07401284290190409,
}

LEFT_FRONT_ROOT_HEIGHT_DELTA_M = RIGHT_FRONT_ROOT_HEIGHT_DELTA_M


def get_initial_stance(name: str | None) -> tuple[dict[str, float], float] | None:
    if name in (None, "standard"):
        return None
    if name == "right_front":
        return RIGHT_FRONT_JOINTS.copy(), RIGHT_FRONT_ROOT_HEIGHT_DELTA_M
    if name == "left_front":
        return LEFT_FRONT_JOINTS.copy(), LEFT_FRONT_ROOT_HEIGHT_DELTA_M
    raise ValueError(f"unknown initial stance: {name!r}")
