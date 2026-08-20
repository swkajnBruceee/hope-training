"""Deployable observation boundary; simulator truth is never accepted here."""
from dataclasses import dataclass
import numpy as np

from hope_training.whole_body_tracking.deployment_v2 import age_command_timing

OBS_NAMES = (
    "predicted_intercept_y_world", "predicted_intercept_z_world",
    "incoming_vx_world", "incoming_vy_world", "incoming_vz_world",
    "control_time_to_intercept", "swing_sign",
)
OBS_NORMALIZATION = "NONE_IN_CONTRACT_V1"

@dataclass(frozen=True)
class PredictorCommand:
    position_world: tuple[float, float, float]
    incoming_velocity_world: tuple[float, float, float]
    header_stamp_s: float
    time_to_strike_s: float

def build_v2_observation(command: PredictorCommand, source_now_s: float, swing_sign: int) -> np.ndarray:
    if swing_sign not in (-1, 1):
        raise ValueError("swing_sign must be exactly +/-1")
    position = np.asarray(command.position_world, dtype=np.float64)
    velocity = np.asarray(command.incoming_velocity_world, dtype=np.float64)
    if position.shape != (3,) or velocity.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("predictor command vectors must be finite shape (3,)")
    control_tts = age_command_timing(command.header_stamp_s, source_now_s, command.time_to_strike_s)
    obs = np.asarray((position[1], position[2], *velocity, control_tts, swing_sign), dtype=np.float32)
    if obs.shape != (7,) or not np.isfinite(obs).all():
        raise ValueError("V2 observation must be finite float32 shape (7,)")
    return obs
