"""Narrow target-to-model input seam; low-level execution remains Stage5-owned."""
import numpy as np

LOW_LEVEL_EXECUTOR_REUSED_FROM_VALIDATED_STAGE5 = True
HIGH_LEVEL_CONTRACT_REPLACED_BY_FROZEN_V2_CONTRACT = True

def bind_target_to_observation(observation_110d, *, target_position_world, base_position_world,
                               target_velocity_world, time_to_strike_s):
    obs = np.asarray(observation_110d, dtype=np.float32).copy()
    pos, base, vel = map(lambda x: np.asarray(x, dtype=np.float32),
                         (target_position_world, base_position_world, target_velocity_world))
    if obs.shape != (110,) or pos.shape != (3,) or base.shape != (3,) or vel.shape != (3,):
        raise ValueError("invalid model21800 input shape")
    obs[103:106] = pos - base
    obs[106:109] = vel
    obs[109] = float(time_to_strike_s)
    if not np.isfinite(obs).all() or obs[109] <= 0:
        raise ValueError("model21800 observation must be finite with positive TTS")
    return obs

def validate_policy_io(observation, time_step, actions):
    return (np.asarray(observation).shape == (110,) and np.isfinite(observation).all()
            and np.asarray(time_step).shape in ((), (1,)) and np.isfinite(time_step).all()
            and np.asarray(actions).shape == (31,) and np.isfinite(actions).all())
