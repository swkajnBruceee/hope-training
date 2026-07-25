import gymnasium as gym

from . import agents, flat_env_cfg, hope_env_cfg, native_strike_env_cfg

##
# Register Gym environments.
##

# Plain BeyondMimic motion tracking on the A3 (baseline).
gym.register(
    id="Tracking-Flat-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.AgibotA3FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:AgibotA3FlatPPORunnerCfg",
    },
)

# HOPE ping-pong WBC with racket-target tracking (step 13/14).
gym.register(
    id="HOPE-PingPong-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": hope_env_cfg.HOPEPingPongAgibotA3EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# A3 native-MOTION strike route: policy commands only waist + right arm.
gym.register(
    id="HOPE-NativeStrike-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3NativeStrikeEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# Fixed-base reference-residual strike training.  This is intentionally a
# separate task id from the native deployment-contract task: it is a local
# motion-library training sandbox for balanced forehand/backhand data and does
# not assert the standalone A3 executor provenance contract used by the native
# deployment path.
gym.register(
    id="HOPE-FixedBaseReferenceStrike-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3FixedBaseReferenceStrikeEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# Backhand-only fixed-base target-conditioned residual stage.  This is kept as
# a separate task so the older mixed forehand/backhand fixed-base experiment
# and its checkpoints remain reproducible.
gym.register(
    id="HOPE-FixedBaseBackhandReferenceStrike-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3FixedBaseBackhandReferenceStrikeEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# Floating-base diagnostic for the self-developed Strike-conditioned Base14
# route.  It is intentionally registered separately from the native-MOTION
# strike executor and is not training-approved by registration alone.
gym.register(
    id="HOPE-StrikeConditionedBase-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3StrikeConditionedBaseEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# F0 paired migration audit: the evaluator toggles only root fixation and the
# external leg-action source while keeping model_900's upper contract shared.
gym.register(
    id="HOPE-FloatingF0-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3FloatingF0EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# Stage-A: fixed upper-body/waist replay with a learned 12-DOF leg stabilizer.
gym.register(
    id="HOPE-StrikeStabilizerA-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3StrikeStabilizerAEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# Unified Stage-A: same leg-only stabilizer plant, with the swing-family
# observation read directly from each manifest entry rather than inferred from
# target geometry.  Keep this ID separate so historical forehand-only runs
# remain bit-for-bit reproducible.
gym.register(
    id="HOPE-StrikeStabilizerAUnified-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3StrikeStabilizerAUnifiedEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# F1 in-place migration: freeze model_900 upper strike control and adapt only
# the Stage-A leg stabilizer on the current backhand manifest.
gym.register(
    id="HOPE-FloatingF1-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3FloatingF1EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)
