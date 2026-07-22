"""Register the bounded A3 Base Stand smoke environment."""

import gymnasium as gym

from . import agents, stand_env_cfg


gym.register(
    id="A3BaseStand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stand_env_cfg.A3BaseStandEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:A3BaseStandPPORunnerCfg",
    },
)

gym.register(
    id="A3BaseStandAuthorityCandidate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stand_env_cfg.A3BaseStandAuthorityCandidateEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:A3BaseStandPPORunnerCfg",
    },
)

gym.register(
    id="A3BaseStandClipCandidate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stand_env_cfg.A3BaseStandClipCandidateEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:A3BaseStandPPORunnerCfg",
    },
)

gym.register(
    id="A3BaseStandAuthorityClipCandidate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stand_env_cfg.A3BaseStandAuthorityClipCandidateEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:A3BaseStandPPORunnerCfg",
    },
)

gym.register(
    id="A3BaseStandPassiveStableCandidate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stand_env_cfg.A3BaseStandPassiveStableCandidateEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:A3BaseStandLowNoisePPORunnerCfg",
    },
)

# Static receive-ready stance: a wider, flexed base working point qualified
# before it is combined with any upper-body strike reference.
gym.register(
    id="A3CatchReadyStand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stand_env_cfg.A3CatchReadyStandEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:A3BaseStandLowNoisePPORunnerCfg",
    },
)

gym.register(
    id="A3BaseStandRecoveryA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stand_env_cfg.A3BaseStandRecoveryAEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:A3BaseStandRecoveryAPPORunnerCfg",
    },
)

for _task_id, _cfg in (
    ("A3BaseStandRecoveryAV2-v0", stand_env_cfg.A3BaseStandRecoveryAV2EnvCfg),
    ("A3BaseStandRecoveryAV2WaistMask-v0", stand_env_cfg.A3BaseStandRecoveryAV2WaistMaskEnvCfg),
    ("A3BaseStandRecoveryAV21WaistMask-v0", stand_env_cfg.A3BaseStandRecoveryAV21WaistMaskEnvCfg),
): 
    gym.register(
        id=_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _cfg,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:A3BaseStandRecoveryAPPORunnerCfg",
        },
    )
