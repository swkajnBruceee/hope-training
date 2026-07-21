"""Register the isolated A3 H1-flat MDP replica."""

import gymnasium as gym

from . import agents, replica_env_cfg


gym.register(
    id="A3BaseStandReplica-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": replica_env_cfg.A3BaseStandReplicaEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:A3BaseStandReplicaPPORunnerCfg",
    },
)
