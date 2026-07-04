import gymnasium as gym

from . import agents, flat_env_cfg, hope_env_cfg

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
