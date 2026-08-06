"""RSL-RL configuration registered for the bounded Stand smoke task."""

import copy

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from training.utils.ppo_cfg import load_ppo_params, runner_kwargs


_KW = runner_kwargs(load_ppo_params(), "a3_base_stand_smoke")
_LOW_NOISE_KW = copy.deepcopy(_KW)
_LOW_NOISE_KW["policy"].init_noise_std = 0.15
_LOW_NOISE_KW["algorithm"].entropy_coef = 0.004


@configclass
class A3BaseStandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = _KW["num_steps_per_env"]
    # This is only a registry default.  The gate requires an explicit smoke
    # invocation capped to 100--500 iterations; long training remains closed.
    max_iterations = min(int(_KW["max_iterations"]), 500)
    save_interval = _KW["save_interval"]
    experiment_name = _KW["experiment_name"]
    empirical_normalization = _KW["empirical_normalization"]
    policy = _KW["policy"]
    algorithm = _KW["algorithm"]


@configclass
class A3BaseStandLowNoisePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Fail-closed low-noise runner for the passive-stable candidate only."""

    num_steps_per_env = _LOW_NOISE_KW["num_steps_per_env"]
    max_iterations = 500
    save_interval = _LOW_NOISE_KW["save_interval"]
    experiment_name = "a3_base_stand_passive_stable_candidate"
    empirical_normalization = _LOW_NOISE_KW["empirical_normalization"]
    policy = _LOW_NOISE_KW["policy"]
    algorithm = _LOW_NOISE_KW["algorithm"]


@configclass
class A3BaseStandRecoveryAPPORunnerCfg(A3BaseStandLowNoisePPORunnerCfg):
    """Separate log namespace for the still-gated Recovery-A curriculum."""

    experiment_name = "a3_base_stand_recovery_a"
