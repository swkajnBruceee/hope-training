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

gym.register(
    id="HOPE-FixedBaseTargetAdapter-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3FixedBaseTargetAdapterEnvCfg,
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

# P5D: generic reference-tracking PPO.  It intentionally has no historical
# target adapter or frozen-policy dependency: a policy action is only a
# bounded residual around the safe trajectory supplied by the manifest.
gym.register(
    id="HOPE-FloatingReferenceTracker-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3FloatingReferenceTrackerEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# P5D bootstrap: the same reference-preview tracker contract, initialized
# around the verified frozen 3396/900 execution and support state machine.
# It is separate from the pure P5D ablation above so reports can never claim
# that a prior-guided result came from reference-only control.
gym.register(
    id="HOPE-FloatingPriorGuidedReferenceTracker-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3FloatingPriorGuidedReferenceTrackerEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# P5U-1 unified tracker: model_3396 is the nominal lower prior and the
# historical model_900 upper prior is deliberately absent.  The NoAssist
# variant additionally enables a learned lower balance residual.
gym.register(
    id="HOPE-FloatingUnifiedUpperReferenceTracker-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3FloatingUnifiedUpperReferenceTrackerEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# P5U unified tracker without the progressive external fall-assist wrench.
# This keeps the historical P5U environment reproducible while providing an
# explicit no-assistance contract for the safe augmented-bank retraining run.
gym.register(
    id="HOPE-FloatingUnifiedUpperReferenceTrackerNoAssist-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3FloatingUnifiedUpperReferenceTrackerNoAssistEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# V1.3B: true reference-free target-conditioned student.  This registration
# is additive; all historical P5U reference tasks remain reproducible.
gym.register(
    id="HOPE-FloatingTargetConditionedReferenceFreeV13B-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3FloatingTargetConditionedReferenceFreeV13BEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# V1.3B training-only bridge: the actor remains reference-free, while a private
# stage-A observation feeds the additive model_3396 prior until alpha reaches 0.
gym.register(
    id="HOPE-FloatingTargetConditionedReferenceFreeV13BAnnealedPrior-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3FloatingTargetConditionedReferenceFreeV13BAnnealedPriorEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# V1.3B final training chain: private 3396 lower and complete model_900 upper
# priors are additive only during the scheduled bootstrap, while the public
# actor remains the same deployable 98-D reference-free policy.
gym.register(
    id="HOPE-FloatingTargetConditionedReferenceFreeV13BCompletePriors-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3FloatingTargetConditionedReferenceFreeV13BAnnealedPriorEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# Opt-in Precision Rescue continuation.  The current CompletePriors registry
# above remains untouched; this entry owns the two wide reward terms and the
# continuation-only prior withdrawal schedule.
gym.register(
    id="HOPE-FloatingTargetConditionedReferenceFreeV13BCompletePriorsPrecisionRescue-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3FloatingTargetConditionedReferenceFreeV13BCompletePriorsPrecisionRescueEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# V1.3B follow-up: pure actor workspace expansion.  This is intentionally a
# separate registration/configuration from CompletePriors; the manifest is
# consumed only as strike-anchor metadata and no private prior is instantiated.
gym.register(
    id="HOPE-FloatingTargetConditionedReferenceFreeV13BWorkspaceExpansion-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3FloatingTargetConditionedReferenceFreeV13BWorkspaceExpansionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

gym.register(
    id="HOPE-FloatingUnifiedUpperReferenceTrackerB-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3FloatingUnifiedUpperReferenceTrackerGlobalPhaseEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

gym.register(
    id="HOPE-FloatingUnifiedUpperReferenceTrackerC-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3FloatingUnifiedUpperReferenceTrackerGroupedPhaseEnvCfg,
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

# Current-contract retraining chain: historical Stage-A strategy with the
# corrected root work point, semantic swing observation, and current backhand
# strike-only manifest.  Keep this ID separate from historical checkpoints.
gym.register(
    id="HOPE-RetrainStrikeStabilizerA-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3RetrainStrikeStabilizerEnvCfg,
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

gym.register(
    id="HOPE-FloatingUpperCorrection-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3FloatingUpperCorrectionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

# Final floating-base strike stage: both historical capabilities remain frozen
# and a single PPO coordinator learns small leg/waist/right-arm corrections.
gym.register(
    id="HOPE-FloatingJointCoordinator-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": native_strike_env_cfg.A3FloatingJointCoordinatorEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEAgibotA3PPORunnerCfg",
    },
)

gym.register(
    id="HOPE-FloatingJointCoordinatorV2-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.native_strike_env_cfg:A3FloatingJointCoordinatorV2EnvCfg",
    },
)

gym.register(
    id="HOPE-FloatingJointCoordinatorV3-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.native_strike_env_cfg:A3FloatingJointCoordinatorV3EnvCfg",
    },
)

gym.register(
    id="HOPE-FloatingJointCoordinatorV4-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.native_strike_env_cfg:A3FloatingJointCoordinatorV4EnvCfg",
    },
)

gym.register(
    id="HOPE-FloatingJointCoordinatorV5Preview-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.native_strike_env_cfg:A3FloatingJointCoordinatorV5PreviewEnvCfg",
    },
)

gym.register(
    id="HOPE-FloatingJointCoordinatorV6MomentumPreview-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.native_strike_env_cfg:"
            "A3FloatingJointCoordinatorV6MomentumPreviewEnvCfg"
        ),
    },
)

gym.register(
    id="HOPE-FloatingJointCoordinatorV7StaggeredRecovery-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.native_strike_env_cfg:"
            "A3FloatingJointCoordinatorV7StaggeredRecoveryEnvCfg"
        ),
    },
)

gym.register(
    id="HOPE-FloatingJointCoordinatorV8StaggerSupport-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.native_strike_env_cfg:"
            "A3FloatingJointCoordinatorV8StaggerSupportEnvCfg"
        ),
    },
)

gym.register(
    id="HOPE-FloatingJointCoordinatorV9WideStaggerSupport-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.native_strike_env_cfg:"
            "A3FloatingJointCoordinatorV9WideStaggerSupportEnvCfg"
        ),
    },
)

gym.register(
    id="HOPE-FloatingJointCoordinatorV10WideStaggerRecovery-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.native_strike_env_cfg:"
            "A3FloatingJointCoordinatorV10WideStaggerRecoveryEnvCfg"
        ),
    },
)

gym.register(
    id="HOPE-FloatingJointCoordinatorV11BentReadyRecovery-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{native_strike_env_cfg.__name__}:"
            "A3FloatingJointCoordinatorV11BentReadyRecoveryEnvCfg"
        ),
    },
)

gym.register(
    id="HOPE-FloatingTargetConditionedCoordinator-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{native_strike_env_cfg.__name__}:"
            "A3FloatingTargetConditionedCoordinatorEnvCfg"
        ),
    },
)

gym.register(
    id="HOPE-FloatingTargetConditionedRecovery-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{native_strike_env_cfg.__name__}:"
            "A3FloatingTargetConditionedRecoveryEnvCfg"
        ),
    },
)

gym.register(
    id="HOPE-FloatingTargetConditionedRecoveryYComp-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{native_strike_env_cfg.__name__}:"
            "A3FloatingTargetConditionedRecoveryYCompEnvCfg"
        ),
    },
)

gym.register(
    id="HOPE-FloatingTargetConditionedRecoveryMotion0Calibrated-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{native_strike_env_cfg.__name__}:"
            "A3FloatingTargetConditionedRecoveryMotion0CalibratedEnvCfg"
        ),
    },
)

gym.register(
    id="HOPE-FloatingTargetConditionedRecoveryMotion2Calibrated-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{native_strike_env_cfg.__name__}:"
            "A3FloatingTargetConditionedRecoveryMotion2CalibratedEnvCfg"
        ),
    },
)

gym.register(
    id="HOPE-FloatingTargetConditionedRecoveryMotion4Calibrated-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{native_strike_env_cfg.__name__}:"
            "A3FloatingTargetConditionedRecoveryMotion4CalibratedEnvCfg"
        ),
    },
)

gym.register(
    id="HOPE-FloatingTargetConditionedRecoveryMotion5Calibrated-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{native_strike_env_cfg.__name__}:"
            "A3FloatingTargetConditionedRecoveryMotion5CalibratedEnvCfg"
        ),
    },
)

gym.register(
    id="HOPE-FloatingTargetConditionedRecoveryMotion1Train-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{native_strike_env_cfg.__name__}:"
            "A3FloatingTargetConditionedRecoveryMotion1TrainEnvCfg"
        ),
    },
)
