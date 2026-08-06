import math
import os

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

from training.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _install_paired_common_exploration_noise(runner: OnPolicyRunner) -> None:
    """Share PPO exploration noise inside each seven-environment target pair.

    The policy mean remains target-conditioned per environment.  Only the
    standard-normal sample is copied from the group's zero-offset baseline,
    which keeps every marginal action distribution valid while removing
    exploration noise from paired incremental rewards.
    """
    env = runner.env.unwrapped
    if not hasattr(env, "command_manager"):
        return
    try:
        command = env.command_manager.get_term("racket_target")
    except (KeyError, ValueError):
        return
    if not bool(getattr(command.cfg, "adapter_external_paired", False)):
        return

    original_act = runner.alg.act

    def paired_act(observations, privileged_observations):
        actions = original_act(observations, privileged_observations)
        policy = runner.alg.policy
        mean = policy.action_mean.detach()
        sigma = policy.action_std.detach().clamp_min(1.0e-8)
        standardized_noise = (actions - mean) / sigma
        baseline = command.adapter_pair_baseline_env.to(
            device=actions.device, dtype=torch.long
        )
        paired_actions = mean + standardized_noise[baseline] * sigma

        # PPO must store the action that was actually executed, along with its
        # probability under each sibling's own target-conditioned mean.
        runner.alg.transition.actions = paired_actions.detach()
        runner.alg.transition.actions_log_prob = policy.get_actions_log_prob(
            paired_actions
        ).detach()
        runner.alg.transition.action_mean = mean
        runner.alg.transition.action_sigma = sigma
        return runner.alg.transition.actions

    runner.alg.act = paired_act
    print(
        "[paired-target] sharing PPO exploration noise within each 7-env group",
        flush=True,
    )


class MyOnPolicyRunner(OnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _install_paired_common_exploration_noise(self)

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            import wandb

            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(
                self.alg.policy,
                normalizer=getattr(self.alg.policy, "actor_obs_normalizer", None),
                path=policy_path,
                filename=filename,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
        _install_paired_common_exploration_noise(self)

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            import wandb

            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            # rsl_rl moved obs normalization onto the policy (actor_obs_normalizer); the runner no
            # longer has self.obs_normalizer. Fall back to None (-> Identity in the exporter) if absent.
            export_motion_policy_as_onnx(
                self.env.unwrapped,
                self.alg.policy,
                normalizer=getattr(self.alg.policy, "actor_obs_normalizer", None),
                path=policy_path,
                filename=filename,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        super().log(locs, width=width, pad=pad)
        if self.disable_logs or self.writer is None:
            return
        self._log_live_metrics(locs["it"])

    def _log_live_metrics(self, step: int) -> None:
        """Log current manager state means every PPO iteration for richer dashboards."""
        env = self.env.unwrapped
        policy = self.alg.policy
        if hasattr(policy, "preview_adapter"):
            with torch.no_grad():
                self._log_scalar(
                    "Live/V19/preview_adapter_weight_l2",
                    torch.linalg.vector_norm(policy.preview_adapter.weight).item(),
                    step,
                )
                self._log_scalar(
                    "Live/V19/preview_encoder_weight_l2",
                    sum(
                        torch.linalg.vector_norm(parameter).item()
                        for name, parameter in policy.preview_encoder.named_parameters()
                        if name.endswith("weight")
                    ),
                    step,
                )
                self._log_scalar(
                    "Live/V19/preview_state_gate_weight_l2",
                    torch.linalg.vector_norm(policy.preview_state_gate.weight).item(),
                    step,
                )
                self._log_scalar(
                    "Live/V19/support_state_encoder_weight_l2",
                    sum(
                        torch.linalg.vector_norm(parameter).item()
                        for name, parameter in policy.support_state_encoder.named_parameters()
                        if name.endswith("weight")
                    ),
                    step,
                )
                self._log_scalar(
                    "Live/V19/support_policy_std",
                    policy.std[: policy.support_action_dim].mean().item(),
                    step,
                )
                self._log_scalar(
                    "Live/V19/fixed_arm_policy_std",
                    float(policy.fixed_arm_std),
                    step,
                )
        if hasattr(policy, "recovery_adapter"):
            with torch.no_grad():
                self._log_scalar(
                    "Live/V23/recovery_adapter_weight_l2",
                    torch.linalg.vector_norm(policy.recovery_adapter.weight).item(),
                    step,
                )
                self._log_scalar(
                    "Live/V23/recovery_adapter_bias_l2",
                    torch.linalg.vector_norm(policy.recovery_adapter.bias).item(),
                    step,
                )
                self._log_scalar(
                    "Live/V23/recovery_encoder_weight_l2",
                    sum(
                        torch.linalg.vector_norm(parameter).item()
                        for name, parameter in policy.recovery_encoder.named_parameters()
                        if name.endswith("weight")
                    ),
                    step,
                )
                self._log_scalar(
                    "Live/V23/recovery_policy_std",
                    policy.std.mean().item(),
                    step,
                )

        if hasattr(env, "command_manager"):
            for term_name in env.command_manager.active_terms:
                term = env.command_manager.get_term(term_name)
                for metric_name, metric_value in term.metrics.items():
                    self._log_scalar(f"Live/{term_name}/{metric_name}", self._mean_tensor(metric_value), step)
                if hasattr(term, "command_counter"):
                    self._log_scalar(
                        f"Live/{term_name}/command_counter", self._mean_tensor(term.command_counter), step
                    )

        if hasattr(env, "reward_manager"):
            self._log_scalar("Live/Reward/total", self._mean_tensor(getattr(env, "reward_buf", None)), step)
            for idx, term_name in enumerate(env.reward_manager.active_terms):
                self._log_scalar(
                    f"Live/Reward/{term_name}", self._mean_tensor(env.reward_manager._step_reward[:, idx]), step
                )

        if hasattr(env, "termination_manager"):
            tm = env.termination_manager
            self._log_scalar("Live/Termination/done_rate", self._mean_tensor(tm.dones), step)
            self._log_scalar("Live/Termination/terminated_rate", self._mean_tensor(tm.terminated), step)
            self._log_scalar("Live/Termination/timeout_rate", self._mean_tensor(tm.time_outs), step)
            for term_name in tm.active_terms:
                self._log_scalar(f"Live/Termination/{term_name}", self._mean_tensor(tm.get_term(term_name)), step)

        if hasattr(env, "action_manager"):
            action = getattr(env.action_manager, "action", None)
            term = env.action_manager.get_term("joint_pos")
            if hasattr(term, "raw_actions") and term.raw_actions.shape[-1] == 22:
                for group_name, start, end in (
                    ("leg", 0, 12),
                    ("waist", 12, 15),
                    ("arm", 15, 22),
                ):
                    group = term.raw_actions[:, start:end]
                    self._log_scalar(
                        f"Live/V19/{group_name}_raw_abs_mean",
                        group.abs().mean().item(),
                        step,
                    )
                    self._log_scalar(
                        f"Live/V19/{group_name}_raw_clip_fraction",
                        (group.abs() >= 0.999).float().mean().item(),
                        step,
                    )
            prev_action = getattr(env.action_manager, "prev_action", None)
            if action is not None:
                action_abs = torch.abs(action)
                self._log_scalar("Live/Action/mean_abs", self._mean_tensor(action_abs), step)
                self._log_scalar("Live/Action/max_abs", self._mean_tensor(torch.max(action_abs, dim=-1).values), step)
            if action is not None and prev_action is not None:
                action_delta_abs = torch.abs(action - prev_action)
                self._log_scalar("Live/Action/delta_mean_abs", self._mean_tensor(action_delta_abs), step)
                self._log_scalar(
                    "Live/Action/delta_max_abs",
                    self._mean_tensor(torch.max(action_delta_abs, dim=-1).values),
                    step,
                )

        self._log_scalar("Live/Env/episode_length", self._mean_tensor(env.episode_length_buf), step)
        self._log_scalar("Live/Env/common_step_counter", float(getattr(env, "common_step_counter", 0)), step)

    @staticmethod
    def _mean_tensor(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return value.float().mean().item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _log_scalar(self, tag: str, value, step: int) -> None:
        if value is None or not math.isfinite(float(value)):
            return
        self.writer.add_scalar(tag, float(value), step)
