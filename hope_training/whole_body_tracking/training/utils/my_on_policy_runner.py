import math
import os
import json
from pathlib import Path

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

from training.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _set_v13b_progress(runner: OnPolicyRunner, iteration: int) -> None:
    env = runner.env.unwrapped
    if not hasattr(env, "v13b_policy_progress"):
        if not getattr(runner, "_v13b_progress_missing_reported", False):
            print("[V1.3B] curriculum progress target env attribute is missing", flush=True)
            runner._v13b_progress_missing_reported = True
        return
    # A diagnostic preflight may deliberately run only 20/100 updates while
    # checking the first portion of a 50k-update curriculum.  Keep the
    # curriculum clock independent from the requested diagnostic length so a
    # short run never silently fast-forwards the annealed priors to zero.
    schedule_total = int(
        getattr(env, "v13b_schedule_total_iterations", runner._v13b_max_iterations)
    )
    progress = min(1.0, max(0.0, iteration / max(schedule_total - 1, 1)))
    # PrecisionRescue is opt-in and continues from a selected historical
    # CompletePriors checkpoint.  It must never restart the sampler/prior
    # curriculum from zero.  Existing tasks have no such attribute and retain
    # the exact branch above.
    rescue_schedule = getattr(env, "v13b_precision_rescue_schedule", None)
    if rescue_schedule is not None:
        rescue_schedule.set_update(iteration)
        # Readiness is intentionally external to the actor.  A false/missing
        # gate holds the upper prior; it can never raise alpha.  Do *not*
        # withdraw here: an approved teacher-off probe is the only place
        # allowed to consume one bounded upper-prior step.
        rescue_schedule.set_readiness(bool(getattr(env, "v13b_precision_rescue_upper_ready", False)))
        progress = rescue_schedule.global_progress
        env.v13b_precision_rescue_snapshot = rescue_schedule.snapshot()
        # PrecisionRescue's first phase is explicitly an adaptation hold:
        # the selected checkpoint's upper-prior authority must be bitwise
        # continuous until the external teacher-off gate is eligible.  Do not
        # merely log a discrepancy here; continuing would silently train under
        # a different physical action distribution than the selected source.
        if iteration < int(rescue_schedule.hold_updates):
            expected = float(rescue_schedule.source_upper_alpha)
            observed = getattr(env, "v13b_precision_rescue_applied_upper_alpha", None)
            if observed is None or not math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=1.0e-6
            ):
                observed_text = "None" if observed is None else f"{float(observed):.9f}"
                raise RuntimeError(
                    "PrecisionRescue source-alpha hold violation: "
                    f"iteration={iteration} applied_upper_alpha={observed_text} "
                    f"expected_source_upper_alpha={expected:.9f}. "
                    "Refuse to continue with a changed upper-prior distribution."
                )
    env.v13b_policy_progress = progress
    workspace_route = hasattr(env, "workspace_curriculum_progress")
    if workspace_route:
        env.workspace_curriculum_progress = progress
    try:
        command = env.command_manager.get_term("racket_target")
        if workspace_route:
            command._workspace_curriculum_progress = progress
        else:
            command._v13b_policy_progress = progress
        if "v13b_curriculum_progress" in command.metrics:
            command.metrics["v13b_curriculum_progress"].fill_(progress)
        if "v13b_workspace_curriculum_progress" in command.metrics:
            command.metrics["v13b_workspace_curriculum_progress"].fill_(progress)
    except (AttributeError, KeyError, ValueError) as exc:
        if not getattr(runner, "_v13b_progress_error_reported", False):
            print(f"[V1.3B] curriculum command progress hook unavailable: {type(exc).__name__}: {exc}", flush=True)
            runner._v13b_progress_error_reported = True


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
        train_cfg = args[1] if len(args) > 1 else kwargs.get("train_cfg", {})
        self._v13b_max_iterations = int(train_cfg.get("max_iterations", 1)) if isinstance(train_cfg, dict) else 1
        self._precision_rescue_probe_pass_streak = 0
        self._precision_rescue_last_probe_update = -1
        print(
            f"[V1.3B] MyOnPolicyRunner active; run max_iterations={self._v13b_max_iterations}",
            flush=True,
        )
        _install_paired_common_exploration_noise(self)

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        # V1.3B has no motion command, so train.py selects this runner rather
        # than MotionOnPolicyRunner.  Keep curriculum progress on the actual
        # PPO-update path instead of relying on logging side effects elsewhere.
        iteration = int(locs.get("it", getattr(self, "current_learning_iteration", 0)))
        _set_v13b_progress(self, iteration)
        super().log(locs, width, pad)
        if self.disable_logs or self.writer is None:
            return
        self._log_v13b_metrics(int(locs.get("it", 0)))

    def _maybe_run_precision_rescue_upper_probe(self, iteration: int) -> None:
        """Run an evaluation-only, deterministic upper-prior-off readiness test.

        The probe deliberately reuses the active Isaac process and vectorized
        scene.  It runs after a PPO update, writes no rollout into the PPO
        buffer, and resets the scene before the next collection.  The public
        actor remains 98-D and never observes the private prior fields.
        """
        raw = self.env.unwrapped
        schedule = getattr(raw, "v13b_precision_rescue_schedule", None)
        if schedule is None:
            return
        probe_cfg = getattr(raw, "v13b_precision_rescue_upper_probe_config", {})
        interval = int(probe_cfg.get("interval_updates", 200))
        hold = int(schedule.hold_updates)
        raw.v13b_precision_rescue_upper_probe_interval_updates = float(interval)
        raw.v13b_precision_rescue_upper_probe_hold_updates = float(hold)
        if interval <= 0 or iteration < hold or iteration == self._precision_rescue_last_probe_update:
            return
        if (iteration - hold) % interval != 0:
            return
        self._precision_rescue_last_probe_update = iteration

        max_steps = int(probe_cfg.get("max_steps", 600))
        min_passes = int(probe_cfg.get("consecutive_passes", 2))
        min_survival = float(probe_cfg.get("min_survival", 0.95))
        min_hits = float(probe_cfg.get("min_hit_rate", 0.95))
        max_position = float(probe_cfg.get("max_position_error_m", 0.03))
        max_normal = float(probe_cfg.get("max_normal_error_deg", 35.0))
        max_velocity = float(probe_cfg.get("max_velocity_error_mps", 1.2))
        seed = int(probe_cfg.get("seed", 20260810))

        # Preserve the training RNG stream: the fixed validation set must not
        # perturb PPO exploration/noise after the probe finishes.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(raw.device) if raw.device.type == "cuda" else None
        sentinel = object()
        previous_upper = getattr(raw, "v13b_force_upper_prior_alpha", sentinel)
        previous_lower = getattr(raw, "v13b_force_lower_prior_alpha", sentinel)
        raw.v13b_force_upper_prior_alpha = 0.0
        raw.v13b_force_lower_prior_alpha = float(schedule.lower_alpha())

        report = {
            "ran": 1.0,
            "passed": 0.0,
            "survival": 0.0,
            "hit_rate": 0.0,
            "position_error_m": float("nan"),
            "normal_error_deg": float("nan"),
            "velocity_error_mps": float("nan"),
            "upper_alpha_before": float(schedule.upper_alpha()),
            "upper_alpha_after": float(schedule.upper_alpha()),
        }
        try:
            torch.manual_seed(seed)
            if raw.device.type == "cuda":
                torch.cuda.manual_seed(seed)
            self.env.reset()
            command = raw.command_manager.get_term("racket_target")
            ids = torch.arange(raw.num_envs, device=raw.device)
            command._resample_command(ids)
            command._compute_strike_timing()
            obs = self.env.get_observations()
            if isinstance(obs, tuple):
                obs = obs[0]
            obs = obs.to(raw.device)
            policy = self.get_inference_policy(device=raw.device)
            recorded = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
            ended = torch.zeros_like(recorded)
            timed_out = torch.zeros_like(recorded)
            position = torch.full((raw.num_envs,), float("nan"), device=raw.device)
            normal = torch.full_like(position, float("nan"))
            velocity = torch.full_like(position, float("nan"))
            with torch.no_grad():
                for _ in range(max_steps):
                    actions = policy(obs)
                    obs, _, terminated, truncated = self.env.step(actions)
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    obs = obs.to(raw.device)
                    command = raw.command_manager.get_term("racket_target")
                    hit = (command.metrics["exact_strike_hit_rate"] > 0.5) & (~recorded) & (~ended)
                    if torch.any(hit):
                        position[hit] = command.metrics["racket_pos_error_exact_strike"][hit]
                        normal[hit] = command.metrics["racket_normal_error_deg_exact_strike"][hit]
                        velocity[hit] = command.metrics["racket_vel_error_exact_strike"][hit]
                        recorded[hit] = True
                    done = torch.as_tensor(terminated, device=raw.device, dtype=torch.bool)
                    if torch.is_tensor(truncated):
                        timeout = torch.as_tensor(truncated, device=raw.device, dtype=torch.bool)
                    else:
                        timeout = torch.as_tensor(
                            truncated.get("time_outs", torch.zeros_like(done)), device=raw.device, dtype=torch.bool
                        )
                    fresh_done = done & (~ended)
                    timed_out[fresh_done] = timeout[fresh_done]
                    ended |= fresh_done
                    if bool(torch.all(ended).item()):
                        break
            finite_mean = lambda value: float(value[torch.isfinite(value)].mean().item()) if bool(torch.any(torch.isfinite(value)).item()) else float("nan")
            report["survival"] = float(timed_out.float().mean().item())
            report["hit_rate"] = float(recorded.float().mean().item())
            report["position_error_m"] = finite_mean(position)
            report["normal_error_deg"] = finite_mean(normal)
            report["velocity_error_mps"] = finite_mean(velocity)
            passed = (
                float(ended.float().mean().item()) >= 0.99
                and report["survival"] >= min_survival
                and report["hit_rate"] >= min_hits
                and math.isfinite(report["position_error_m"])
                and report["position_error_m"] <= max_position
                and math.isfinite(report["normal_error_deg"])
                and report["normal_error_deg"] <= max_normal
                and math.isfinite(report["velocity_error_mps"])
                and report["velocity_error_mps"] <= max_velocity
            )
            self._precision_rescue_probe_pass_streak = (
                self._precision_rescue_probe_pass_streak + 1 if passed else 0
            )
            if self._precision_rescue_probe_pass_streak >= min_passes:
                # One probe approval permits exactly one bounded decrement.
                schedule.set_readiness(True)
                schedule.advance_upper_once()
                schedule.set_readiness(False)
                self._precision_rescue_probe_pass_streak = 0
            report["passed"] = float(passed)
            report["upper_alpha_after"] = float(schedule.upper_alpha())
        except Exception as exc:
            # A failed probe must fail closed: retain the current prior level.
            self._precision_rescue_probe_pass_streak = 0
            print(f"[V1.3B PrecisionRescue] upper-off probe failed closed: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if previous_upper is sentinel:
                delattr(raw, "v13b_force_upper_prior_alpha")
            else:
                raw.v13b_force_upper_prior_alpha = previous_upper
            if previous_lower is sentinel:
                delattr(raw, "v13b_force_lower_prior_alpha")
            else:
                raw.v13b_force_lower_prior_alpha = previous_lower
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, raw.device)
            # Probes never contribute rollout data.  A clean reset gives the
            # next PPO collection an ordinary training distribution.
            self.env.reset()
        for name, value in report.items():
            setattr(raw, f"v13b_precision_rescue_upper_probe_{name}", value)
        raw.v13b_precision_rescue_upper_probe_pass_streak = float(self._precision_rescue_probe_pass_streak)

    def _queue_precision_rescue_upper_probe(self, iteration: int) -> None:
        """Queue, but do not execute, an eligible probe for the next rollout."""
        raw = self.env.unwrapped
        schedule = getattr(raw, "v13b_precision_rescue_schedule", None)
        if schedule is None:
            return
        probe_cfg = getattr(raw, "v13b_precision_rescue_upper_probe_config", {})
        interval = int(probe_cfg.get("interval_updates", 200))
        hold = int(schedule.hold_updates)
        if interval <= 0 or iteration < hold or iteration == self._precision_rescue_last_probe_update:
            return
        if (iteration - hold) % interval == 0:
            self._precision_rescue_probe_pending_iteration = int(iteration)

    def _log_v13b_metrics(self, step: int) -> None:
        """Record the annealed-prior contract on every PPO update.

        These are diagnostics only: a fall is a learning signal, never a
        V1.3B admission rejection.  They make it possible to verify that the
        student remains active while the two private priors are withdrawn.
        """
        env = self.env.unwrapped

        def mean(name: str):
            value = getattr(env, name, None)
            if isinstance(value, torch.Tensor) and value.numel():
                return value.detach().float().mean().item()
            if isinstance(value, (float, int)):
                return float(value)
            return None

        rescue_schedule = getattr(env, "v13b_precision_rescue_schedule", None)
        scheduled_lower = (
            float(rescue_schedule.lower_alpha()) if rescue_schedule is not None
            else mean("v13b_annealed_prior_alpha")
        )
        scheduled_upper = (
            float(rescue_schedule.upper_alpha()) if rescue_schedule is not None
            else mean("v13b_annealed_upper_prior_alpha")
        )
        values = {
            "V13B/curriculum_progress": mean("v13b_policy_progress"),
            # Global schedule values, not reset-contaminated per-env means.
            "V13B/alpha_lower": scheduled_lower,
            "V13B/alpha_upper": scheduled_upper,
            "V13B/alpha_lower_active_env_mean": mean("v13b_annealed_prior_alpha"),
            "V13B/alpha_upper_active_env_mean": mean("v13b_annealed_upper_prior_alpha"),
            "V13B/lower_prior_rms": mean("v13b_annealed_prior_prior_rms"),
            "V13B/lower_student_rms": mean("v13b_annealed_prior_student_rms"),
            "V13B/lower_student_prior_ratio": mean("v13b_annealed_prior_student_ratio"),
            "V13B/upper_prior_rms": mean("v13b_annealed_upper_prior_prior_rms"),
            "V13B/upper_student_rms": mean("v13b_annealed_upper_prior_student_rms"),
            "V13B/upper_student_prior_ratio": mean("v13b_annealed_upper_prior_student_ratio"),
            "V13B/upper_probe_ran": mean("v13b_precision_rescue_upper_probe_ran"),
            "V13B/upper_probe_interval_updates": mean("v13b_precision_rescue_upper_probe_interval_updates"),
            "V13B/upper_probe_hold_updates": mean("v13b_precision_rescue_upper_probe_hold_updates"),
            "V13B/upper_probe_pass": mean("v13b_precision_rescue_upper_probe_passed"),
            "V13B/upper_probe_pass_streak": mean("v13b_precision_rescue_upper_probe_pass_streak"),
            "V13B/upper_probe_survival": mean("v13b_precision_rescue_upper_probe_survival"),
            "V13B/upper_probe_hit_rate": mean("v13b_precision_rescue_upper_probe_hit_rate"),
            "V13B/upper_probe_position_error_m": mean("v13b_precision_rescue_upper_probe_position_error_m"),
            "V13B/upper_probe_normal_error_deg": mean("v13b_precision_rescue_upper_probe_normal_error_deg"),
            "V13B/upper_probe_velocity_error_mps": mean("v13b_precision_rescue_upper_probe_velocity_error_mps"),
        }
        for tag, value in values.items():
            if value is not None and math.isfinite(value):
                self.writer.add_scalar(tag, value, step)
        try:
            command = env.command_manager.get_term("racket_target")
            for name in (
                "v13b_workspace_curriculum_progress",
                "v13b_workspace_eligible_anchor_fraction",
                "v13b_workspace_eligible_anchor_count",
                "v13b_workspace_anchor_distance_m",
                "v13b_workspace_nominal_fallback_count",
                "v13b_workspace_out_of_bounds_reject_count",
                "v13b_workspace_resample_count",
                "v13b_workspace_motion_anchor_fraction",
                "v13b_workspace_global_fraction",
            ):
                value = command.metrics.get(name)
                if isinstance(value, torch.Tensor) and value.numel():
                    self.writer.add_scalar(f"V13B/Workspace/{name.removeprefix('v13b_workspace_')}", value.float().mean().item(), step)
        except (AttributeError, KeyError):
            pass

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
        self._v13b_max_iterations = int(train_cfg.get("max_iterations", 1)) if isinstance(train_cfg, dict) else 1
        self._precision_rescue_probe_pass_streak = 0
        self._precision_rescue_last_probe_update = -1
        self._precision_rescue_last_consumed_gate_index = 0
        _install_paired_common_exploration_noise(self)

    def _consume_precision_rescue_upper_gate(self, iteration: int) -> None:
        """Consume one externally verified teacher-off approval, at most once.

        The 10-second deterministic probe is run by a separate Isaac worker
        (normally GPU1).  It never resets or steps the PPO training scene.
        A malformed/stale gate fails closed and leaves the current alpha intact.
        """
        raw = self.env.unwrapped
        schedule = getattr(raw, "v13b_precision_rescue_schedule", None)
        gate_cfg = getattr(raw, "v13b_precision_rescue_upper_gate_config", None)
        if schedule is None or not isinstance(gate_cfg, dict):
            return
        path_text = str(gate_cfg.get("file", "")).strip()
        if not path_text or iteration < int(schedule.hold_updates):
            return
        path = Path(path_text)
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("contract") != "v13b_precision_rescue_upper_off_gate_v1":
                return
            if payload.get("run_id") != gate_cfg.get("run_id"):
                return
            if payload.get("source_checkpoint") != gate_cfg.get("source_checkpoint"):
                return
            approved = int(payload.get("approved_withdrawal_index", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return
        if approved != self._precision_rescue_last_consumed_gate_index + 1:
            return
        schedule.set_readiness(True)
        before = float(schedule.upper_alpha())
        after = float(schedule.advance_upper_once())
        schedule.set_readiness(False)
        if after > before + 1.0e-12:
            return
        self._precision_rescue_last_consumed_gate_index = approved
        raw.v13b_precision_rescue_upper_gate_consumed_index = float(approved)
        raw.v13b_precision_rescue_upper_gate_alpha_before = before
        raw.v13b_precision_rescue_upper_gate_alpha_after = after
        print(
            "[V1.3B PrecisionRescue] external upper-off gate consumed: "
            f"index={approved} alpha={before:.3f}->{after:.3f}",
            flush=True,
        )

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
        iteration = int(locs.get("it", getattr(self, "current_learning_iteration", 0)))
        _set_v13b_progress(self, iteration)
        self._consume_precision_rescue_upper_gate(iteration)
        _set_v13b_progress(self, iteration)
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

        # V1.3B has a private motion command during training, therefore it is
        # routed through MotionOnPolicyRunner even though its public actor is
        # reference-free.  Keep the annealed-prior audit here (rather than in
        # MyOnPolicyRunner only) so the values are present in every V1.3B run.
        rescue_schedule = getattr(env, "v13b_precision_rescue_schedule", None)
        if rescue_schedule is not None:
            # The global schedule is the authoritative alpha.  Per-env action
            # buffers are intentionally reset to zero on a terminated env and
            # are exported separately below as active-env diagnostics.
            self._log_scalar("V13B/alpha_lower", float(rescue_schedule.lower_alpha()), step)
            self._log_scalar("V13B/alpha_upper", float(rescue_schedule.upper_alpha()), step)
        for tag, attr in (
            ("V13B/curriculum_progress", "v13b_policy_progress"),
            (
                "V13B/alpha_lower_active_env_mean"
                if rescue_schedule is not None else "V13B/alpha_lower",
                "v13b_annealed_prior_alpha",
            ),
            (
                "V13B/alpha_upper_active_env_mean"
                if rescue_schedule is not None else "V13B/alpha_upper",
                "v13b_annealed_upper_prior_alpha",
            ),
            ("V13B/lower_prior_rms", "v13b_annealed_prior_prior_rms"),
            ("V13B/lower_student_rms", "v13b_annealed_prior_student_rms"),
            ("V13B/lower_student_prior_ratio", "v13b_annealed_prior_student_ratio"),
            ("V13B/upper_prior_rms", "v13b_annealed_upper_prior_prior_rms"),
            ("V13B/upper_student_rms", "v13b_annealed_upper_prior_student_rms"),
            ("V13B/upper_student_prior_ratio", "v13b_annealed_upper_prior_student_ratio"),
            ("V13B/upper_probe_ran", "v13b_precision_rescue_upper_probe_ran"),
            ("V13B/upper_probe_pass", "v13b_precision_rescue_upper_probe_passed"),
            ("V13B/upper_probe_pass_streak", "v13b_precision_rescue_upper_probe_pass_streak"),
            ("V13B/upper_probe_survival", "v13b_precision_rescue_upper_probe_survival"),
            ("V13B/upper_probe_hit_rate", "v13b_precision_rescue_upper_probe_hit_rate"),
            ("V13B/upper_probe_position_error_m", "v13b_precision_rescue_upper_probe_position_error_m"),
            ("V13B/upper_probe_normal_error_deg", "v13b_precision_rescue_upper_probe_normal_error_deg"),
            ("V13B/upper_probe_velocity_error_mps", "v13b_precision_rescue_upper_probe_velocity_error_mps"),
        ):
            self._log_scalar(tag, self._mean_tensor(getattr(env, attr, None)), step)

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
