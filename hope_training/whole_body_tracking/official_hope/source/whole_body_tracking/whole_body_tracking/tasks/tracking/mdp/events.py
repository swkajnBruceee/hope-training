from __future__ import annotations

import math
import re

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.actuators import ImplicitActuator
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from whole_body_tracking.utils.stance_curriculum import smoothstep_stance_alpha

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class randomize_structured_pd_gains(ManagerTermBase):
    """Randomize A3 PD gains with one correlated latent per actuator family.

    Unlike Isaac Lab's generic gain randomizer, this term keeps left/right
    homologous joints correlated and couples damping to stiffness:

    ``Kp' = alpha * Kp`` and ``Kd' = beta * sqrt(alpha) * Kd``.

    A deterministic fraction of environments stays exactly nominal.  The
    default joint gains are read as the immutable anchor on every reset, so
    repeated resets never compound the randomization and action scale remains
    independent of the sampled plant.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.group_patterns = {
            str(name): tuple(str(pattern) for pattern in patterns)
            for name, patterns in dict(cfg.params["groups"]).items()
        }
        self.group_names = tuple(self.group_patterns)
        if not self.group_names:
            raise ValueError("structured PD randomization requires actuator groups")

        self.alpha_range = self._validate_range(
            cfg.params["alpha_range"], "alpha_range"
        )
        self.beta_range = self._validate_range(
            cfg.params["beta_range"], "beta_range"
        )
        self.nominal_fraction = float(cfg.params["nominal_fraction"])
        if (
            not math.isfinite(self.nominal_fraction)
            or self.nominal_fraction < 0.0
            or self.nominal_fraction > 1.0
        ):
            raise ValueError("nominal_fraction must be finite and in [0, 1]")

        self.activation_command_name = str(
            cfg.params.get("activation_command_name", "")
        )
        self.activation_recovery_scale = float(
            cfg.params.get("activation_recovery_scale", 0.0)
        )
        if (
            not math.isfinite(self.activation_recovery_scale)
            or not 0.0 <= self.activation_recovery_scale <= 1.0
        ):
            raise ValueError("activation_recovery_scale must be finite and in [0, 1]")
        self.activation_requires_coverage_unlocked = bool(
            cfg.params.get("activation_requires_coverage_unlocked", False)
        )
        self.randomization_enabled = self.activation_command_name == ""

        excluded = set(str(name) for name in cfg.params.get("excluded_joint_names", ()))
        assignments: dict[str, list[int]] = {name: [] for name in self.group_names}
        for joint_id, joint_name in enumerate(self.asset.joint_names):
            if joint_name in excluded:
                continue
            matches = [
                group_name
                for group_name, patterns in self.group_patterns.items()
                if any(re.fullmatch(pattern, joint_name) for pattern in patterns)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "structured PD groups must match every non-excluded joint exactly once: "
                    f"joint={joint_name!r}, matches={matches}"
                )
            assignments[matches[0]].append(joint_id)
        empty = [name for name, ids in assignments.items() if not ids]
        if empty:
            raise RuntimeError(f"structured PD groups matched no joints: {empty}")
        self.group_joint_ids = tuple(
            torch.tensor(assignments[name], dtype=torch.long, device=self.asset.device)
            for name in self.group_names
        )

        num_envs = int(env.scene.num_envs)
        num_groups = len(self.group_names)
        self.alpha = torch.ones(num_envs, num_groups, device=self.asset.device)
        self.beta = torch.ones_like(self.alpha)
        self.nominal_mask = torch.ones(
            num_envs, dtype=torch.bool, device=self.asset.device
        )

    @staticmethod
    def _validate_range(value, label: str) -> tuple[float, float]:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(f"{label} must be a [lo, hi] pair")
        lo, hi = float(value[0]), float(value[1])
        if (
            not math.isfinite(lo)
            or not math.isfinite(hi)
            or lo <= 0.0
            or hi < lo
        ):
            raise ValueError(f"{label} must satisfy 0 < lo <= hi")
        return lo, hi

    def _select_nominal(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self.nominal_fraction <= 0.0:
            return torch.zeros_like(env_ids, dtype=torch.bool)
        if self.nominal_fraction >= 1.0:
            return torch.ones_like(env_ids, dtype=torch.bool)
        reciprocal = 1.0 / self.nominal_fraction
        stride = int(round(reciprocal))
        if math.isclose(
            reciprocal, float(stride), rel_tol=0.0, abs_tol=1.0e-9
        ):
            # Stable anchor population: env 0,4,8,... stays nominal for 25%.
            return torch.remainder(env_ids, stride) == 0
        return (
            torch.rand(env_ids.numel(), device=self.asset.device)
            < self.nominal_fraction
        )

    def _publish_metrics(self, env: ManagerBasedEnv) -> None:
        command_manager = getattr(env, "command_manager", None)
        if command_manager is None:
            return
        try:
            command = command_manager.get_term("racket_target")
        except (AttributeError, KeyError, ValueError):
            return
        command.metrics["pd_nominal_env"] = self.nominal_mask.float().clone()
        command.metrics["pd_randomization_enabled"] = torch.full(
            (env.scene.num_envs,),
            float(self.randomization_enabled),
            device=self.asset.device,
        )
        for group_index, group_name in enumerate(self.group_names):
            command.metrics[f"pd_alpha_{group_name}"] = (
                self.alpha[:, group_index].clone()
            )
            command.metrics[f"pd_beta_{group_name}"] = (
                self.beta[:, group_index].clone()
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        groups: dict,
        alpha_range: tuple[float, float],
        beta_range: tuple[float, float],
        nominal_fraction: float,
        excluded_joint_names: tuple[str, ...] = (),
        activation_command_name: str = "",
        activation_recovery_scale: float = 0.0,
        activation_requires_coverage_unlocked: bool = False,
    ):
        del asset_cfg, groups, alpha_range, beta_range, nominal_fraction
        del excluded_joint_names, activation_command_name
        del activation_recovery_scale, activation_requires_coverage_unlocked
        if env_ids is None:
            env_ids = torch.arange(
                env.scene.num_envs, device=self.asset.device, dtype=torch.long
            )
        else:
            env_ids = env_ids.to(device=self.asset.device, dtype=torch.long)

        enabled = self.activation_command_name == ""
        if not enabled:
            try:
                command = env.command_manager.get_term(
                    self.activation_command_name
                )
            except (AttributeError, KeyError, ValueError) as exc:
                raise RuntimeError(
                    "structured PD activation command is unavailable: "
                    f"{self.activation_command_name!r}"
                ) from exc
            enabled = float(
                getattr(command, "_recovery_current_scale", 0.0)
            ) >= self.activation_recovery_scale - 1.0e-12
            if self.activation_requires_coverage_unlocked:
                enabled = enabled and bool(
                    getattr(
                        command,
                        "_recovery_stage1_coverage_unlocked",
                        False,
                    )
                )
        self.randomization_enabled = bool(enabled)

        if self.randomization_enabled:
            nominal = self._select_nominal(env_ids)
            sampled_alpha = torch.empty(
                env_ids.numel(), len(self.group_names), device=self.asset.device
            ).uniform_(*self.alpha_range)
            sampled_beta = torch.empty_like(sampled_alpha).uniform_(*self.beta_range)
            sampled_alpha[nominal] = 1.0
            sampled_beta[nominal] = 1.0
        else:
            nominal = torch.ones_like(env_ids, dtype=torch.bool)
            sampled_alpha = torch.ones(
                env_ids.numel(), len(self.group_names), device=self.asset.device
            )
            sampled_beta = torch.ones_like(sampled_alpha)
        self.alpha[env_ids] = sampled_alpha
        self.beta[env_ids] = sampled_beta
        self.nominal_mask[env_ids] = nominal

        stiffness = self.asset.data.default_joint_stiffness[env_ids].clone()
        damping = self.asset.data.default_joint_damping[env_ids].clone()
        for group_index, joint_ids in enumerate(self.group_joint_ids):
            alpha = sampled_alpha[:, group_index].unsqueeze(-1)
            beta = sampled_beta[:, group_index].unsqueeze(-1)
            stiffness[:, joint_ids] *= alpha
            damping[:, joint_ids] *= beta * torch.sqrt(alpha)

        self.asset.write_joint_stiffness_to_sim(
            stiffness, env_ids=env_ids
        )
        self.asset.write_joint_damping_to_sim(damping, env_ids=env_ids)
        for actuator in self.asset.actuators.values():
            joint_ids = actuator.joint_indices
            actuator.stiffness[env_ids] = stiffness[:, joint_ids]
            actuator.damping[env_ids] = damping[:, joint_ids]
            if not isinstance(actuator, ImplicitActuator):
                raise RuntimeError(
                    "RallyV17 structured PD contract requires implicit actuators"
                )
        self._publish_metrics(env)


class randomize_effective_ground_friction(ManagerTermBase):
    """Sample static and dynamic foot-floor friction per environment at reset.

    The venue-informed nominal contract is ``mu_static=1.2`` and ``mu_dynamic=0.8``.
    During the stance migration this pair is fixed.  The later curriculum expands a
    competition distribution by sampling static friction first and then a correlated
    dynamic/static ratio.  Once the expansion is complete, 20% of resets use a separate
    low-friction stress distribution.  The sampled values are not observations; they are
    exported only through command metrics.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.static_nominal = float(cfg.params.get("static_nominal", 1.2))
        self.dynamic_nominal = float(cfg.params.get("dynamic_nominal", 0.8))
        self.competition_static_min = float(cfg.params.get("competition_static_min", 1.0))
        self.competition_static_max = float(cfg.params.get("competition_static_max", 1.5))
        self.competition_ratio_min = float(cfg.params.get("competition_ratio_min", 0.65))
        self.competition_ratio_max = float(cfg.params.get("competition_ratio_max", 0.90))
        self.stress_probability = float(cfg.params.get("stress_probability", 0.20))
        self.stress_static_min = float(cfg.params.get("stress_static_min", 0.5))
        self.stress_static_max = float(cfg.params.get("stress_static_max", 1.0))
        self.stress_dynamic_min = float(cfg.params.get("stress_dynamic_min", 0.3))
        self.stress_dynamic_max = float(cfg.params.get("stress_dynamic_max", 0.7))
        self.curriculum_start_iteration = int(
            cfg.params.get("curriculum_start_iteration", 2100)
        )
        self.curriculum_end_iteration = int(
            cfg.params.get("curriculum_end_iteration", 2700)
        )
        self.restitution_range = tuple(float(x) for x in cfg.params.get("restitution_range", (0.0, 0.0)))
        self.bucket_edges = tuple(float(x) for x in cfg.params.get(
            "bucket_edges", (0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5)
        ))
        if not (0.0 < self.dynamic_nominal <= self.static_nominal):
            raise ValueError("effective ground friction requires 0 < dynamic_nominal <= static_nominal")
        nominal_ratio = self.dynamic_nominal / self.static_nominal
        if not (0.0 < self.competition_static_min <= self.static_nominal <= self.competition_static_max):
            raise ValueError("competition static range must contain static_nominal")
        if not (self.competition_ratio_min <= nominal_ratio <= self.competition_ratio_max):
            raise ValueError("competition ratio range must contain dynamic_nominal/static_nominal")
        if not (0.0 <= self.stress_probability <= 1.0):
            raise ValueError("stress_probability must be in [0, 1]")
        if not (0.0 < self.stress_static_min <= self.stress_static_max):
            raise ValueError("stress static range must be positive and ordered")
        if not (0.0 < self.stress_dynamic_min <= self.stress_dynamic_max):
            raise ValueError("stress dynamic range must be positive and ordered")
        if self.curriculum_start_iteration < 0 or self.curriculum_end_iteration <= self.curriculum_start_iteration:
            raise ValueError(
                "friction curriculum requires 0 <= start_iteration < end_iteration"
            )
        if len(self.restitution_range) != 2 or self.restitution_range[1] < self.restitution_range[0]:
            raise ValueError("restitution_range must be an increasing [lo, hi] pair")
        if len(self.bucket_edges) < 2 or any(
            not math.isfinite(value) for value in self.bucket_edges
        ) or any(a >= b for a, b in zip(self.bucket_edges, self.bucket_edges[1:])):
            raise ValueError("bucket_edges must be finite and strictly increasing")

        expected = {"left_ankle_roll_Link", "right_ankle_roll_Link"}
        if self.asset_cfg.body_ids == slice(None):
            raise RuntimeError("effective ground friction must be scoped to the two A3 foot bodies")
        resolved = {self.asset.body_names[int(index)] for index in self.asset_cfg.body_ids}
        if resolved != expected:
            raise RuntimeError(
                "effective ground friction scope mismatch: "
                f"resolved={sorted(resolved)}, expected={sorted(expected)}"
            )

        # IsaacLab's material buffer is shape-indexed, not body-indexed.  Reuse the same robust
        # link-path parsing as the upstream material randomizer and fail closed on a mismatch.
        self.num_shapes_per_body = []
        for link_path in self.asset.root_physx_view.link_paths[0]:
            link_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)
            self.num_shapes_per_body.append(link_view.max_shapes)
        if sum(self.num_shapes_per_body) != self.asset.root_physx_view.max_shapes:
            raise ValueError("could not resolve A3 link shape counts for friction randomization")

        num_envs = int(env.scene.num_envs)
        self.mu_static = torch.full((num_envs,), self.static_nominal, device=self.asset.device)
        self.mu_dynamic = torch.full((num_envs,), self.dynamic_nominal, device=self.asset.device)
        self.mu_ratio = self.mu_dynamic / self.mu_static
        self.beta = torch.zeros_like(self.mu_static)
        self.static_low = torch.full_like(self.mu_static, self.static_nominal)
        self.static_high = torch.full_like(self.mu_static, self.static_nominal)
        self.dynamic_low = torch.full_like(self.mu_dynamic, self.dynamic_nominal)
        self.dynamic_high = torch.full_like(self.mu_dynamic, self.dynamic_nominal)

    def _publish_metrics(self, env) -> None:
        try:
            command = env.command_manager.get_term("racket_target")
        except (AttributeError, KeyError, ValueError):
            return
        command.metrics["friction_mu_static"] = self.mu_static.clone()
        command.metrics["friction_mu_dynamic"] = self.mu_dynamic.clone()
        command.metrics["friction_mu_ratio"] = self.mu_ratio.clone()
        command.metrics["friction_beta"] = self.beta.clone()
        command.metrics["friction_static_low"] = self.static_low.clone()
        command.metrics["friction_static_high"] = self.static_high.clone()
        command.metrics["friction_dynamic_low"] = self.dynamic_low.clone()
        command.metrics["friction_dynamic_high"] = self.dynamic_high.clone()
        command.metrics["friction_curriculum_iteration"] = torch.full_like(
            self.mu_static, float(getattr(env, "_hope_stance_curriculum_iteration", 0))
        )
        edges = torch.as_tensor(self.bucket_edges, device=self.asset.device, dtype=self.mu_dynamic.dtype)
        static_bucket = torch.bucketize(self.mu_static, edges[1:-1]).to(dtype=torch.long)
        dynamic_bucket = torch.bucketize(self.mu_dynamic, edges[1:-1]).to(dtype=torch.long)
        command.metrics["friction_bucket_static_index"] = static_bucket.float()
        command.metrics["friction_bucket_dynamic_index"] = dynamic_bucket.float()
        for index, (low, high) in enumerate(zip(self.bucket_edges, self.bucket_edges[1:])):
            label = f"friction_bucket_{low:g}_{high:g}".replace(".", "p")
            command.metrics[f"{label}_static"] = (static_bucket == index).float()
            command.metrics[f"{label}_dynamic"] = (dynamic_bucket == index).float()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        static_nominal: float,
        dynamic_nominal: float,
        competition_static_min: float,
        competition_static_max: float,
        competition_ratio_min: float,
        competition_ratio_max: float,
        stress_probability: float,
        stress_static_min: float,
        stress_static_max: float,
        stress_dynamic_min: float,
        stress_dynamic_max: float,
        curriculum_start_iteration: int = 2100,
        curriculum_end_iteration: int = 2700,
        restitution_range: tuple[float, float] = (0.0, 0.0),
        bucket_edges: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5),
    ):
        del (
            asset_cfg,
            static_nominal,
            dynamic_nominal,
            competition_static_min,
            competition_static_max,
            competition_ratio_min,
            competition_ratio_max,
            stress_probability,
            stress_static_min,
            stress_static_max,
            stress_dynamic_min,
            stress_dynamic_max,
            curriculum_start_iteration,
            curriculum_end_iteration,
            restitution_range,
            bucket_edges,
        )
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.asset.device, dtype=torch.long)
        cpu_ids = env_ids.to(device="cpu")

        iteration = int(getattr(env, "_hope_stance_curriculum_iteration", 0))
        beta = smoothstep_stance_alpha(
            iteration,
            ramp_start_iteration=self.curriculum_start_iteration,
            ramp_end_iteration=self.curriculum_end_iteration,
        )
        beta_tensor = torch.full((len(env_ids),), beta, device=self.asset.device)
        beta_cpu = beta_tensor.to(device="cpu")
        # Expand the competition neighborhood continuously from the fixed nominal pair.
        nominal_ratio = self.dynamic_nominal / self.static_nominal
        static_low_cpu = self.static_nominal + beta_cpu * (
            self.competition_static_min - self.static_nominal
        )
        static_high_cpu = self.static_nominal + beta_cpu * (
            self.competition_static_max - self.static_nominal
        )
        ratio_low_cpu = nominal_ratio + beta_cpu * (
            self.competition_ratio_min - nominal_ratio
        )
        ratio_high_cpu = nominal_ratio + beta_cpu * (
            self.competition_ratio_max - nominal_ratio
        )
        sampled_static_cpu = static_low_cpu + torch.rand(len(cpu_ids), device="cpu") * (
            static_high_cpu - static_low_cpu
        )
        sampled_ratio_cpu = ratio_low_cpu + torch.rand(len(cpu_ids), device="cpu") * (
            ratio_high_cpu - ratio_low_cpu
        )
        sampled_dynamic_cpu = sampled_static_cpu * sampled_ratio_cpu

        # The stress mixture is introduced only after beta reaches one.  Dynamic friction is
        # clipped to the sampled static value so the Coulomb ordering is always respected.
        stress_mask = (beta >= 1.0 - 1e-6) & (
            torch.rand(len(cpu_ids), device="cpu") < self.stress_probability
        )
        if bool(torch.any(stress_mask).item()):
            stress_static_cpu = self.stress_static_min + torch.rand(
                len(cpu_ids), device="cpu"
            ) * (self.stress_static_max - self.stress_static_min)
            stress_dynamic_cpu = self.stress_dynamic_min + torch.rand(
                len(cpu_ids), device="cpu"
            ) * (self.stress_dynamic_max - self.stress_dynamic_min)
            stress_dynamic_cpu = torch.minimum(stress_dynamic_cpu, stress_static_cpu)
            sampled_static_cpu = torch.where(stress_mask, stress_static_cpu, sampled_static_cpu)
            sampled_dynamic_cpu = torch.where(stress_mask, stress_dynamic_cpu, sampled_dynamic_cpu)
        sampled_ratio_cpu = sampled_dynamic_cpu / torch.clamp_min(sampled_static_cpu, 1e-6)

        static_low_cpu = torch.where(
            stress_mask,
            torch.full_like(static_low_cpu, self.stress_static_min),
            static_low_cpu,
        )
        static_high_cpu = torch.where(
            stress_mask,
            torch.full_like(static_high_cpu, self.stress_static_max),
            static_high_cpu,
        )
        dynamic_low_cpu = torch.where(
            stress_mask,
            torch.full_like(static_low_cpu, self.stress_dynamic_min),
            static_low_cpu * ratio_low_cpu,
        )
        dynamic_high_cpu = torch.where(
            stress_mask,
            torch.full_like(static_high_cpu, self.stress_dynamic_max),
            static_high_cpu * ratio_high_cpu,
        )
        rest_lo, rest_hi = self.restitution_range
        restitution_cpu = rest_lo + torch.rand(len(cpu_ids), device="cpu") * (rest_hi - rest_lo)

        materials = self.asset.root_physx_view.get_material_properties()
        for body_id in self.asset_cfg.body_ids:
            start = sum(self.num_shapes_per_body[:int(body_id)])
            end = start + self.num_shapes_per_body[int(body_id)]
            materials[cpu_ids, start:end, 0] = sampled_static_cpu.unsqueeze(-1)
            materials[cpu_ids, start:end, 1] = sampled_dynamic_cpu.unsqueeze(-1)
            materials[cpu_ids, start:end, 2] = restitution_cpu.unsqueeze(-1)
        self.asset.root_physx_view.set_material_properties(materials, cpu_ids)

        self.mu_static[env_ids] = sampled_static_cpu.to(device=self.asset.device)
        self.mu_dynamic[env_ids] = sampled_dynamic_cpu.to(device=self.asset.device)
        self.mu_ratio[env_ids] = sampled_ratio_cpu.to(device=self.asset.device)
        self.beta[env_ids] = beta_tensor
        self.static_low[env_ids] = static_low_cpu.to(device=self.asset.device)
        self.static_high[env_ids] = static_high_cpu.to(device=self.asset.device)
        self.dynamic_low[env_ids] = dynamic_low_cpu.to(device=self.asset.device)
        self.dynamic_high[env_ids] = dynamic_high_cpu.to(device=self.asset.device)
        self._publish_metrics(env)


class randomize_a3_message_pd_gains(ManagerTermBase):
    """HKUST-style A3 gain DR with separately randomized passive damping.

    The A3 Isaac drive stores ``Kd_message + damping_passive``.  Consequently the
    randomized drive must be ``beta * Kd_message + damping_passive`` rather than
    ``beta * default_joint_damping``.  A deterministic env-id cohort stays exactly
    nominal; robust environments draw per-joint log-uniform multipliers once at
    startup.  The affine action scale is intentionally untouched.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.alpha_range = randomize_structured_pd_gains._validate_range(
            cfg.params["alpha_range"], "alpha_range"
        )
        self.beta_range = randomize_structured_pd_gains._validate_range(
            cfg.params["beta_range"], "beta_range"
        )
        self.passive_damping_range = randomize_structured_pd_gains._validate_range(
            cfg.params.get("passive_damping_range", (1.0, 1.0)),
            "passive_damping_range",
        )
        self.nominal_fraction = float(cfg.params["nominal_fraction"])
        if not (
            math.isfinite(self.nominal_fraction)
            and 0.0 <= self.nominal_fraction <= 1.0
        ):
            raise ValueError("nominal_fraction must be finite and in [0, 1]")

        from whole_body_tracking.robots.agibot_a3 import (
            a3_deploy_joint_kd,
            a3_passive_joint_damping,
        )

        message_kd = a3_deploy_joint_kd(list(self.asset.joint_names))
        if message_kd is None:
            raise RuntimeError(
                "A3 message/passive PD randomization was attached to a non-A3 asset"
            )
        passive = a3_passive_joint_damping(list(self.asset.joint_names))
        self.message_kd = torch.tensor(
            message_kd, dtype=torch.float, device=self.asset.device
        )
        self.passive_damping = torch.tensor(
            passive, dtype=torch.float, device=self.asset.device
        )
        expected_total = self.message_kd + self.passive_damping
        nominal_total = self.asset.data.default_joint_damping[0]
        if not bool(
            torch.allclose(
                nominal_total,
                expected_total,
                rtol=0.0,
                atol=1.0e-6,
            )
        ):
            raise RuntimeError(
                "A3 damping contract drift: Isaac default must equal "
                "deploy message Kd plus passive damping"
            )

        num_envs = int(env.scene.num_envs)
        num_joints = len(self.asset.joint_names)
        self.alpha = torch.ones(
            num_envs, num_joints, device=self.asset.device
        )
        self.beta = torch.ones_like(self.alpha)
        self.passive_multiplier = torch.ones_like(self.alpha)
        env_ids = torch.arange(
            num_envs, dtype=torch.long, device=self.asset.device
        )
        self.nominal_mask = self._select_nominal(env_ids)
        # CommandTerm.reset() zeros command metrics, while this startup event does not run again.
        # Keep one authoritative, read-only telemetry view on the articulation so the racket
        # command can republish the fixed cohort and gains every policy step.
        self.asset._hope_a3_pd_telemetry = {
            "nominal_mask": self.nominal_mask,
            "alpha": self.alpha,
            "beta": self.beta,
            "passive_multiplier": self.passive_multiplier,
        }

    def _select_nominal(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self.nominal_fraction <= 0.0:
            return torch.zeros_like(env_ids, dtype=torch.bool)
        if self.nominal_fraction >= 1.0:
            return torch.ones_like(env_ids, dtype=torch.bool)
        reciprocal = 1.0 / self.nominal_fraction
        stride = int(round(reciprocal))
        if not math.isclose(
            reciprocal, float(stride), rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise RuntimeError(
                "A3 nominal fraction must be the reciprocal of an integer"
            )
        return torch.remainder(env_ids, stride) == 0

    @staticmethod
    def _sample_log_uniform(
        shape: tuple[int, ...], value_range: tuple[float, float], device
    ) -> torch.Tensor:
        lo, hi = value_range
        return torch.empty(shape, device=device).uniform_(
            math.log(lo), math.log(hi)
        ).exp_()

    def _publish_metrics(self, env: ManagerBasedEnv) -> None:
        command_manager = getattr(env, "command_manager", None)
        if command_manager is None:
            return
        try:
            command = command_manager.get_term("racket_target")
        except (AttributeError, KeyError, ValueError):
            return
        values = {
            "pd_nominal_env": self.nominal_mask.float(),
            "pd_kp_multiplier_mean": self.alpha.mean(dim=-1),
            "pd_kd_message_multiplier_mean": self.beta.mean(dim=-1),
            "pd_passive_damping_multiplier_mean": self.passive_multiplier.mean(dim=-1),
        }
        for name, value in values.items():
            existing = command.metrics.get(name)
            if torch.is_tensor(existing) and existing.shape == value.shape:
                existing.copy_(value)
            else:
                command.metrics[name] = value.clone()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        alpha_range: tuple[float, float],
        beta_range: tuple[float, float],
        nominal_fraction: float,
        passive_damping_range: tuple[float, float] = (1.0, 1.0),
    ):
        del asset_cfg, alpha_range, beta_range, nominal_fraction, passive_damping_range
        if env_ids is None:
            env_ids = torch.arange(
                env.scene.num_envs,
                dtype=torch.long,
                device=self.asset.device,
            )
        else:
            env_ids = env_ids.to(device=self.asset.device, dtype=torch.long)

        shape = (env_ids.numel(), len(self.asset.joint_names))
        alpha = self._sample_log_uniform(
            shape, self.alpha_range, self.asset.device
        )
        beta = self._sample_log_uniform(
            shape, self.beta_range, self.asset.device
        )
        passive_multiplier = self._sample_log_uniform(
            shape, self.passive_damping_range, self.asset.device
        )
        nominal = self._select_nominal(env_ids)
        alpha[nominal] = 1.0
        beta[nominal] = 1.0
        passive_multiplier[nominal] = 1.0
        self.alpha[env_ids] = alpha
        self.beta[env_ids] = beta
        self.passive_multiplier[env_ids] = passive_multiplier
        self.nominal_mask[env_ids] = nominal

        stiffness = self.asset.data.default_joint_stiffness[env_ids] * alpha
        damping = (
            self.message_kd.unsqueeze(0) * beta
            + self.passive_damping.unsqueeze(0) * passive_multiplier
        )
        self.asset.write_joint_stiffness_to_sim(stiffness, env_ids=env_ids)
        self.asset.write_joint_damping_to_sim(damping, env_ids=env_ids)
        for actuator in self.asset.actuators.values():
            if not isinstance(actuator, ImplicitActuator):
                raise RuntimeError(
                    "A3 message/passive PD contract requires implicit actuators"
                )
            joint_ids = actuator.joint_indices
            actuator.stiffness[env_ids] = stiffness[:, joint_ids]
            actuator.damping[env_ids] = damping[:, joint_ids]
        self._publish_metrics(env)


class randomize_a3_torque_capacity(ManagerTermBase):
    """Startup-randomize A3 actuator torque capacity with a fixed nominal cohort.

    The sampled limit is written to PhysX and to the q_des projector's cached effort-limit
    contract, so torque clipping, torque headroom reward, and feasible-q_des projection all use
    the same per-environment capacity.  Sampling is startup-only: each environment represents a
    fixed motor/voltage/thermal cohort for the whole run.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.capacity_range = randomize_structured_pd_gains._validate_range(
            cfg.params.get("capacity_range", (1.0, 1.0)), "capacity_range"
        )
        self.nominal_fraction = float(cfg.params.get("nominal_fraction", 0.0))
        if not math.isfinite(self.nominal_fraction) or not 0.0 <= self.nominal_fraction <= 1.0:
            raise ValueError("torque-capacity nominal_fraction must be finite and in [0, 1]")
        self.nominal_limits = self.asset.data.joint_effort_limits[0].detach().clone()
        if bool(torch.any(self.nominal_limits <= 0.0)):
            raise RuntimeError("A3 torque-capacity randomization requires positive effort limits")
        num_envs = int(env.scene.num_envs)
        num_joints = len(self.asset.joint_names)
        self.multiplier = torch.ones(num_envs, num_joints, device=self.asset.device)
        env_ids = torch.arange(num_envs, dtype=torch.long, device=self.asset.device)
        self.nominal_mask = self._select_nominal(env_ids)
        self.asset._hope_a3_torque_capacity_telemetry = {
            "nominal_mask": self.nominal_mask,
            "multiplier": self.multiplier,
            "nominal_limits": self.nominal_limits,
        }

    def _select_nominal(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self.nominal_fraction <= 0.0:
            return torch.zeros_like(env_ids, dtype=torch.bool)
        if self.nominal_fraction >= 1.0:
            return torch.ones_like(env_ids, dtype=torch.bool)
        reciprocal = 1.0 / self.nominal_fraction
        stride = int(round(reciprocal))
        if not math.isclose(reciprocal, float(stride), rel_tol=0.0, abs_tol=1.0e-9):
            raise RuntimeError("A3 torque-capacity nominal fraction must be the reciprocal of an integer")
        return torch.remainder(env_ids, stride) == 0

    @staticmethod
    def _sample_log_uniform(shape: tuple[int, ...], value_range: tuple[float, float], device) -> torch.Tensor:
        lo, hi = value_range
        return torch.empty(shape, device=device).uniform_(math.log(lo), math.log(hi)).exp_()

    def _publish_metrics(self, env: ManagerBasedEnv) -> None:
        try:
            command = env.command_manager.get_term("racket_target")
        except (AttributeError, KeyError, ValueError):
            return
        values = {
            "motor_capacity_nominal_env": self.nominal_mask.float(),
            "motor_capacity_multiplier_mean": self.multiplier.mean(dim=-1),
            "motor_capacity_multiplier_min": self.multiplier.min(dim=-1).values,
            "motor_capacity_multiplier_max": self.multiplier.max(dim=-1).values,
        }
        for name, value in values.items():
            existing = command.metrics.get(name)
            if torch.is_tensor(existing) and existing.shape == value.shape:
                existing.copy_(value)
            else:
                command.metrics[name] = value.clone()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        capacity_range: tuple[float, float],
        nominal_fraction: float,
    ):
        del asset_cfg, capacity_range, nominal_fraction
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, dtype=torch.long, device=self.asset.device)
        else:
            env_ids = env_ids.to(device=self.asset.device, dtype=torch.long)
        shape = (env_ids.numel(), len(self.asset.joint_names))
        multiplier = self._sample_log_uniform(shape, self.capacity_range, self.asset.device)
        nominal = self._select_nominal(env_ids)
        multiplier[nominal] = 1.0
        limits = self.nominal_limits.unsqueeze(0) * multiplier
        writer = getattr(self.asset, "write_joint_effort_limit_to_sim", None)
        if not callable(writer):
            raise RuntimeError("Isaac articulation does not expose write_joint_effort_limit_to_sim")
        writer(limits, env_ids=env_ids)
        for actuator in self.asset.actuators.values():
            joint_ids = actuator.joint_indices
            if hasattr(actuator, "effort_limit"):
                actuator.effort_limit[env_ids] = limits[:, joint_ids]

        self.multiplier[env_ids] = multiplier
        self.nominal_mask[env_ids] = nominal
        # The q_des action term caches limits at construction; keep its feasibility projector
        # synchronized with the same sampled actuator capacity used by PhysX.
        try:
            action_term = env.action_manager.get_term("joint_pos")
            action_joint_ids = getattr(action_term, "_action_joint_ids", None)
            cached_limits = getattr(action_term, "_effort_limit", None)
            if torch.is_tensor(action_joint_ids) and torch.is_tensor(cached_limits):
                cached_limits[env_ids] = limits.index_select(1, action_joint_ids)
        except (AttributeError, KeyError, ValueError):
            pass
        self._publish_metrics(env)


def push_a3_competence_gated(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    force_magnitude_range: tuple[float, float] = (20.0, 60.0),
    torque_magnitude_range: tuple[float, float] = (5.0, 10.0),
    ramp_start_delay_steps: int = 4000,
    ramp_steps: int = 8000,
    skip_hold: bool = True,
):
    """Apply a delayed, ramped one-step 20--60 N / 5--10 Nm wrench after competence is learned.

    The event is called in ``interval`` mode.  IsaacLab's public
    ``set_external_force_and_torque`` API stores a persistent wrench, so it is not suitable here:
    using it would leave the push active until the next event instead of applying one impulse-like
    physics-step disturbance.  We therefore call the underlying articulation PhysX view directly;
    ``apply_forces_and_torques_at_position`` applies the wrench for the next physics step only.

    The event starts after a short post-gate delay and then ramps independently of the
    sensor-corruption curriculum.  This keeps recovery perturbations from arriving at the same
    time as the first mocap/target corruption.
    """
    if int(ramp_start_delay_steps) < 0:
        raise ValueError("ramp_start_delay_steps must be non-negative")
    if int(ramp_steps) <= 0:
        raise ValueError("ramp_steps must be positive")
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    unlocked = bool(getattr(command, "_ability_unlocked", True))
    if not unlocked:
        return
    unlock_step = int(getattr(command, "_ability_unlock_step", -1))
    current_step = int(getattr(env, "common_step_counter", 0))
    if unlock_step < 0:
        scale = 1.0
    else:
        ramp_elapsed = current_step - unlock_step - int(ramp_start_delay_steps)
        scale = min(max(ramp_elapsed / int(ramp_steps), 0.0), 1.0)
    if scale <= 0.0:
        return
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, dtype=torch.long, device=asset.device)
    else:
        env_ids = env_ids.to(device=asset.device, dtype=torch.long)
    if skip_hold:
        hold = getattr(command, "in_hold", None)
        if torch.is_tensor(hold):
            env_ids = env_ids[~hold[env_ids].bool()]
    if env_ids.numel() == 0:
        return
    force_lo, force_hi = (float(value) for value in force_magnitude_range)
    torque_lo, torque_hi = (float(value) for value in torque_magnitude_range)
    if force_lo < 0.0 or force_hi < force_lo or torque_lo < 0.0 or torque_hi < torque_lo:
        raise ValueError("external push magnitude ranges must be ordered and non-negative")
    try:
        body_ids = asset.find_bodies("pelvis_link", preserve_order=True)[0]
    except Exception:
        body_ids = [0]
    body_ids = torch.as_tensor(body_ids, dtype=torch.long, device=asset.device)
    force_direction = torch.randn(env_ids.numel(), 1, 3, device=asset.device)
    force_direction = force_direction / torch.linalg.norm(force_direction, dim=-1, keepdim=True).clamp_min(1.0e-6)
    torque_direction = torch.randn(env_ids.numel(), 1, 3, device=asset.device)
    torque_direction = torque_direction / torch.linalg.norm(torque_direction, dim=-1, keepdim=True).clamp_min(1.0e-6)
    force_magnitude = torch.empty(env_ids.numel(), 1, 1, device=asset.device).uniform_(force_lo, force_hi) * scale
    torque_magnitude = torch.empty(env_ids.numel(), 1, 1, device=asset.device).uniform_(torque_lo, torque_hi) * scale
    # ArticulationView expects one force/torque slot for every link in each selected
    # articulation.  Fill only the pelvis slot(s); zero entries leave the other links
    # untouched.  Applying directly through the PhysX view is transient for the next
    # physics step, unlike Articulation.set_external_force_and_torque(), which buffers
    # a persistent wrench.
    forces = torch.zeros(
        (env_ids.numel(), asset.num_bodies, 3), device=asset.device, dtype=force_direction.dtype
    )
    torques = torch.zeros_like(forces)
    forces[:, body_ids, :] = force_direction * force_magnitude
    torques[:, body_ids, :] = torque_direction * torque_magnitude
    asset.root_physx_view.apply_forces_and_torques_at_position(
        force_data=forces,
        torque_data=torques,
        position_data=None,
        indices=env_ids,
        is_global=True,
    )
    try:
        command.metrics["external_push_scale"][env_ids] = float(scale)
        command.metrics["external_push_count"][env_ids] += 1.0
    except (AttributeError, KeyError):
        pass


def assert_material_randomization_scope(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    expected_body_names: tuple[str, ...],
):
    """Fail closed if the material event resolves beyond the two foot bodies."""

    del env_ids
    asset: Articulation = env.scene[asset_cfg.name]
    if isinstance(asset_cfg.body_ids, slice):
        resolved = tuple(asset.body_names[asset_cfg.body_ids])
    else:
        resolved = tuple(asset.body_names[int(index)] for index in asset_cfg.body_ids)
    expected = tuple(expected_body_names)
    if set(resolved) != set(expected) or len(resolved) != len(expected):
        raise RuntimeError(
            "RallyV17 foot material scope mismatch: "
            f"resolved={resolved}, expected={expected}. Refusing to randomize "
            "wrist/racket or other robot collision shapes."
        )


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)
