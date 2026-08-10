"""HOPE-specific command term: racket / base target tracking on top of BeyondMimic.

This adds the HITTER (arXiv:2508.21043) racket-target objective to the BeyondMimic motion
tracker. The base ``MotionCommand`` (in ``commands.py``) drives the imitation reward and owns the
per-env motion clock (``time_steps``). ``RacketTargetCommand`` rides on top of it:

* it samples a *desired* racket state (position, velocity, face normal) and a *desired* base XY
  position each swing — exactly the quantities the model-based planner emits at deploy time via
  ``msgs/RacketCommand`` (position, velocity, normal). No ball is needed in simulation;
  training samples targets, the planner supplies them at runtime.
* it computes the *actual* racket state in simulation by forward kinematics through the fixed
  racket mount ``T_mount`` (wrist -> paddle center), so the reward can compare actual vs desired.
* it derives the strike timing from the motion clip phase, exposing ``time_to_strike`` plus the
  ``pre_strike`` / ``strike_window`` masks that gate the goal rewards.

Per the HOPE racket-tracking prohibition, there is NO measured racket feedback at deploy time:
``r_racket`` is a simulation-only signal; on hardware the policy runs open-loop on racket pose.

HITTER alignment notes (see the project HITTER verification):
* racket *position* is observed relative to the base; racket *velocity* is observed in world.
* HOPE currently also observes desired racket *normal* in the actor so the policy can respond to
  normal targets; actual racket normal remains a privileged simulation-only critic/reward signal.
* swing type is a *sampled* variable used here to (a) flag forehand/backhand and (b) select the
  reference clip; it is not required in the actor observation when separate forehand/backhand
  policies are trained (the HOPE default).
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_mul,
    quat_rotate_inverse,
    sample_uniform,
    yaw_quat,
)

from training.tasks.tracking.mdp.commands import MotionCommand, MotionLibraryLoader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class EpisodeStrikeEvent:
    """Latched, per-environment V1.3B strike contract.

    This structure is intentionally private to the environment.  The public
    policy receives only its canonical 10-D goal, never a motion id, a frame
    index, an event phase, or any other teacher feature.
    """

    event_id: torch.Tensor
    motion_id: torch.Tensor
    teacher_start_frame: torch.Tensor
    teacher_hit_frame: torch.Tensor
    episode_strike_time_s: torch.Tensor
    teacher_physical_strike_time_s: torch.Tensor
    teacher_position_b: torch.Tensor
    teacher_velocity_b: torch.Tensor
    teacher_normal_b: torch.Tensor
    sampled_position_b: torch.Tensor
    sampled_velocity_b: torch.Tensor
    sampled_normal_b: torch.Tensor
    sampled_timing_offset_s: torch.Tensor
    strike_armed: torch.Tensor
    strike_consumed: torch.Tensor
    goal_sample_count: torch.Tensor
    goal_resample_count_after_reset: torch.Tensor
    strike_event_count: torch.Tensor
    upper_prior_wrap_count: torch.Tensor


class RacketTargetCommand(CommandTerm):
    """Samples desired racket/base targets and computes the actual racket state by FK."""

    cfg: RacketTargetCommandCfg

    def __init__(self, cfg: RacketTargetCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        if cfg.adapter_external_paired and self.num_envs % 7 != 0:
            raise ValueError(
                "adapter_external_paired requires num_envs divisible by 7 "
                "for complete [0,+x,-x,+y,-y,+z,-z] groups"
            )

        self.robot: Articulation = env.scene[cfg.asset_name]

        # Resolve the racket FK source: prefer the racket body if it survived the physics import,
        # otherwise fall back to (wrist body pose) * (constant mount offset).
        # NOTE: Articulation.find_bodies() RAISES (resolve_matching_names) when a name matches no
        # body — it does not return []. So gate on body_names membership before calling it, or the
        # wrist-offset fallback below becomes unreachable.
        requested_fk_mode = str(getattr(cfg, "racket_fk_mode", "auto"))
        if requested_fk_mode not in ("auto", "body", "wrist_offset"):
            raise ValueError(f"unsupported racket_fk_mode={requested_fk_mode!r}")
        if requested_fk_mode == "body" and cfg.racket_body_name not in self.robot.body_names:
            raise RuntimeError(
                f"V1.3B requested body FK but body {cfg.racket_body_name!r} is absent; refusing wrist fallback"
            )
        if requested_fk_mode == "wrist_offset":
            if cfg.wrist_body_name not in self.robot.body_names:
                raise RuntimeError(
                    f"V1.3B requested wrist-offset FK but wrist body {cfg.wrist_body_name!r} is absent"
                )
            self._racket_mode = "wrist_offset"
            self._racket_body_index = -1
            self._wrist_body_index = self.robot.find_bodies(cfg.wrist_body_name, preserve_order=True)[0][0]
        elif cfg.racket_body_name in self.robot.body_names:
            self._racket_mode = "body"
            self._racket_body_index = self.robot.find_bodies(cfg.racket_body_name, preserve_order=True)[0][0]
            self._wrist_body_index = -1
        else:
            self._racket_mode = "wrist_offset"
            self._racket_body_index = -1
            assert cfg.wrist_body_name in self.robot.body_names, (
                f"RacketTargetCommand: neither racket body '{cfg.racket_body_name}' nor wrist body "
                f"'{cfg.wrist_body_name}' found on asset '{cfg.asset_name}'."
            )
            self._wrist_body_index = self.robot.find_bodies(cfg.wrist_body_name, preserve_order=True)[0][0]

        self._mount_offset = torch.tensor(cfg.mount_offset, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )
        # Fixed wrist->racket rotation (used only in wrist_offset fallback mode so the face normal is
        # taken in the racket frame, not the bare wrist frame). Identity for the A3 mount (all mount
        # joints are rpy=0); set non-identity if your mount tilts the paddle relative to the wrist.
        self._mount_quat = torch.tensor(cfg.mount_quat, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )
        print(
            f"[RacketTargetCommand] racket FK mode={self._racket_mode} "
            f"racket_body='{cfg.racket_body_name}' racket_index={self._racket_body_index} "
            f"wrist_body='{cfg.wrist_body_name}' wrist_index={self._wrist_body_index} "
            f"mount_offset={tuple(float(x) for x in cfg.mount_offset)}",
            flush=True,
        )

        # The motion command (resolved lazily on first update; not guaranteed to exist at __init__).
        self._motion_term: MotionCommand | None = None
        # Per-env motion phase at the last target resample; used to detect clip wraps (new swings).
        # Stamped on every resample so a reset-time resample is not double-triggered next step.
        self._prev_motion_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Desired (sampled) targets, world frame.
        self.racket_target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_target_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_target_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_target_normal_w[:, 2] = 1.0
        # Immutable nominal strike target used by the frozen anchor policy.
        # Runtime external position latches update only ``racket_target_*``;
        # this buffer remains the manifest/anchor contract seen by model_900.
        self.racket_anchor_target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_anchor_target_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_anchor_target_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_anchor_target_normal_w[:, 2] = 1.0
        self.base_target_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.swing_sign = torch.ones(self.num_envs, device=self.device)
        # Runtime planner/audit override.  The position is expressed in the
        # base yaw-heading frame at command receipt, then converted once to a
        # fixed world point.  It therefore does not follow root motion during
        # the swing.  The base-frame copy is retained so a reset/clip resample
        # can rebuild the same command instead of silently restoring the
        # manifest anchor.
        self._external_target_position_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._external_target_position_b = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        # Optional immutable command-receipt-frame copy.  Floating-base target
        # adapters must not reinterpret a fixed world target as the base yaws
        # during the swing.
        self._external_target_delta_receipt_b = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        # P0 paired-incremental curriculum bookkeeping.  A group is ordered
        # ``0, +x, -x, +y, -y, +z, -z`` and shares its first environment as
        # the physical zero-offset baseline.
        self.adapter_pair_baseline_env = torch.arange(self.num_envs, device=self.device)
        self.adapter_pair_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Actual racket state, world frame (from FK).
        self.racket_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.racket_quat_w[:, 0] = 1.0
        self.racket_lin_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_normal_w[:, 2] = 1.0

        # Reference racket state at the strike frame (CONSTANT per clip): pos (env-origin relative),
        # world linear velocity, and face normal, computed by the SAME FK as the actual racket
        # (_compute_racket_state) but fed the reference MOTION's body poses. Used by the
        # "reference_perturbed" target mode so a sampled target is one the imitated swing can actually
        # reach (a perfect imitator hits it exactly). Cached lazily on first resample, after the motion
        # term is resolved and its motion_file is loaded.
        self._ref_strike_cached = False
        self._ref_racket_pos_rel = torch.zeros(3, device=self.device)
        self._ref_racket_vel_w = torch.zeros(3, device=self.device)
        self._ref_racket_normal_w = torch.zeros(3, device=self.device)
        # Reference base (root) XY at the strike + the base->racket horizontal offset. Used to COUPLE
        # base_target to racket_target so standing at base_target keeps the racket reachable.
        self._ref_base_pos_rel = torch.zeros(3, device=self.device)
        self._ref_reach_offset_xy = torch.zeros(2, device=self.device)

        # Success-gated curriculum: running perturbation scale, advanced only when the smoothed
        # exact-strike composite success clears the threshold (see _perturb_scale / _update_metrics).
        self._curr_perturb_scale = float(cfg.ref_perturb_curriculum_start)

        # Decayed accumulators for the CONDITIONAL exact-strike pass rates (see _update_metrics). The
        # logged strike_*_pass_exact / strike_composite_success_exact report the fraction of *exact-strike
        # samples* that clear each acceptance threshold — NOT a per-env value held over the long
        # non-strike portion of every episode (which diluted the old metric ~10x at reset-logging time).
        # Sample-weighted EMA: acc = decay*acc + this-step-count; rate = pass_acc / sample_acc.
        self._exact_n_acc = 0.0
        self._exact_pass_comp_acc = 0.0
        self._exact_pass_pos_acc = 0.0
        self._exact_pass_vel_acc = 0.0
        self._exact_pass_normal_acc = 0.0
        self._exact_composite_rate = 0.0

        # Strike timing / gating.
        self.time_to_strike = torch.zeros(self.num_envs, device=self.device)
        self.pre_strike = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.strike_window = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Episode-wide tracking errors (instantaneous; averaged over terminating envs at reset).
        self.metrics["racket_pos_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_vel_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_normal_error_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_normal_reward_raw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_normal_reward_temporal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_normal_reward_std_rad"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_hit_coupled_reward_raw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_hit_coupled_pos_raw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_hit_coupled_vel_raw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_hit_coupled_normal_raw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_velocity_position_gated_reward_raw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_velocity_position_gated_velocity_raw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_velocity_position_gated_position_gate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_pos_error"] = torch.zeros(self.num_envs, device=self.device)
        # Strike-window metrics: hold the value from the MOST RECENT strike — these map directly to
        # the acceptance criteria (racket pos < 7.5 cm, vel < 0.5 m/s, normal < 15 deg AT strike) and
        # are the real "is the policy learning to hit" signal (the episode-wide ones above are diluted
        # by the long non-strike portion of each swing).
        self.metrics["racket_pos_error_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_vel_error_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_normal_error_deg_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_success"] = torch.zeros(self.num_envs, device=self.device)
        # Exact-strike metrics: sampled only on the nearest control frame to the configured strike
        # step. These avoid the "within-window" dilution from the +/- strike reward window.
        self.metrics["racket_pos_error_exact_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_vel_error_exact_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_normal_error_deg_exact_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_composite_success_exact"] = torch.zeros(self.num_envs, device=self.device)
        # Per-axis position error AT the exact strike frame (which axis is the miss?).
        self.metrics["racket_pos_error_x_exact_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_pos_error_y_exact_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_pos_error_z_exact_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["exact_strike_hit_rate"] = torch.zeros(self.num_envs, device=self.device)
        # Conditional exact-strike pass rates (broadcast scalar; the trustworthy success signal). Each is
        # the fraction of exact-strike samples that clear that acceptance threshold, undiluted by the
        # non-strike steps that wrecked the old held metric. strike_composite_success_exact requires all
        # three (pos & vel & normal) and drives the success-gated perturbation curriculum.
        self.metrics["strike_pos_pass_exact"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_vel_pass_exact"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_normal_pass_exact"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["exact_strike_sample_count_decayed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_window_hit_rate"] = torch.zeros(self.num_envs, device=self.device)
        # Base-position error while the base target is active (pre-strike), held at its last value.
        self.metrics["base_pos_error_pre_strike"] = torch.zeros(self.num_envs, device=self.device)
        # Swing-quality detail at the most recent strike: actual paddle speed, per-axis position error,
        # and success at tighter/looser thresholds (5 cm / 10 cm) for a fuller accuracy distribution.
        self.metrics["racket_speed_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_target_speed_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_pos_error_x_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_pos_error_y_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["racket_pos_error_z_at_strike"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_success_5cm"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_success_10cm"] = torch.zeros(self.num_envs, device=self.device)
        # Robot-health diagnostics (episode-wide, instantaneous) — logged here because this term already
        # holds ``self.robot``. Useful for sim2real: standing height, peak joint speed, actuator effort.
        self.metrics["base_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_upright"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["joint_vel_abs_max"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["time_to_strike_s"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pre_strike_flag"] = torch.zeros(self.num_envs, device=self.device)
        # Curriculum perturbation scale (reference_perturbed mode): 0 at start ramping to 1; lets you
        # watch the reachable target ball widen in logs. Stays 0 in "uniform" mode.
        self.metrics["ref_perturb_scale"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["adapter_external_target_delta_norm"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["adapter_pair_active"] = torch.zeros(self.num_envs, device=self.device)
        self._has_jpos_limits = hasattr(self.robot.data, "soft_joint_pos_limits") or hasattr(
            self.robot.data, "joint_pos_limits"
        )
        if self._has_jpos_limits:
            self.metrics["joint_pos_near_limit_frac"] = torch.zeros(self.num_envs, device=self.device)
        self._has_torque = hasattr(self.robot.data, "applied_torque")
        if self._has_torque:
            self.metrics["joint_torque_abs_mean"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["joint_torque_abs_max"] = torch.zeros(self.num_envs, device=self.device)
        # Policy action magnitude (saturation check for sim2real).
        self.metrics["action_abs_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["action_abs_max"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["action_delta_abs_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["action_delta_abs_max"] = torch.zeros(self.num_envs, device=self.device)
        for axis in ("x", "y"):
            self.metrics[f"base_pos_{axis}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"base_pos_error_{axis}"] = torch.zeros(self.num_envs, device=self.device)
        for axis in ("x", "y", "z"):
            self.metrics[f"racket_pos_error_{axis}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"racket_vel_error_{axis}"] = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------ #
    # CommandTerm API
    # ------------------------------------------------------------------ #
    @property
    def command(self) -> torch.Tensor:
        """Raw desired-target vector (world frame): [pos(3), vel(3), normal(3), t_left(1), base_xy(2), swing(1)]."""
        return torch.cat(
            [
                self.racket_target_pos_w,
                self.racket_target_vel_w,
                self.racket_target_normal_w,
                self.time_to_strike.unsqueeze(-1),
                self.base_target_pos_w,
                self.swing_sign.unsqueeze(-1),
            ],
            dim=-1,
        )

    @property
    def base_pos_w(self) -> torch.Tensor:
        return self.robot.data.root_pos_w

    @property
    def base_quat_w(self) -> torch.Tensor:
        return self.robot.data.root_quat_w

    def _motion(self) -> MotionCommand:
        if self._motion_term is None:
            self._motion_term = self._env.command_manager.get_term(self.cfg.motion_command_name)
        return self._motion_term

    def _reference_body_state(
        self, motion, step: int, body_index: int, motion_id: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return full-articulation reference body state from the loaded NPZ.

        MotionCommand's public body-state properties expose only its tracking
        subset. Racket-target sampling needs the full articulation order because
        the racket body may be outside that subset, so this adapter validates
        the expected loader buffers and keeps the private-field dependency in
        one place.
        """
        required = ("_body_pos_w", "_body_quat_w", "_body_lin_vel_w", "_body_ang_vel_w")
        missing = [name for name in required if not hasattr(motion, name)]
        if missing:
            raise AttributeError(
                "RacketTargetCommand requires full body-state arrays in MotionLoader; "
                f"missing: {', '.join(missing)}"
            )
        if isinstance(motion, MotionLibraryLoader):
            if motion_id is None:
                raise ValueError("manifest reference state requires an explicit motion_id")
            return (
                motion._body_pos_w[motion_id, step, body_index],
                motion._body_quat_w[motion_id, step, body_index],
                motion._body_lin_vel_w[motion_id, step, body_index],
                motion._body_ang_vel_w[motion_id, step, body_index],
            )
        return (
            motion._body_pos_w[step, body_index],
            motion._body_quat_w[step, body_index],
            motion._body_lin_vel_w[step, body_index],
            motion._body_ang_vel_w[step, body_index],
        )

    def _ensure_reference_strike_state(self):
        """Cache the reference racket state at the strike frame (once, after the motion is loaded).

        Uses the SAME FK as :meth:`_compute_racket_state` (racket body, or wrist+mount offset) but
        reads the reference MOTION's body poses instead of the live robot, so that a robot perfectly
        tracking the clip would land its racket exactly on this state. The motion's raw body arrays
        are indexed in live-articulation order (``MotionLoader`` selects tracked bodies from them via
        the live ``body_indexes``), so the live ``_racket_body_index`` / ``_wrist_body_index`` index
        them directly. Positions are env-origin relative (as the motion stores them).
        """
        if self._ref_strike_cached:
            return
        motion = self._motion().motion  # MotionLoader
        total = max(int(motion.time_step_total), 1)
        # Manifest-backed motions carry their own calibrated hit frame.  Do
        # not fall back to a single global phase for target-conditioned
        # training; that would shift the target window for motions whose hit
        # timing differs from the legacy 0.46 convention.
        if isinstance(motion, MotionLibraryLoader):
            # A manifest contains independent clips with independent hit
            # frames.  Cache the strike state for every clip; using the first
            # environment's ID as a global time index is incorrect and can
            # index the motion dimension with a frame number.
            self._ref_racket_pos_rel_by_motion = []
            self._ref_racket_vel_w_by_motion = []
            self._ref_racket_normal_w_by_motion = []
            self._ref_base_pos_rel_by_motion = []
            self._ref_reach_offset_xy_by_motion = []
            for motion_id in range(motion.num_motions):
                strike_step = int(motion.hit_frame[motion_id].item())
                if self._racket_mode == "body":
                    idx = self._racket_body_index
                    pos, quat, lin, _ang = self._reference_body_state(motion, strike_step, idx, motion_id)
                else:
                    widx = self._wrist_body_index
                    wpos, wquat, wlin, wang = self._reference_body_state(motion, strike_step, widx, motion_id)
                    offset_w = quat_apply(wquat.unsqueeze(0), self._mount_offset[0:1]).squeeze(0)
                    pos = wpos + offset_w
                    lin = wlin + torch.cross(wang, offset_w, dim=-1)
                    quat = quat_mul(wquat.unsqueeze(0), self._mount_quat[0:1]).squeeze(0)
                normal = matrix_from_quat(quat.unsqueeze(0))[0, :, self.cfg.mount_normal_axis] * self.cfg.mount_normal_sign
                normal = normal / (torch.norm(normal) + 1e-6)
                base_pos, _base_quat, _base_lin, _base_ang = self._reference_body_state(motion, strike_step, 0, motion_id)
                reach = (pos[:2] - base_pos[:2]).detach().clone()
                self._ref_racket_pos_rel_by_motion.append(pos.detach().clone())
                self._ref_racket_vel_w_by_motion.append(lin.detach().clone())
                self._ref_racket_normal_w_by_motion.append(normal.detach().clone())
                self._ref_base_pos_rel_by_motion.append(base_pos.detach().clone())
                self._ref_reach_offset_xy_by_motion.append(reach)
            self._ref_racket_pos_rel_by_motion = torch.stack(self._ref_racket_pos_rel_by_motion)
            self._ref_racket_vel_w_by_motion = torch.stack(self._ref_racket_vel_w_by_motion)
            self._ref_racket_normal_w_by_motion = torch.stack(self._ref_racket_normal_w_by_motion)
            self._ref_base_pos_rel_by_motion = torch.stack(self._ref_base_pos_rel_by_motion)
            self._ref_reach_offset_xy_by_motion = torch.stack(self._ref_reach_offset_xy_by_motion)
            # Keep scalar aliases for compatibility with diagnostics that use
            # a single-motion command; multi-motion sampling indexes the bank
            # explicitly below.
            self._ref_racket_pos_rel = self._ref_racket_pos_rel_by_motion[0]
            self._ref_racket_vel_w = self._ref_racket_vel_w_by_motion[0]
            self._ref_racket_normal_w = self._ref_racket_normal_w_by_motion[0]
            self._ref_base_pos_rel = self._ref_base_pos_rel_by_motion[0]
            self._ref_reach_offset_xy = self._ref_reach_offset_xy_by_motion[0]
            self._ref_strike_cached = True
            return
        else:
            strike_step = round(self.cfg.strike_phase * (total - 1))
        if self._racket_mode == "body":
            idx = self._racket_body_index
            pos, quat, lin, _ang = self._reference_body_state(motion, strike_step, idx)
        else:
            widx = self._wrist_body_index
            wpos, wquat, wlin, wang = self._reference_body_state(motion, strike_step, widx)
            offset_w = quat_apply(wquat.unsqueeze(0), self._mount_offset[0:1]).squeeze(0)
            pos = wpos + offset_w
            lin = wlin + torch.cross(wang, offset_w, dim=-1)
            quat = quat_mul(wquat.unsqueeze(0), self._mount_quat[0:1]).squeeze(0)
        normal = matrix_from_quat(quat.unsqueeze(0))[0, :, self.cfg.mount_normal_axis] * self.cfg.mount_normal_sign
        self._ref_racket_pos_rel = pos.detach().clone()
        self._ref_racket_vel_w = lin.detach().clone()
        self._ref_racket_normal_w = (normal / (torch.norm(normal) + 1e-6)).detach().clone()
        # Reference base (root) XY at the strike — root is articulation body index 0 (same order the
        # motion arrays use). The base->racket horizontal offset couples base_target to racket_target.
        base_pos, _base_quat, _base_lin, _base_ang = self._reference_body_state(motion, strike_step, 0)
        self._ref_base_pos_rel = base_pos.detach().clone()
        self._ref_reach_offset_xy = (self._ref_racket_pos_rel[:2] - self._ref_base_pos_rel[:2]).detach().clone()
        self._ref_strike_cached = True
        p, v, nrm = self._ref_racket_pos_rel, self._ref_racket_vel_w, self._ref_racket_normal_w
        b, off = self._ref_base_pos_rel, self._ref_reach_offset_xy
        print(
            f"[RacketTargetCommand] reference_perturbed: strike frame {strike_step}/{total - 1} "
            f"(phase {self.cfg.strike_phase}); reference racket @ strike (env-origin rel): "
            f"pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
            f"vel=({v[0]:.3f},{v[1]:.3f},{v[2]:.3f}) |v|={float(torch.norm(v)):.2f} "
            f"normal=({nrm[0]:.3f},{nrm[1]:.3f},{nrm[2]:.3f}); "
            f"reference base XY=({b[0]:.3f},{b[1]:.3f}) base->racket offset XY=({off[0]:.3f},{off[1]:.3f})",
            flush=True,
        )

    def _perturb_scale(self) -> float:
        """Curriculum factor in [start, 1.0] that widens the reference perturbation over training.

        Success-gated mode (default): return the running ``_curr_perturb_scale``, which advances only
        when the policy demonstrates exact-strike success (see :meth:`_update_metrics`). Otherwise fall
        back to the legacy open-loop ramp keyed to ``env.common_step_counter``. The returned scale is
        clamped to ``[ref_perturb_curriculum_start, 1.0]``.
        """
        start = float(self.cfg.ref_perturb_curriculum_start)
        if self.cfg.target_mode in ("reference_perturbed", "manifest_perturbed") and self.cfg.ref_perturb_success_gated:
            scale = self._curr_perturb_scale
        else:
            steps = float(getattr(self._env, "common_step_counter", 0))
            c = float(self.cfg.ref_perturb_curriculum_steps)
            frac = 1.0 if c <= 0.0 else min(1.0, steps / c)
            scale = start + (1.0 - start) * frac
        return min(1.0, max(start, scale))

    def _sample_targets_uniform(self, env_ids: Sequence[int], origins: torch.Tensor, n: int):
        """Independent box sampling (legacy mode). Ranges are PLACEHOLDERS not tied to the swing."""
        pos = origins.clone()
        pos[:, 0] += sample_uniform(*self.cfg.racket_pos_x_range, (n,), self.device)
        pos[:, 1] += sample_uniform(*self.cfg.racket_pos_y_range, (n,), self.device)
        pos[:, 2] += sample_uniform(*self.cfg.racket_pos_z_range, (n,), self.device)
        self.racket_target_pos_w[env_ids] = pos

        vel = torch.empty(n, 3, device=self.device)
        vel[:, 0] = sample_uniform(*self.cfg.racket_vel_x_range, (n,), self.device)
        vel[:, 1] = sample_uniform(*self.cfg.racket_vel_y_range, (n,), self.device)
        vel[:, 2] = sample_uniform(*self.cfg.racket_vel_z_range, (n,), self.device)
        self.racket_target_vel_w[env_ids] = vel

        if self.cfg.normal_mode == "velocity":
            normal = vel / (torch.norm(vel, dim=-1, keepdim=True) + 1e-6)
        else:  # "sampled"
            normal = torch.empty(n, 3, device=self.device)
            normal[:, 0] = sample_uniform(*self.cfg.racket_normal_x_range, (n,), self.device)
            normal[:, 1] = sample_uniform(*self.cfg.racket_normal_y_range, (n,), self.device)
            normal[:, 2] = sample_uniform(*self.cfg.racket_normal_z_range, (n,), self.device)
            normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
        self.racket_target_normal_w[env_ids] = normal

    def _sample_targets_reference_perturbed(self, env_ids: Sequence[int], origins: torch.Tensor, n: int):
        """Target = reference racket state @ strike + curriculum-scaled uniform perturbation.

        Guarantees the target is reachable by the imitated swing (a perfect imitator scores exactly),
        with the perturbation ball widening over training (``_perturb_scale``) for generalization.
        """
        self._ensure_reference_strike_state()
        scale = self._perturb_scale()
        dev = self.device
        pos_h = torch.tensor(self.cfg.ref_perturb_pos, device=dev) * scale
        vel_h = torch.tensor(self.cfg.ref_perturb_vel, device=dev) * scale
        nrm_h = float(self.cfg.ref_perturb_normal) * scale

        dpos = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * pos_h
        motion_cmd = self._motion()
        root_relative = None
        if hasattr(self, "_ref_racket_pos_rel_by_motion"):
            motion_ids = motion_cmd.motion_ids[env_ids]
            ref_pos = self._ref_racket_pos_rel_by_motion[motion_ids]
            ref_vel = self._ref_racket_vel_w_by_motion[motion_ids]
            ref_normal = self._ref_racket_normal_w_by_motion[motion_ids]
            # Dense A3 candidate clips store their canonical target in the
            # immutable initial-root-heading frame.  The raw body arrays are
            # still world poses with the source root anchor (normally z=1.0684),
            # so using them directly here and adding ``origins`` double-counts
            # the root anchor.  Convert the canonical root-relative state into
            # the live robot frame exactly as the manifest target path does.
            motion = motion_cmd.motion
            if isinstance(motion, MotionLibraryLoader):
                root_relative = motion.strike_target_is_root_relative[motion_ids]
                if torch.any(root_relative):
                    heading = yaw_quat(self.base_quat_w[env_ids])
                    root_pos = self.base_pos_w[env_ids]
                    canonical_pos = quat_apply(heading, motion.strike_pos_b0[motion_ids])
                    canonical_vel = quat_apply(heading, motion.strike_vel_b0[motion_ids])
                    canonical_normal = quat_apply(heading, motion.strike_normal_b0[motion_ids])
                    ref_pos = torch.where(
                        root_relative.unsqueeze(-1), root_pos + canonical_pos, ref_pos
                    )
                    ref_vel = torch.where(
                        root_relative.unsqueeze(-1), canonical_vel, ref_vel
                    )
                    ref_normal = torch.where(
                        root_relative.unsqueeze(-1), canonical_normal, ref_normal
                    )
        else:
            ref_pos = self._ref_racket_pos_rel.unsqueeze(0).expand(n, -1)
            ref_vel = self._ref_racket_vel_w.unsqueeze(0).expand(n, -1)
            ref_normal = self._ref_racket_normal_w.unsqueeze(0).expand(n, -1)
        # Legacy reference clips store body positions relative to the
        # environment origin; dense candidate clips use the live root pose for
        # root-relative canonical targets.  Do not add ``origins`` twice for
        # the latter.
        if root_relative is None:
            target_pos = origins + ref_pos
        else:
            target_pos = torch.where(root_relative.unsqueeze(-1), ref_pos, origins + ref_pos)
        self.racket_target_pos_w[env_ids] = target_pos + dpos

        dvel = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * vel_h
        self.racket_target_vel_w[env_ids] = ref_vel + dvel

        dnrm = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * nrm_h
        normal = ref_normal + dnrm
        self.racket_target_normal_w[env_ids] = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)

        self.metrics["ref_perturb_scale"][env_ids] = scale

    def _sample_targets_manifest_perturbed(self, env_ids: Sequence[int], origins: torch.Tensor, n: int):
        """Perturb each motion's calibrated manifest strike state locally.

        This is the fixed-base target-conditioned stage: the motion remains
        the teacher, while the actor must map the same reference family to
        nearby impact targets.  The perturbation is deliberately small and
        tied to the selected motion rather than sampled from a global box.
        """

        motion_cmd = self._motion()
        motion = motion_cmd.motion
        if not isinstance(motion, MotionLibraryLoader):
            raise RuntimeError("target_mode='manifest_perturbed' requires MotionCommandCfg.motion_manifest")

        motion_ids = motion_cmd.motion_ids[env_ids]
        scale = self._perturb_scale()
        dev = self.device
        pos_h = torch.tensor(self.cfg.manifest_perturb_pos, device=dev) * scale
        vel_h = torch.tensor(self.cfg.manifest_perturb_vel, device=dev) * scale
        nrm_h = float(self.cfg.manifest_perturb_normal) * scale

        nominal = torch.rand(n, device=dev) < float(self.cfg.manifest_nominal_probability)
        dpos = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * pos_h
        dpos[nominal] = 0.0
        root_relative = motion.strike_target_is_root_relative[motion_ids]
        heading = yaw_quat(self.base_quat_w[env_ids])
        root_relative_pos = self.base_pos_w[env_ids] + quat_apply(
            heading, motion.strike_pos_b0[motion_ids]
        )
        world_pos = origins + motion.strike_pos_w[motion_ids]
        self.racket_target_pos_w[env_ids] = torch.where(
            root_relative.unsqueeze(-1), root_relative_pos, world_pos
        ) + dpos

        dvel = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * vel_h
        dvel[nominal] = 0.0
        root_relative_vel = quat_apply(heading, motion.strike_vel_b0[motion_ids])
        self.racket_target_vel_w[env_ids] = torch.where(
            root_relative.unsqueeze(-1), root_relative_vel, motion.strike_vel_w[motion_ids]
        ) + dvel

        dnrm = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * nrm_h
        dnrm[nominal] = 0.0
        root_relative_normal = quat_apply(heading, motion.strike_normal_b0[motion_ids])
        nominal_normal = torch.where(
            root_relative.unsqueeze(-1), root_relative_normal, motion.strike_normal_w[motion_ids]
        )
        normal = nominal_normal + dnrm
        self.racket_target_normal_w[env_ids] = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
        self.metrics["ref_perturb_scale"][env_ids] = scale

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if bool(getattr(self._env, "v13b_private_motion_disabled", False)):
            # Final V1.3B iterations keep this private target term registered
            # only for manager compatibility.  It must not sample or resolve
            # a motion clip after the prior handoff.
            return
        n = len(env_ids)
        origins = self._env.scene.env_origins[env_ids]

        # Desired racket pos/vel/normal — either independent box sampling (legacy) or coupled to the
        # reference swing's strike state.
        if self.cfg.target_mode == "manifest":
            self._sample_targets_manifest(env_ids, origins, n)
        elif self.cfg.target_mode == "reference_perturbed":
            self._sample_targets_reference_perturbed(env_ids, origins, n)
        elif self.cfg.target_mode == "manifest_perturbed":
            self._sample_targets_manifest_perturbed(env_ids, origins, n)
        else:
            self._sample_targets_uniform(env_ids, origins, n)

        # Snapshot the sampled nominal target before applying a runtime
        # external position.  Velocity and normal are already anchor values
        # in the first adapter stage, so keep their snapshots explicit too.
        self.racket_anchor_target_pos_w[env_ids] = self.racket_target_pos_w[env_ids]
        self.racket_anchor_target_vel_w[env_ids] = self.racket_target_vel_w[env_ids]
        self.racket_anchor_target_normal_w[env_ids] = self.racket_target_normal_w[env_ids]

        # P0/P1 adapter curriculum: sample the *external* command around the
        # frozen manifest anchor only after the anchor snapshot above.  This
        # differs intentionally from ``manifest_perturbed``: model_900 never
        # sees this displacement, while the target adapter does.
        self._sample_adapter_external_offset(env_ids)

        # A latched external position has priority over the sampled/manifest
        # position.  Desired velocity and face normal intentionally remain
        # those of the selected anchor in the first target-conditioning audit.
        self._apply_external_target_position(env_ids)
        self._latch_external_delta_receipt(env_ids)
        self.metrics["adapter_external_target_delta_norm"] = torch.linalg.vector_norm(
            self.racket_target_pos_w - self.racket_anchor_target_pos_w, dim=-1
        )
        self.metrics["adapter_pair_active"] = self.adapter_pair_active.to(
            dtype=self.racket_target_pos_w.dtype
        )

        # Desired base XY (world): COUPLE it to the racket target so standing there keeps the racket
        # reachable by the imitated swing — base_target = racket_target_xy - (reference base->racket
        # offset). Independent sampling used to fight the arm's reach (the base_position reward pulled
        # the base away from where the racket needed it). base_target_*_range is now a SMALL JITTER
        # around the coupled point. Legacy "uniform" mode keeps the old origin-relative sampling.
        if self.cfg.target_mode == "manifest":
            if self.cfg.manifest_base_aligned:
                base_xy = self.base_pos_w[env_ids, :2].clone()
            else:
                base_xy = origins[:, :2].clone() + self._manifest_base_target_xy(env_ids)
        elif self.cfg.target_mode == "reference_perturbed":
            self._ensure_reference_strike_state()
            # Use the selected motion's reach offset.  The old scalar alias
            # pointed at motion 0, which silently coupled every sampled clip
            # to the first reference in a multi-motion bank.  For root-relative
            # candidate clips, rotate the canonical offset into the live base
            # yaw frame; legacy world-frame clips retain their cached offset.
            motion_cmd = self._motion()
            motion_ids = motion_cmd.motion_ids[env_ids]
            if hasattr(self, "_ref_reach_offset_xy_by_motion"):
                reach = self._ref_reach_offset_xy_by_motion[motion_ids]
                motion = motion_cmd.motion
                if isinstance(motion, MotionLibraryLoader):
                    root_relative = motion.strike_target_is_root_relative[motion_ids]
                    if torch.any(root_relative):
                        heading = yaw_quat(self.base_quat_w[env_ids])
                        canonical_reach = quat_apply(heading, motion.strike_pos_b0[motion_ids])[:, :2]
                        reach = torch.where(root_relative.unsqueeze(-1), canonical_reach, reach)
            else:
                reach = self._ref_reach_offset_xy.unsqueeze(0).expand(n, -1)
            base_xy = self.racket_target_pos_w[env_ids][:, :2] - reach
        elif self.cfg.target_mode == "manifest_perturbed":
            # Fixed-base training has no base-position objective.  Keep the
            # command well-defined without coupling a multi-motion batch to
            # the cached reference state of the first environment.
            base_xy = origins[:, :2].clone()
        else:
            base_xy = origins[:, :2].clone()
        base_xy[:, 0] += sample_uniform(*self.cfg.base_target_x_range, (n,), self.device)
        base_xy[:, 1] += sample_uniform(*self.cfg.base_target_y_range, (n,), self.device)
        self.base_target_pos_w[env_ids] = base_xy

        # Swing type from target Y relative to the nominal base Y (right arm holds the paddle).
        base_y_nom = origins[:, 1] + self.cfg.base_nominal_offset[1]
        dy = self.racket_target_pos_w[env_ids][:, 1] - base_y_nom
        if self.cfg.forehand_on_negative_y:
            self.swing_sign[env_ids] = torch.where(dy <= 0.0, 1.0, -1.0)
        else:
            self.swing_sign[env_ids] = torch.where(dy >= 0.0, 1.0, -1.0)

        # Stamp the motion phase baseline for these envs so the per-swing wrap detector in
        # _update_command does not immediately re-trigger after this (e.g. reset-time) resample.
        self._prev_motion_steps[env_ids] = self._motion().time_steps[env_ids]

    def _sample_adapter_external_offset(self, env_ids: Sequence[int]) -> None:
        half_range = torch.tensor(
            self.cfg.adapter_external_offset_half_range, dtype=self.racket_target_pos_w.dtype, device=self.device
        )
        if half_range.shape != (3,) or torch.any(half_range < 0.0):
            raise ValueError("adapter_external_offset_half_range must be three non-negative values")
        if not torch.any(half_range > 0.0):
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        n = ids.numel()
        delta_b = (2.0 * torch.rand(n, 3, device=self.device) - 1.0) * half_range
        self.adapter_pair_active[ids] = False
        self.adapter_pair_baseline_env[ids] = ids
        if bool(self.cfg.adapter_external_paired):
            rows = torch.zeros((7, 3), dtype=delta_b.dtype, device=self.device)
            rows[1, 0], rows[2, 0] = half_range[0], -half_range[0]
            rows[3, 1], rows[4, 1] = half_range[1], -half_range[1]
            rows[5, 2], rows[6, 2] = half_range[2], -half_range[2]
            # CommandManager may resample one completed environment at a time.
            # Pair by stable global env id rather than by this resample batch.
            role = torch.remainder(ids, 7)
            delta_b[:] = rows[role]
            self.adapter_pair_baseline_env[ids] = ids - role
            self.adapter_pair_active[ids] = True
        zero_probability = float(self.cfg.adapter_external_zero_probability)
        if not 0.0 <= zero_probability <= 1.0:
            raise ValueError("adapter_external_zero_probability must be in [0, 1]")
        if zero_probability > 0.0 and not bool(self.cfg.adapter_external_paired):
            delta_b[torch.rand(n, device=self.device) < zero_probability] = 0.0
        base_yaw = yaw_quat(self.base_quat_w[env_ids])
        self.racket_target_pos_w[env_ids] = (
            self.racket_anchor_target_pos_w[env_ids] + quat_apply(base_yaw, delta_b)
        )

    def set_external_target_position_b(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        target_position_b: torch.Tensor | Sequence[float] | Sequence[Sequence[float]],
    ) -> None:
        """Latch a fixed racket position supplied in the command-time base-yaw frame.

        ``target_position_b`` may be one ``[x, y, z]`` vector (broadcast to
        all selected environments) or one vector per selected environment.
        Only position is overridden; the selected motion continues to supply
        desired velocity, face normal, timing, and reference motion.
        """
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if ids.numel() == 0:
            return
        if torch.any(ids < 0) or torch.any(ids >= self.num_envs):
            raise IndexError(
                f"external target env_ids outside [0, {self.num_envs - 1}]: "
                f"{ids.detach().cpu().tolist()}"
            )
        target = torch.as_tensor(
            target_position_b, dtype=self.racket_target_pos_w.dtype, device=self.device
        )
        if target.shape == (3,):
            target = target.unsqueeze(0).expand(ids.numel(), -1)
        if target.shape != (ids.numel(), 3):
            raise ValueError(
                "target_position_b must have shape (3,) or "
                f"({ids.numel()}, 3), got {tuple(target.shape)}"
            )
        if not torch.isfinite(target).all():
            raise ValueError("target_position_b must contain only finite values")

        self._external_target_position_b[ids] = target
        self._external_target_position_active[ids] = True
        self._apply_external_target_position(ids)
        self._latch_external_delta_receipt(ids)

    def clear_external_target_position(
        self, env_ids: Sequence[int] | torch.Tensor, *, resample: bool = True
    ) -> None:
        """Release a runtime position latch and optionally restore normal sampling."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if ids.numel() == 0:
            return
        self._external_target_position_active[ids] = False
        if resample:
            self._resample_command(ids)

    def _apply_external_target_position(
        self, env_ids: Sequence[int] | torch.Tensor
    ) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if ids.numel() == 0:
            return
        active_ids = ids[self._external_target_position_active[ids]]
        if active_ids.numel() == 0:
            return
        heading = yaw_quat(self.base_quat_w[active_ids])
        self.racket_target_pos_w[active_ids] = (
            self.base_pos_w[active_ids]
            + quat_apply(heading, self._external_target_position_b[active_ids])
        )

    def _latch_external_delta_receipt(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Store external-minus-anchor position in the receipt base-yaw frame."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if ids.numel() == 0:
            return
        heading = yaw_quat(self.base_quat_w[ids])
        delta_w = self.racket_target_pos_w[ids] - self.racket_anchor_target_pos_w[ids]
        self._external_target_delta_receipt_b[ids] = quat_rotate_inverse(heading, delta_w)

    def _sample_targets_manifest(self, env_ids: Sequence[int], origins: torch.Tensor, n: int):
        """Target = manifest impact racket state for the selected per-env motion."""
        motion_cmd = self._motion()
        motion = motion_cmd.motion
        if not isinstance(motion, MotionLibraryLoader):
            raise RuntimeError("target_mode='manifest' requires MotionCommandCfg.motion_manifest")
        motion_ids = motion_cmd.motion_ids[env_ids]
        root_relative = motion.strike_target_is_root_relative[motion_ids]
        if self.cfg.manifest_base_aligned:
            hit_steps = motion.hit_frame[motion_ids]
            ref_base_pos = motion._body_pos_w[motion_ids, hit_steps, 0]
            ref_reach = motion.strike_pos_w[motion_ids] - ref_base_pos
            self.racket_target_pos_w[env_ids] = self.base_pos_w[env_ids] + ref_reach
        elif bool(root_relative.any().item()):
            # Dense candidate-bank targets are root-relative.  Convert them
            # once from the current floating-base heading frame into world
            # coordinates.  World-frame scene-placed manifests continue to
            # use the legacy branch below, so mixed manifests remain safe.
            heading = yaw_quat(self.base_quat_w[env_ids])
            root_relative_pos = self.base_pos_w[env_ids] + quat_apply(
                heading, motion.strike_pos_b0[motion_ids]
            )
            world_pos = origins + motion.strike_pos_w[motion_ids]
            self.racket_target_pos_w[env_ids] = torch.where(
                root_relative.unsqueeze(-1), root_relative_pos, world_pos
            )
        else:
            self.racket_target_pos_w[env_ids] = origins + motion.strike_pos_w[motion_ids]
        heading = yaw_quat(self.base_quat_w[env_ids])
        root_relative_vel = quat_apply(heading, motion.strike_vel_b0[motion_ids])
        root_relative_normal = quat_apply(heading, motion.strike_normal_b0[motion_ids])
        self.racket_target_vel_w[env_ids] = torch.where(
            root_relative.unsqueeze(-1), root_relative_vel, motion.strike_vel_w[motion_ids]
        )
        self.racket_target_normal_w[env_ids] = torch.where(
            root_relative.unsqueeze(-1), root_relative_normal, motion.strike_normal_w[motion_ids]
        )
        self.metrics["ref_perturb_scale"][env_ids] = 0.0

    def _manifest_base_target_xy(self, env_ids: Sequence[int]) -> torch.Tensor:
        motion_cmd = self._motion()
        motion = motion_cmd.motion
        if not isinstance(motion, MotionLibraryLoader):
            return torch.zeros(len(env_ids), 2, device=self.device)
        motion_ids = motion_cmd.motion_ids[env_ids]
        steps = motion.hit_frame[motion_ids]
        base_pos_xy = motion._body_pos_w[motion_ids, steps, 0, :2]
        return base_pos_xy

    def _compute_racket_state(self):
        data = self.robot.data
        if self._racket_mode == "body":
            idx = self._racket_body_index
            self.racket_pos_w = data.body_pos_w[:, idx]
            self.racket_quat_w = data.body_quat_w[:, idx]
            self.racket_lin_vel_w = data.body_lin_vel_w[:, idx]
        else:
            widx = self._wrist_body_index
            wpos = data.body_pos_w[:, widx]
            wquat = data.body_quat_w[:, widx]
            wlin = data.body_lin_vel_w[:, widx]
            wang = data.body_ang_vel_w[:, widx]
            offset_w = quat_apply(wquat, self._mount_offset)
            self.racket_pos_w = wpos + offset_w
            self.racket_lin_vel_w = wlin + torch.cross(wang, offset_w, dim=-1)
            self.racket_quat_w = quat_mul(wquat, self._mount_quat)
        # Face normal = chosen local axis of the racket frame, mapped to world.
        # TODO(asset): confirm mount_normal_axis/sign against pingpang_red_Link.STL (see hope-a3-racket-mount).
        self.racket_normal_w = (
            matrix_from_quat(self.racket_quat_w)[:, :, self.cfg.mount_normal_axis] * self.cfg.mount_normal_sign
        )

    def _compute_strike_timing(self):
        """Refresh time_to_strike / pre_strike / strike_window from the CURRENT motion phase.

        The ``motion`` command term computes before this one (it is a parent-class field, registered
        first), so by now ``motion.time_steps`` is already advanced for this control step. Computing the
        timing here — at the top of _update_metrics, alongside the fresh racket FK — keeps the strike
        masks ALIGNED with the racket pose they gate. Previously the masks were only set in
        _update_command (which runs AFTER _update_metrics), so _update_metrics read a 1-step-stale
        time_to_strike: ``exact_strike`` fired one control frame LATE, after the paddle had flown ~one
        step past the strike (~12 cm at a 6 m/s swing) — which collapsed the measured position accuracy
        (exact-strike pos<7.5cm read ~11% instead of the true ~68%) while barely moving velocity (flat
        near the peak). That made strike_composite_success_exact ~6x pessimistic vs the honest probe.
        """
        motion = self._motion()
        if isinstance(motion.motion, MotionLibraryLoader):
            strike_step = motion.motion.hit_frame[motion.motion_ids]
        else:
            total = max(int(motion.motion.time_step_total), 1)
            strike_step = torch.full(
                (self.num_envs,),
                round(self.cfg.strike_phase * (total - 1)),
                dtype=torch.long,
                device=self.device,
            )
        self.time_to_strike = (strike_step - motion.time_steps).float() * self._env.step_dt
        self.pre_strike = self.time_to_strike > 0.0
        self.strike_window = self.time_to_strike.abs() <= self.cfg.strike_window_s

    def _update_command(self):
        motion = self._motion()
        # Timing is refreshed in _update_metrics (aligned with the FK); recompute here too so a direct
        # _update_command call outside the compute() path stays correct. Idempotent within a step
        # (motion.time_steps is unchanged between the two calls).
        self._compute_strike_timing()

        # Re-sample the target at each new swing (the motion clip wrapped to an earlier frame).
        # _resample_command re-stamps _prev_motion_steps for the affected envs; the full clone below
        # keeps every env current. Targets for fresh episodes are sampled by the manager's reset.
        wrapped = torch.where(motion.time_steps < self._prev_motion_steps)[0]
        if len(wrapped) > 0:
            self._resample_command(wrapped)
        self._prev_motion_steps = motion.time_steps.clone()

    def _update_metrics(self):
        # CommandTerm.compute() runs _update_metrics() BEFORE _update_command(), so refresh the
        # actual racket FK AND the strike timing here (once per step) — metrics, rewards, and
        # observations then all read the same fresh, phase-aligned buffers (rewards/obs read them
        # after the full command_manager.compute()). _compute_strike_timing must run before any
        # exact_strike / strike_window gating below (see its docstring: the old stale read measured
        # the strike one control frame too late).
        self._compute_racket_state()
        self._compute_strike_timing()
        origins = self._env.scene.env_origins
        pos_err = torch.norm(self.racket_pos_w - self.racket_target_pos_w, dim=-1)
        vel_err = torch.norm(self.racket_lin_vel_w - self.racket_target_vel_w, dim=-1)
        cos_ang = torch.sum(self.racket_normal_w * self.racket_target_normal_w, dim=-1).clamp(-1.0, 1.0)
        normal_err_deg = torch.acos(cos_ang) * (180.0 / math.pi)
        base_err = torch.norm(self.base_pos_w[:, :2] - self.base_target_pos_w, dim=-1)
        base_pos_rel = self.base_pos_w[:, :2] - origins[:, :2]
        base_err_xy = self.base_pos_w[:, :2] - self.base_target_pos_w
        racket_pos_err_vec = self.racket_pos_w - self.racket_target_pos_w
        racket_vel_err_vec = self.racket_lin_vel_w - self.racket_target_vel_w

        # Episode-wide (instantaneous) errors.
        self.metrics["racket_pos_error"] = pos_err
        self.metrics["racket_vel_error"] = vel_err
        self.metrics["racket_normal_error_deg"] = normal_err_deg
        self.metrics["base_pos_error"] = base_err
        self.metrics["time_to_strike_s"] = self.time_to_strike
        self.metrics["pre_strike_flag"] = self.pre_strike.float()
        self.metrics["strike_window_hit_rate"] = self.strike_window.float()
        if self.cfg.target_mode in ("reference_perturbed", "manifest_perturbed"):
            self.metrics["ref_perturb_scale"] = torch.full_like(pos_err, self._perturb_scale())
        else:
            self.metrics["ref_perturb_scale"].zero_()
        # Per-axis ERROR components only (which direction is the miss?). The per-axis actual/target
        # state and the speed/normal-cos scalars were dropped as redundant log clutter.
        for axis_idx, axis in enumerate(("x", "y")):
            self.metrics[f"base_pos_{axis}"] = base_pos_rel[:, axis_idx]
            self.metrics[f"base_pos_error_{axis}"] = base_err_xy[:, axis_idx]
        for axis_idx, axis in enumerate(("x", "y", "z")):
            self.metrics[f"racket_pos_error_{axis}"] = racket_pos_err_vec[:, axis_idx]
            self.metrics[f"racket_vel_error_{axis}"] = racket_vel_err_vec[:, axis_idx]

        # Strike-window-gated: hold the value sampled during the most recent strike window. The gating
        # masks come from the previous _update_command (<=1-step / 20 ms lag at 50 Hz — negligible vs
        # the ±strike_window_s window). Between strikes the held value carries to the next reset.
        in_win = self.strike_window
        exact_strike = torch.abs(self.time_to_strike) <= (0.5 * self._env.step_dt + 1e-6)
        self.metrics["exact_strike_hit_rate"] = exact_strike.float()
        self.metrics["racket_pos_error_exact_strike"] = torch.where(
            exact_strike, pos_err, self.metrics["racket_pos_error_exact_strike"]
        )
        self.metrics["racket_vel_error_exact_strike"] = torch.where(
            exact_strike, vel_err, self.metrics["racket_vel_error_exact_strike"]
        )
        self.metrics["racket_normal_error_deg_exact_strike"] = torch.where(
            exact_strike, normal_err_deg, self.metrics["racket_normal_error_deg_exact_strike"]
        )
        # --- CONDITIONAL exact-strike success (the trustworthy, undiluted metric) -------------------
        # Old bug: strike_composite_success_exact was a per-env HELD value (last exact-strike result,
        # else init 0). CommandTerm.reset() logs mean(metric[env_ids]) over the RESETTING envs then zeros
        # them, so every env that reset without ever registering an exact-strike frame contributed 0 ->
        # the logged value was ~10x diluted vs the true conditional pass rate (raw probe ~0.32 logged
        # ~0.03), and the success-gated curriculum never advanced off ref_perturb_curriculum_start.
        # Fix: report the fraction of *exact-strike samples* that pass each threshold as a sample-weighted
        # EMA, broadcast to every env, so the reset-mean, the curriculum's .mean(), and the per-env value
        # all equal the conditional rate. pos/vel/normal are also logged separately.
        pass_pos = (pos_err < self.cfg.strike_success_pos_thresh) & exact_strike
        pass_vel = (vel_err < self.cfg.strike_success_vel_thresh) & exact_strike
        pass_normal = (normal_err_deg < self.cfg.strike_success_normal_thresh_deg) & exact_strike
        pass_comp = pass_pos & pass_vel & pass_normal
        decay = float(self.cfg.exact_success_decay)
        self._exact_n_acc = decay * self._exact_n_acc + float(exact_strike.sum())
        self._exact_pass_comp_acc = decay * self._exact_pass_comp_acc + float(pass_comp.sum())
        self._exact_pass_pos_acc = decay * self._exact_pass_pos_acc + float(pass_pos.sum())
        self._exact_pass_vel_acc = decay * self._exact_pass_vel_acc + float(pass_vel.sum())
        self._exact_pass_normal_acc = decay * self._exact_pass_normal_acc + float(pass_normal.sum())
        enough = self._exact_n_acc >= float(self.cfg.exact_success_min_count)
        denom = max(self._exact_n_acc, 1e-6)
        self._exact_composite_rate = (self._exact_pass_comp_acc / denom) if enough else 0.0
        # Broadcast in place so the entries reset() zeros are refreshed before the next reset logs them.
        self.metrics["strike_composite_success_exact"][:] = self._exact_composite_rate
        self.metrics["strike_pos_pass_exact"][:] = (self._exact_pass_pos_acc / denom) if enough else 0.0
        self.metrics["strike_vel_pass_exact"][:] = (self._exact_pass_vel_acc / denom) if enough else 0.0
        self.metrics["strike_normal_pass_exact"][:] = (self._exact_pass_normal_acc / denom) if enough else 0.0
        self.metrics["exact_strike_sample_count_decayed"][:] = self._exact_n_acc
        # Per-axis position error AT the exact strike frame (which axis is the miss?). The position-only
        # strike_success_exact was dropped — strike_pos_pass_exact above is the same signal, undiluted.
        _axis_err_exact = torch.abs(self.racket_pos_w - self.racket_target_pos_w)
        for _ai, _ax in enumerate(("x", "y", "z")):
            self.metrics[f"racket_pos_error_{_ax}_exact_strike"] = torch.where(
                exact_strike, _axis_err_exact[:, _ai], self.metrics[f"racket_pos_error_{_ax}_exact_strike"]
            )
        # Success-gated curriculum: widen the perturbation only once the smoothed CONDITIONAL exact-strike
        # composite success (fraction of exact-strike samples passing all three thresholds) clears the bar.
        if self.cfg.target_mode in ("reference_perturbed", "manifest_perturbed") and self.cfg.ref_perturb_success_gated:
            if (
                self._curr_perturb_scale < 1.0
                and enough
                and self._exact_composite_rate > self.cfg.ref_perturb_advance_threshold
            ):
                self._curr_perturb_scale = min(
                    1.0, self._curr_perturb_scale + float(self.cfg.ref_perturb_advance_rate)
                )
        self.metrics["racket_pos_error_at_strike"] = torch.where(
            in_win, pos_err, self.metrics["racket_pos_error_at_strike"]
        )
        self.metrics["racket_vel_error_at_strike"] = torch.where(
            in_win, vel_err, self.metrics["racket_vel_error_at_strike"]
        )
        self.metrics["racket_normal_error_deg_at_strike"] = torch.where(
            in_win, normal_err_deg, self.metrics["racket_normal_error_deg_at_strike"]
        )
        self.metrics["strike_success"] = torch.where(
            in_win, (pos_err < self.cfg.strike_success_pos_thresh).float(), self.metrics["strike_success"]
        )


        # Base target is tracked before the strike, so log that error during the pre-strike phase.
        self.metrics["base_pos_error_pre_strike"] = torch.where(
            self.pre_strike, base_err, self.metrics["base_pos_error_pre_strike"]
        )

        # Swing-quality detail held at the most recent strike: actual/target paddle speed and the
        # per-axis position error (which direction is the miss?).
        racket_speed = torch.norm(self.racket_lin_vel_w, dim=-1)
        target_speed = torch.norm(self.racket_target_vel_w, dim=-1)
        axis_err = torch.abs(self.racket_pos_w - self.racket_target_pos_w)
        self.metrics["racket_speed_at_strike"] = torch.where(
            in_win, racket_speed, self.metrics["racket_speed_at_strike"]
        )
        self.metrics["racket_target_speed_at_strike"] = torch.where(
            in_win, target_speed, self.metrics["racket_target_speed_at_strike"]
        )
        self.metrics["racket_pos_error_x_at_strike"] = torch.where(
            in_win, axis_err[:, 0], self.metrics["racket_pos_error_x_at_strike"]
        )
        self.metrics["racket_pos_error_y_at_strike"] = torch.where(
            in_win, axis_err[:, 1], self.metrics["racket_pos_error_y_at_strike"]
        )
        self.metrics["racket_pos_error_z_at_strike"] = torch.where(
            in_win, axis_err[:, 2], self.metrics["racket_pos_error_z_at_strike"]
        )
        self.metrics["strike_success_5cm"] = torch.where(
            in_win, (pos_err < 0.05).float(), self.metrics["strike_success_5cm"]
        )
        self.metrics["strike_success_10cm"] = torch.where(
            in_win, (pos_err < 0.10).float(), self.metrics["strike_success_10cm"]
        )

        # Robot-health diagnostics (episode-wide, instantaneous).
        data = self.robot.data
        self.metrics["base_height"] = data.root_pos_w[:, 2]
        self.metrics["base_upright"] = matrix_from_quat(self.base_quat_w)[:, 2, 2]  # 1.0 = perfectly upright
        self.metrics["joint_vel_abs_max"] = torch.max(torch.abs(data.joint_vel), dim=-1).values
        if self._has_jpos_limits:
            limits = getattr(data, "soft_joint_pos_limits", None)
            if limits is None:
                limits = data.joint_pos_limits
            half_span = ((limits[..., 1] - limits[..., 0]) * 0.5).clamp(min=1e-6)
            dist = torch.minimum(data.joint_pos - limits[..., 0], limits[..., 1] - data.joint_pos).clamp(min=0.0)
            self.metrics["joint_pos_near_limit_frac"] = ((dist / half_span) < 0.1).float().mean(dim=-1)
        if self._has_torque:
            tau_abs = torch.abs(data.applied_torque)
            self.metrics["joint_torque_abs_mean"] = torch.mean(tau_abs, dim=-1)
            self.metrics["joint_torque_abs_max"] = torch.max(tau_abs, dim=-1).values
        act = getattr(self._env.action_manager, "action", None)
        if act is not None:
            a_abs = torch.abs(act)
            self.metrics["action_abs_mean"] = torch.mean(a_abs, dim=-1)
            self.metrics["action_abs_max"] = torch.max(a_abs, dim=-1).values
            prev_act = getattr(self._env.action_manager, "prev_action", None)
            if prev_act is not None:
                delta_abs = torch.abs(act - prev_act)
                self.metrics["action_delta_abs_mean"] = torch.mean(delta_abs, dim=-1)
                self.metrics["action_delta_abs_max"] = torch.max(delta_abs, dim=-1).values
            else:
                self.metrics["action_delta_abs_mean"].zero_()
                self.metrics["action_delta_abs_max"].zero_()

    # ------------------------------------------------------------------ #
    # Observation helpers (base-relative quantities)
    # ------------------------------------------------------------------ #
    def racket_target_pos_b(self) -> torch.Tensor:
        """Desired racket position relative to the base (yaw-heading frame). HITTER actor obs."""
        return quat_rotate_inverse(yaw_quat(self.base_quat_w), self.racket_target_pos_w - self.base_pos_w)

    def racket_anchor_target_pos_b(self) -> torch.Tensor:
        """Nominal anchor position in the current base yaw frame."""
        return quat_rotate_inverse(
            yaw_quat(self.base_quat_w), self.racket_anchor_target_pos_w - self.base_pos_w
        )

    def racket_anchor_target_vel_b(self) -> torch.Tensor:
        return quat_rotate_inverse(yaw_quat(self.base_quat_w), self.racket_anchor_target_vel_w)

    def racket_anchor_target_normal_b(self) -> torch.Tensor:
        return quat_rotate_inverse(yaw_quat(self.base_quat_w), self.racket_anchor_target_normal_w)

    def external_target_delta_local_b(self) -> torch.Tensor:
        """External-minus-anchor position in the strike-local base frame.

        The fixed-base P0 contract uses the base yaw frame, which is identical
        to the anchor strike frame.  Keeping this method on the command term
        makes the frame explicit and leaves room for a full anchor quaternion
        in the floating-base migration.
        """
        if bool(getattr(self.cfg, "external_delta_receipt_frame", False)):
            return self._external_target_delta_receipt_b
        return self.racket_target_pos_b() - self.racket_anchor_target_pos_b()

    def racket_target_vel_b(self) -> torch.Tensor:
        """Desired racket linear velocity in the base yaw-heading frame."""
        return quat_rotate_inverse(yaw_quat(self.base_quat_w), self.racket_target_vel_w)

    def racket_target_normal_b(self) -> torch.Tensor:
        """Desired racket face normal in the base yaw-heading frame."""
        return quat_rotate_inverse(yaw_quat(self.base_quat_w), self.racket_target_normal_w)

    def base_target_pos_b(self) -> torch.Tensor:
        """Desired base XY position relative to the current base (yaw-heading frame). HITTER actor obs."""
        delta_xy = self.base_target_pos_w - self.base_pos_w[:, :2]
        delta = torch.cat([delta_xy, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
        return quat_rotate_inverse(yaw_quat(self.base_quat_w), delta)[:, :2]

    def strike_temporal_weight(self) -> torch.Tensor:
        """Event-centered strike reward weight."""
        std = float(self.cfg.strike_time_std_s)
        if std <= 0.0:
            return self.strike_window.float()
        return torch.exp(-0.5 * torch.square(self.time_to_strike / std))

    def strike_reward_mask(self) -> torch.Tensor:
        """Legacy temporal weight; V1.3B overrides this with a one-shot pulse."""
        return self.strike_temporal_weight()

    # ------------------------------------------------------------------ #
    # Debug visualization (no-op stubs; targets are world-frame buffers).
    # ------------------------------------------------------------------ #
    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


class ReferenceFreeRacketTargetCommand(RacketTargetCommand):
    """Reference-free global target sampler for V1.3B."""

    cfg: "ReferenceFreeRacketTargetCommandCfg"

    def __init__(self, cfg: "ReferenceFreeRacketTargetCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # ``episode_time_s`` is the sole public strike clock.  Teacher frame
        # indices are derived from the event sampled at reset; they never
        # create a second hit timer.
        self._episode_time_s = torch.zeros(self.num_envs, device=self.device)
        self._previous_tau = torch.zeros(self.num_envs, device=self.device)
        self._next_event_id = 0
        # Isaac's manager may perform an initialization resample before the
        # first public env.reset().  Keep reset-time sampling distinct from a
        # forbidden mid-episode command resample.
        self._reset_in_progress = False
        zeros_f = torch.zeros(self.num_envs, device=self.device)
        zeros_i = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        zeros_b = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        zeros_v = torch.zeros(self.num_envs, 3, device=self.device)
        self.strike_event = EpisodeStrikeEvent(
            event_id=zeros_i.clone(),
            motion_id=zeros_i.clone(),
            teacher_start_frame=zeros_i.clone(),
            teacher_hit_frame=zeros_i.clone(),
            episode_strike_time_s=zeros_f.clone(),
            teacher_physical_strike_time_s=zeros_f.clone(),
            teacher_position_b=zeros_v.clone(),
            teacher_velocity_b=zeros_v.clone(),
            teacher_normal_b=zeros_v.clone(),
            sampled_position_b=zeros_v.clone(),
            sampled_velocity_b=zeros_v.clone(),
            sampled_normal_b=zeros_v.clone(),
            sampled_timing_offset_s=zeros_f.clone(),
            strike_armed=zeros_b.clone(),
            strike_consumed=zeros_b.clone(),
            goal_sample_count=zeros_i.clone(),
            goal_resample_count_after_reset=zeros_i.clone(),
            strike_event_count=zeros_i.clone(),
            upper_prior_wrap_count=zeros_i.clone(),
        )
        self._strike_reward_now = zeros_b.clone()
        self._v13b_previous_distance = torch.zeros(self.num_envs, device=self.device)
        self.metrics["v13b_goal_resample_exhausted"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["v13b_goal_fallback_nominal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["v13b_curriculum_progress"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["v13b_workspace_curriculum_progress"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["v13b_goal_acceptance_rate"] = torch.ones(self.num_envs, device=self.device)
        self.metrics["v13b_reference_free"] = torch.ones(self.num_envs, device=self.device)
        # Workspace-expansion diagnostics.  These remain zero in the current
        # nominal-local sampler, so adding them cannot change the existing
        # training contract.
        self.metrics["v13b_workspace_motion_anchor_fraction"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["v13b_workspace_global_fraction"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["v13b_workspace_eligible_anchor_fraction"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["v13b_workspace_eligible_anchor_count"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["v13b_workspace_anchor_id"] = torch.full(
            (self.num_envs,), -1.0, device=self.device
        )
        self.metrics["v13b_workspace_anchor_distance_m"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["v13b_workspace_nominal_fallback_count"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["v13b_workspace_out_of_bounds_reject_count"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["v13b_workspace_resample_count"] = torch.zeros(
            self.num_envs, device=self.device
        )
        # Training-only alignment diagnostics.  The public actor still sees
        # only the sampled 10-D racket goal; these values merely record how
        # strongly the sampler was anchored to the private motion used by the
        # frozen priors.
        self.metrics["v13b_motion_goal_alignment_weight"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["v13b_motion_goal_alignment_time_s"] = torch.zeros(self.num_envs, device=self.device)
        # Event/audit fields are never observations.  They make the one-shot
        # and teacher/public-time contracts externally testable.
        for name in (
            "v13b_event_id", "v13b_motion_id", "v13b_teacher_start_frame",
            "v13b_teacher_frame", "v13b_teacher_hit_frame", "v13b_goal_sample_count",
            "v13b_goal_resample_count_after_reset", "v13b_strike_event_count",
            "v13b_upper_prior_wrap_count", "v13b_strike_armed", "v13b_strike_consumed",
            "v13b_strike_reward_trigger", "v13b_public_strike_time_s",
            "v13b_teacher_physical_strike_time_s", "v13b_teacher_public_time_error_s",
            "v13b_goal_teacher_position_error_m", "v13b_goal_teacher_velocity_error_mps",
            "v13b_goal_teacher_normal_error_deg", "v13b_post_hit_phase",
            "v13b_episode_time_s", "v13b_episode_step", "v13b_teacher_time_s",
            "v13b_teacher_hit_episode_time_s", "v13b_recovery_gate",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        for prefix in ("v13b_goal", "v13b_teacher"):
            for component in ("position_x", "position_y", "position_z", "velocity_x", "velocity_y", "velocity_z", "normal_x", "normal_y", "normal_z"):
                self.metrics[f"{prefix}_{component}"] = torch.zeros(self.num_envs, device=self.device)
        # Post-hit recovery diagnostics are deliberately environment-side
        # metrics.  They use the public signed time-to-hit, never expose a
        # phase or recovery state to the 98-D actor, and let long training
        # distinguish one clean strike from a persistent oscillation.
        self.metrics["v13b_post_hit_gate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["v13b_post_hit_torso_ang_vel_rms"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["v13b_post_hit_torso_tilt_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["v13b_post_hit_root_forward_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["v13b_post_hit_recovered"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["v13b_post_hit_time_to_recover_s"] = torch.zeros(self.num_envs, device=self.device)
        self._v13b_recovery_stable_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._v13b_recovery_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._v13b_recovery_time_s = torch.zeros(self.num_envs, device=self.device)
        self._workspace_runtime_counts = getattr(self._env, "v13b_workspace_runtime_counters", {
            "workspace_anchor_sample_count": 0,
            "workspace_global_sample_count": 0,
            "workspace_nominal_fallback_count": 0,
            "workspace_resample_count": 0,
            "workspace_out_of_bounds_reject_count": 0,
            "motion_command_activation_count": 0,
            "reference_action_apply_count": 0,
            "p5u_migration_runtime_count": 0,
            "model900_forward_count": 0,
            "model3396_forward_count": 0,
        })
        for _key in (
            "workspace_anchor_sample_count", "workspace_global_sample_count",
            "workspace_nominal_fallback_count", "workspace_resample_count",
            "workspace_out_of_bounds_reject_count", "motion_command_activation_count",
            "reference_action_apply_count", "p5u_migration_runtime_count",
            "model900_forward_count", "model3396_forward_count",
        ):
            self._workspace_runtime_counts.setdefault(_key, 0)
        self._env.v13b_workspace_runtime_counters = self._workspace_runtime_counts
        self._v13b_torso_body_id: int | None = None
        self._workspace_anchor_bank = None
        if bool(getattr(cfg, "workspace_anchor_bank_enabled", False)):
            if not bool(getattr(cfg, "workspace_expansion_enabled", False)):
                raise RuntimeError("workspace anchor bank requires workspace_expansion_enabled=true")
            if str(getattr(cfg, "workspace_sampling_mode", "")) != "audited_anchor_bank":
                raise RuntimeError("workspace expansion must use workspace_sampling_mode=audited_anchor_bank")
            from training.utils.workspace_anchor_bank import WorkspaceStrikeAnchorBank
            manifest = str(getattr(cfg, "workspace_anchor_manifest", ""))
            if not manifest:
                raise RuntimeError("workspace_anchor_manifest is required for WorkspaceExpansion")
            self._workspace_anchor_bank = WorkspaceStrikeAnchorBank(
                manifest,
                self.device,
                require_qualified=bool(getattr(cfg, "workspace_anchor_requires_qualified", False)),
                nominal_local=tuple(cfg.nominal_target_local_xyz),
                support_half_range=tuple(getattr(cfg, "workspace_local_support_half_range_xyz", (0.08, 0.08, 0.08))),
                support_tolerance_m=float(getattr(cfg, "workspace_support_distance_tolerance_m", 1.0e-4)),
            )
            stats = self._workspace_anchor_bank.statistics(tuple(cfg.nominal_target_local_xyz))
            print(
                "[V1.3B WorkspaceExpansion] anchor-only metadata bank loaded: "
                f"count={stats['anchor_count']} support_inside={stats['inside_current_support_count']} "
                f"manifest={stats['manifest']}", flush=True
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset exactly one strike event for each newly reset environment.

        CommandTerm.reset legitimately samples the new episode goal.  The
        contract assertion in ``_latch_episode_strike_event`` must therefore
        only reject resampling triggered by ``compute()`` while an episode is
        already running.
        """
        self._reset_in_progress = True
        try:
            return super().reset(env_ids)
        finally:
            self._reset_in_progress = False

    def _progress(self) -> float:
        workspace_override = getattr(self, "_workspace_curriculum_progress", None)
        if workspace_override is not None:
            return float(max(0.0, min(1.0, workspace_override)))
        override = getattr(self, "_v13b_policy_progress", None)
        if override is not None:
            return float(max(0.0, min(1.0, override)))
        # Workspace expansion owns an independent clock. It must not infer
        # target coverage from the old prior/teacher annealing clock.
        value = getattr(self._env, "workspace_curriculum_progress", None)
        if value is None:
            value = getattr(getattr(self._env, "unwrapped", None), "workspace_curriculum_progress", None)
        if value is not None:
            return float(max(0.0, min(1.0, value)))
        value = getattr(self._env, "v13b_policy_progress", None)
        if value is None:
            value = getattr(getattr(self._env, "unwrapped", None), "v13b_policy_progress", None)
        return 0.0 if value is None else float(max(0.0, min(1.0, value)))

    def _motion_goal_alignment(self, ids: torch.Tensor):
        """Return a private teacher strike state only while teacher is active.

        The returned state is evaluated at the same wrist+mount TCP used by
        V1.3B rewards.  Crucially, this helper never extends public timing:
        :meth:`_sample_reference_free_goal` instead rephases teacher playback
        to the sampled public time-to-hit.
        """
        workspace_anchor_enabled = bool(
            getattr(self.cfg, "workspace_anchor_bank_enabled", False)
        )
        if not workspace_anchor_enabled and not bool(
            getattr(self.cfg, "motion_alignment_enabled", False)
        ):
            return None
        if workspace_anchor_enabled:
            start = float(getattr(self.cfg, "workspace_anchor_start_progress", 0.0))
            end = float(getattr(self.cfg, "workspace_anchor_end_progress", 1.0))
        else:
            start = float(getattr(self.cfg, "motion_alignment_start_progress", 0.0))
            end = float(getattr(self.cfg, "motion_alignment_end_progress", 0.65))
        progress = self._progress()
        # The current CompletePriors run hands off to the nominal global goal
        # at ``end``.  The later workspace-expansion run may explicitly keep
        # the private motion bank available as a sampler-side anchor; this is
        # still invisible to the public 98-D actor.
        keep_final = bool(getattr(self.cfg, "workspace_keep_motion_anchor_final", False))
        if progress >= end and not keep_final:
            return None
        try:
            motion_cmd = self._motion()
        except Exception:
            return None
        motion = getattr(motion_cmd, "motion", None)
        motion_ids = getattr(motion_cmd, "motion_ids", None)
        if motion is None or motion_ids is None or not hasattr(motion, "hit_frame"):
            return None
        motion_ids = motion_ids[ids].to(dtype=torch.long)
        base_pos = self.base_pos_w[ids]
        base_yaw = yaw_quat(self.base_quat_w[ids])
        origins = self._env.scene.env_origins[ids]
        root_relative = motion.strike_target_is_root_relative[motion_ids]
        pos_w = torch.where(
            root_relative.unsqueeze(-1),
            base_pos + quat_apply(base_yaw, motion.strike_pos_b0[motion_ids]),
            origins + motion.strike_pos_w[motion_ids],
        )
        vel_w = torch.where(
            root_relative.unsqueeze(-1),
            quat_apply(base_yaw, motion.strike_vel_b0[motion_ids]),
            motion.strike_vel_w[motion_ids],
        )
        normal_w = torch.where(
            root_relative.unsqueeze(-1),
            quat_apply(base_yaw, motion.strike_normal_b0[motion_ids]),
            motion.strike_normal_w[motion_ids],
        )
        normal_w = normal_w / torch.linalg.vector_norm(normal_w, dim=-1, keepdim=True).clamp_min(1.0e-6)
        pos_b = quat_rotate_inverse(base_yaw, pos_w - base_pos)
        vel_b = quat_rotate_inverse(base_yaw, vel_w)
        normal_b = quat_rotate_inverse(base_yaw, normal_w)
        if keep_final:
            weight = torch.ones((ids.numel(),), dtype=base_pos.dtype, device=self.device)
        elif end <= start:
            weight = torch.full(
                (ids.numel(),), 1.0 if progress <= start else 0.0,
                dtype=base_pos.dtype, device=self.device,
            )
        else:
            # ``_progress`` is a Python float because the runner owns the
            # update counter.  Keep the interpolation scalar-safe instead of
            # calling Tensor.clamp on a float during env.reset().
            u = max(0.0, min(1.0, (progress - start) / (end - start)))
            smooth = u * u * (3.0 - 2.0 * u)
            weight = torch.full((ids.numel(),), 1.0 - smooth, dtype=base_pos.dtype, device=self.device)
        hit_frame = motion.hit_frame[motion_ids].to(dtype=torch.long)
        fps = float(max(int(motion.fps), 1))
        return pos_b, vel_b, normal_b, motion_ids, hit_frame, fps, weight

    def _workspace_anchor_sample(self, count: int):
        """Sample audited anchor metadata without instantiating a motion teacher."""
        if self._workspace_anchor_bank is None:
            return None
        return self._workspace_anchor_bank.sample(
            count,
            self._progress(),
            tuple(self.cfg.nominal_target_local_xyz),
            tuple(getattr(self.cfg, "workspace_local_support_half_range_xyz", (0.08, 0.08, 0.08))),
        )

    def workspace_runtime_audit(self) -> dict[str, int]:
        """Return counters accumulated from actual Workspace sampler paths."""
        return {key: int(value) for key, value in self._workspace_runtime_counts.items()}

    def _latch_episode_strike_event(
        self,
        *,
        ids: torch.Tensor,
        motion_ids: torch.Tensor,
        teacher_start: torch.Tensor,
        teacher_hit: torch.Tensor,
        public_hit_time: torch.Tensor,
        teacher_physical_hit: torch.Tensor,
        teacher_pos: torch.Tensor,
        teacher_vel: torch.Tensor,
        teacher_normal: torch.Tensor,
        sampled_pos: torch.Tensor,
        sampled_vel: torch.Tensor,
        sampled_normal: torch.Tensor,
    ) -> None:
        """Atomically establish the only strike event for each reset env."""
        event = self.strike_event
        n = int(ids.numel())
        # A V1.3B reference-free target must never be regenerated mid-episode.
        if (not self._reset_in_progress) and torch.any(self._episode_time_s[ids] > 1.0e-6):
            event.goal_resample_count_after_reset[ids] += 1
            raise RuntimeError("V1.3B contract violation: target resampled after episode reset")
        event.event_id[ids] = torch.arange(
            self._next_event_id, self._next_event_id + n, dtype=torch.long, device=self.device
        )
        self._next_event_id += n
        event.motion_id[ids] = motion_ids
        event.teacher_start_frame[ids] = teacher_start
        event.teacher_hit_frame[ids] = teacher_hit
        event.episode_strike_time_s[ids] = public_hit_time
        event.teacher_physical_strike_time_s[ids] = teacher_physical_hit
        event.teacher_position_b[ids] = teacher_pos
        event.teacher_velocity_b[ids] = teacher_vel
        event.teacher_normal_b[ids] = teacher_normal
        event.sampled_position_b[ids] = sampled_pos
        event.sampled_velocity_b[ids] = sampled_vel
        event.sampled_normal_b[ids] = sampled_normal
        event.sampled_timing_offset_s[ids] = public_hit_time - float(self.cfg.nominal_time_to_hit_s)
        event.strike_armed[ids] = True
        event.strike_consumed[ids] = False
        event.goal_sample_count[ids] = 1
        event.goal_resample_count_after_reset[ids] = 0
        event.strike_event_count[ids] = 0
        event.upper_prior_wrap_count[ids] = 0
        self._strike_reward_now[ids] = False
        self._episode_time_s[ids] = 0.0
        self._previous_tau[ids] = public_hit_time
        self._write_event_metrics(ids)

    def _write_event_metrics(self, ids: torch.Tensor | None = None) -> None:
        event = self.strike_event
        sl = slice(None) if ids is None else ids
        normal_dot = torch.sum(event.sampled_normal_b[sl] * event.teacher_normal_b[sl], dim=-1).clamp(-1.0, 1.0)
        self.metrics["v13b_event_id"][sl] = event.event_id[sl].float()
        self.metrics["v13b_motion_id"][sl] = event.motion_id[sl].float()
        self.metrics["v13b_teacher_start_frame"][sl] = event.teacher_start_frame[sl].float()
        self.metrics["v13b_teacher_hit_frame"][sl] = event.teacher_hit_frame[sl].float()
        self.metrics["v13b_goal_sample_count"][sl] = event.goal_sample_count[sl].float()
        self.metrics["v13b_goal_resample_count_after_reset"][sl] = event.goal_resample_count_after_reset[sl].float()
        self.metrics["v13b_strike_event_count"][sl] = event.strike_event_count[sl].float()
        self.metrics["v13b_upper_prior_wrap_count"][sl] = event.upper_prior_wrap_count[sl].float()
        self.metrics["v13b_strike_armed"][sl] = event.strike_armed[sl].float()
        self.metrics["v13b_strike_consumed"][sl] = event.strike_consumed[sl].float()
        self.metrics["v13b_public_strike_time_s"][sl] = event.episode_strike_time_s[sl]
        self.metrics["v13b_teacher_physical_strike_time_s"][sl] = event.teacher_physical_strike_time_s[sl]
        self.metrics["v13b_teacher_hit_episode_time_s"][sl] = event.teacher_physical_strike_time_s[sl]
        self.metrics["v13b_teacher_public_time_error_s"][sl] = torch.abs(
            event.episode_strike_time_s[sl] - event.teacher_physical_strike_time_s[sl]
        )
        self.metrics["v13b_goal_teacher_position_error_m"][sl] = torch.linalg.vector_norm(
            event.sampled_position_b[sl] - event.teacher_position_b[sl], dim=-1
        )
        self.metrics["v13b_goal_teacher_velocity_error_mps"][sl] = torch.linalg.vector_norm(
            event.sampled_velocity_b[sl] - event.teacher_velocity_b[sl], dim=-1
        )
        self.metrics["v13b_goal_teacher_normal_error_deg"][sl] = torch.rad2deg(torch.acos(normal_dot))
        for axis, suffix in enumerate(("x", "y", "z")):
            self.metrics[f"v13b_goal_position_{suffix}"][sl] = event.sampled_position_b[sl, axis]
            self.metrics[f"v13b_teacher_position_{suffix}"][sl] = event.teacher_position_b[sl, axis]
            self.metrics[f"v13b_goal_velocity_{suffix}"][sl] = event.sampled_velocity_b[sl, axis]
            self.metrics[f"v13b_teacher_velocity_{suffix}"][sl] = event.teacher_velocity_b[sl, axis]
            self.metrics[f"v13b_goal_normal_{suffix}"][sl] = event.sampled_normal_b[sl, axis]
            self.metrics[f"v13b_teacher_normal_{suffix}"][sl] = event.teacher_normal_b[sl, axis]

    def strike_reward_mask(self) -> torch.Tensor:
        """One control-frame pulse used by the exact-strike reward term."""
        return self._strike_reward_now

    def _sample_reference_free_goal(self, ids: torch.Tensor) -> None:
        n = ids.numel()
        p = self._progress()
        warm = float(self.cfg.curriculum_warmup_progress)
        ramp = float(self.cfg.curriculum_ramp_end_progress)
        alpha = 0.0 if p <= warm else min(1.0, (p - warm) / max(ramp - warm, 1.0e-6))
        self.metrics["v13b_curriculum_progress"][ids] = p
        if "v13b_workspace_curriculum_progress" in self.metrics:
            self.metrics["v13b_workspace_curriculum_progress"][ids] = p
        initial = torch.tensor(self.cfg.initial_position_half_range_m, device=self.device)
        final = torch.tensor(self.cfg.final_position_half_range_m, device=self.device)
        pos_half = (initial + alpha * (final - initial)).unsqueeze(0).expand(n, -1)
        normal_deg = float(self.cfg.initial_normal_half_angle_deg) + alpha * (
            float(self.cfg.final_normal_half_angle_deg) - float(self.cfg.initial_normal_half_angle_deg)
        )
        base_yaw = yaw_quat(self.base_quat_w[ids])
        base_pos = self.base_pos_w[ids]
        nominal_local = torch.tensor(self.cfg.nominal_target_local_xyz, device=self.device, dtype=base_pos.dtype)
        workspace_mode = str(getattr(self.cfg, "workspace_sampling_mode", "nominal_local"))
        workspace_enabled = bool(getattr(self.cfg, "workspace_expansion_enabled", False))
        anchor_sample = self._workspace_anchor_sample(n) if workspace_enabled else None
        alignment = None if anchor_sample is not None else self._motion_goal_alignment(ids)
        if workspace_mode not in {
            "nominal_local",
            "motion_anchor_per_action",
            "motion_anchor_workspace_mixture",
            "audited_anchor_bank",
            "workspace_box_uniform",
        }:
            raise ValueError(f"Unknown V1.3B workspace_sampling_mode={workspace_mode!r}")
        if workspace_enabled and workspace_mode == "audited_anchor_bank" and anchor_sample is None:
            raise RuntimeError("WorkspaceExpansion anchor sample is unavailable")
        if workspace_enabled and workspace_mode != "nominal_local" and alignment is None and anchor_sample is None:
            raise RuntimeError(
                "V1.3B workspace expansion requires an active private motion anchor; "
                "keep motion_alignment enabled and workspace_keep_motion_anchor_final=true"
            )
        time_half_range = float(self.cfg.initial_time_half_range_s) + alpha * (
            float(self.cfg.final_time_half_range_s) - float(self.cfg.initial_time_half_range_s)
        )
        # Public target timing is always the Planner/deployment contract.
        # It is never stretched to match a legacy motion duration.
        public_hit_time = float(self.cfg.nominal_time_to_hit_s) + sample_uniform(
            -time_half_range, time_half_range, (n,), self.device
        )
        time_min, time_max = self.cfg.time_to_hit_range_s
        public_hit_time = public_hit_time.clamp(min=float(time_min), max=float(time_max))
        if anchor_sample is not None:
            alignment_pos, alignment_vel, alignment_normal, anchor_times, motion_ids, eligible_count = anchor_sample
            alignment_weight = torch.ones(n, device=self.device, dtype=base_pos.dtype)
            hit_frames = torch.full((n,), -1, dtype=torch.long, device=self.device)
            motion_fps = 0.0
            self.metrics["v13b_workspace_anchor_id"][ids] = motion_ids.to(dtype=base_pos.dtype)
            self.metrics["v13b_workspace_eligible_anchor_count"][ids] = float(eligible_count)
            self.metrics["v13b_workspace_eligible_anchor_fraction"][ids] = float(eligible_count) / float(self._workspace_anchor_bank.anchor_count)
            self.metrics["v13b_workspace_anchor_distance_m"][ids] = torch.linalg.vector_norm(
                alignment_pos - nominal_local.unsqueeze(0), dim=-1
            )
            self._workspace_runtime_counts["workspace_anchor_sample_count"] += int(n)
        elif alignment is None:
            alignment_pos = nominal_local.unsqueeze(0).expand(n, -1)
            alignment_vel = torch.zeros(n, 3, device=self.device, dtype=base_pos.dtype)
            alignment_normal = torch.zeros(n, 3, device=self.device, dtype=base_pos.dtype)
            alignment_normal[:, 0] = 1.0
            alignment_weight = torch.zeros(n, device=self.device, dtype=base_pos.dtype)
            motion_ids = torch.full((n,), -1, dtype=torch.long, device=self.device)
            hit_frames = torch.full((n,), -1, dtype=torch.long, device=self.device)
            motion_fps = 0.0
        else:
            alignment_pos, alignment_vel, alignment_normal, motion_ids, hit_frames, motion_fps, alignment_weight = alignment
            alignment_normal = alignment_normal / torch.linalg.vector_norm(alignment_normal, dim=-1, keepdim=True).clamp_min(1.0e-6)
        # Keep the private teacher values intact for the event audit.  The
        # workspace-expansion sampler may select a global target for some
        # environments, but that must not rewrite the teacher/public
        # comparison fields.
        teacher_alignment_pos = alignment_pos.clone()
        teacher_alignment_vel = alignment_vel.clone()
        teacher_alignment_normal = alignment_normal.clone()

        workspace_motion_mask = (
            torch.ones(n, dtype=torch.bool, device=self.device)
            if anchor_sample is not None
            else torch.zeros(n, dtype=torch.bool, device=self.device)
        )
        workspace_global_center = nominal_local.unsqueeze(0).expand(n, -1)
        if workspace_enabled and workspace_mode not in ("nominal_local", "audited_anchor_bank"):
            workspace_local_min = torch.tensor(self.cfg.workspace_local_min_xyz, device=self.device, dtype=base_pos.dtype)
            workspace_local_max = torch.tensor(self.cfg.workspace_local_max_xyz, device=self.device, dtype=base_pos.dtype)
            workspace_global_center = workspace_local_min + torch.rand(
                n, 3, device=self.device, dtype=base_pos.dtype
            ) * (workspace_local_max - workspace_local_min)
            if workspace_mode == "motion_anchor_per_action":
                workspace_motion_mask[:] = True
            elif workspace_mode == "motion_anchor_workspace_mixture":
                motion_probability = float(getattr(self.cfg, "workspace_motion_anchor_probability", 0.70))
                if not 0.0 <= motion_probability <= 1.0:
                    raise ValueError("workspace_motion_anchor_probability must be in [0, 1]")
                workspace_motion_mask = torch.rand(n, device=self.device) < motion_probability
            # workspace_box_uniform deliberately leaves the mask all false.
            alignment_weight = workspace_motion_mask.to(dtype=base_pos.dtype)
            alignment_pos = torch.where(
                workspace_motion_mask.unsqueeze(-1),
                teacher_alignment_pos,
                workspace_global_center,
            )
            zero_vel = torch.zeros_like(teacher_alignment_vel)
            alignment_vel = torch.where(
                workspace_motion_mask.unsqueeze(-1), teacher_alignment_vel, zero_vel
            )
            default_normal = torch.zeros_like(teacher_alignment_normal)
            default_normal[:, 0] = 1.0
            alignment_normal = torch.where(
                workspace_motion_mask.unsqueeze(-1), teacher_alignment_normal, default_normal
            )
            alignment_normal = alignment_normal / torch.linalg.vector_norm(
                alignment_normal, dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
            self._workspace_runtime_counts["workspace_global_sample_count"] += int((~workspace_motion_mask).sum().item())
        self.metrics["v13b_motion_goal_alignment_weight"][ids] = alignment_weight
        self.metrics["v13b_motion_goal_alignment_time_s"][ids] = public_hit_time
        self.metrics["v13b_workspace_motion_anchor_fraction"][ids] = (
            workspace_motion_mask.float() if workspace_enabled else 0.0
        )
        self.metrics["v13b_workspace_global_fraction"][ids] = (
            (~workspace_motion_mask).float() if workspace_enabled else 0.0
        )
        center_local = alignment_weight.unsqueeze(-1) * alignment_pos + (1.0 - alignment_weight.unsqueeze(-1)) * nominal_local
        if workspace_enabled and workspace_mode not in ("nominal_local", "audited_anchor_bank"):
            center_local = torch.where(
                workspace_motion_mask.unsqueeze(-1), alignment_pos, workspace_global_center
            )
        local_pos = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * pos_half
        local_pos += center_local
        if workspace_enabled and workspace_mode not in ("nominal_local", "audited_anchor_bank"):
            # Motion-anchor environments receive the audited per-action
            # perturbation.  Global environments are sampled directly from
            # the bounded local workspace, rather than from nominal +/- 8 cm.
            local_pos = torch.where(
                workspace_motion_mask.unsqueeze(-1),
                local_pos,
                workspace_global_center,
            )
        # Cheap fail-closed workspace filter.  Detailed IK/collision probes
        # remain an offline admission gate; no rejected sample is reported as
        # a PPO failure. Boundary values are resampled up to the configured
        # attempt count. In audited-anchor mode exhaustion falls back to that
        # same anchor; no nominal/global or unchecked/clamped target appears.
        local_min = torch.tensor(self.cfg.workspace_local_min_xyz, device=self.device)
        local_max = torch.tensor(self.cfg.workspace_local_max_xyz, device=self.device)
        accepted = (local_pos >= local_min) & (local_pos <= local_max)
        accepted = accepted.all(dim=-1)
        attempts = 1
        while not bool(torch.all(accepted)) and attempts < int(self.cfg.max_resample_attempts):
            bad = ~accepted
            replacement = (torch.rand(int(bad.sum()), 3, device=self.device) * 2.0 - 1.0) * pos_half[bad]
            replacement += center_local[bad]
            if workspace_enabled and workspace_mode not in ("nominal_local", "audited_anchor_bank"):
                global_min = torch.tensor(self.cfg.workspace_local_min_xyz, device=self.device, dtype=base_pos.dtype)
                global_max = torch.tensor(self.cfg.workspace_local_max_xyz, device=self.device, dtype=base_pos.dtype)
                global_replacement = global_min + torch.rand(
                    int(bad.sum()), 3, device=self.device, dtype=base_pos.dtype
                ) * (global_max - global_min)
                replacement = torch.where(
                    workspace_motion_mask[bad].unsqueeze(-1), replacement, global_replacement
                )
            local_pos[bad] = replacement
            accepted = ((local_pos >= local_min) & (local_pos <= local_max)).all(dim=-1)
            attempts += 1
        exhausted = ~accepted
        self._workspace_runtime_counts["workspace_out_of_bounds_reject_count"] += int((~accepted).sum().item())
        self._workspace_runtime_counts["workspace_resample_count"] += int(max(attempts - 1, 0) * max(int(n), 1))
        if torch.any(exhausted):
            local_pos[exhausted] = center_local[exhausted]
        nominal_fallback = exhausted & (~workspace_motion_mask)
        self._workspace_runtime_counts["workspace_nominal_fallback_count"] += int(nominal_fallback.sum().item())
        self.metrics["v13b_goal_resample_exhausted"][ids] = exhausted.to(dtype=self.racket_target_pos_w.dtype)
        self.metrics["v13b_goal_fallback_nominal"][ids] = nominal_fallback.to(dtype=self.racket_target_pos_w.dtype)
        self.metrics["v13b_goal_acceptance_rate"][ids] = (~exhausted).to(dtype=self.racket_target_pos_w.dtype)
        self.metrics["v13b_workspace_nominal_fallback_count"][ids] = nominal_fallback.to(dtype=self.racket_target_pos_w.dtype)
        self.metrics["v13b_workspace_out_of_bounds_reject_count"][ids] = exhausted.to(dtype=self.racket_target_pos_w.dtype)
        self.metrics["v13b_workspace_resample_count"][ids] = float(max(attempts - 1, 0))
        self.racket_target_pos_w[ids] = base_pos + quat_apply(base_yaw, local_pos)
        # Uniformly sample a disk in the tangent plane so the normal angular
        # deviation is bounded by (rather than independently exceeded by)
        # the configured half-angle.
        angle_limit = math.sin(math.radians(normal_deg))
        random_vec = torch.randn(n, 3, device=self.device)
        tangent = random_vec - torch.sum(random_vec * alignment_normal, dim=-1, keepdim=True) * alignment_normal
        tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1.0e-6)
        tangent_radius = torch.sqrt(torch.rand(n, device=self.device)) * angle_limit
        normal_local = alignment_normal * torch.sqrt(torch.clamp(1.0 - tangent_radius.square(), min=1.0e-6)).unsqueeze(-1)
        normal_local = normal_local + tangent * tangent_radius.unsqueeze(-1)
        normal_local = normal_local / torch.linalg.vector_norm(normal_local, dim=-1, keepdim=True).clamp_min(1.0e-6)
        self.racket_target_normal_w[ids] = quat_apply(base_yaw, normal_local)
        speed_fraction = float(self.cfg.initial_speed_fraction) + alpha * (
            float(self.cfg.final_speed_fraction) - float(self.cfg.initial_speed_fraction)
        )
        speed = float(self.cfg.nominal_speed_mps) * (
            1.0 + sample_uniform(-speed_fraction, speed_fraction, (n,), self.device)
        )
        speed = speed.clamp(min=float(self.cfg.speed_range_mps[0]), max=float(self.cfg.speed_range_mps[1]))
        global_vel_local = normal_local * speed.unsqueeze(-1)
        # Early training must be a true teacher velocity perturbation, not a
        # teacher direction rescaled to an unrelated global 2.5 m/s nominal.
        teacher_speed_scale = 1.0 + sample_uniform(
            -speed_fraction, speed_fraction, (n,), self.device
        )
        aligned_vel = alignment_vel * teacher_speed_scale.unsqueeze(-1)
        target_vel_local = alignment_weight.unsqueeze(-1) * aligned_vel + (1.0 - alignment_weight.unsqueeze(-1)) * global_vel_local
        self.racket_target_vel_w[ids] = quat_apply(base_yaw, target_vel_local)
        self.base_target_pos_w[ids] = base_pos[:, :2]
        self.swing_sign[ids] = torch.where(local_pos[:, 1] <= 0.0, 1.0, -1.0)
        # Rephase the private teacher, rather than corrupting public tau.
        # teacher physical hit = (hit_frame - start_frame) / fps ~= public_tau
        teacher_start = torch.full((n,), -1, dtype=torch.long, device=self.device)
        teacher_physical_hit = torch.zeros(n, dtype=base_pos.dtype, device=self.device)
        if alignment is not None:
            teacher_steps_to_hit = torch.round(public_hit_time * motion_fps).to(dtype=torch.long)
            teacher_start = torch.clamp(hit_frames - teacher_steps_to_hit, min=0)
            teacher_physical_hit = (hit_frames - teacher_start).to(dtype=base_pos.dtype) / motion_fps
            motion_cmd = self._motion()
            motion_cmd.configure_v13b_episode_strike(ids, teacher_start)
        elif anchor_sample is not None:
            # Anchor metadata is not a teacher clock.  Keep the public final
            # timing distribution authoritative and retain the anchor timing
            # only as an offline feasibility field.
            teacher_physical_hit = public_hit_time.clone()
        self._latch_episode_strike_event(
            ids=ids,
            motion_ids=motion_ids,
            teacher_start=teacher_start,
            teacher_hit=hit_frames,
            public_hit_time=public_hit_time,
            teacher_physical_hit=teacher_physical_hit,
            teacher_pos=teacher_alignment_pos,
            teacher_vel=teacher_alignment_vel,
            teacher_normal=teacher_alignment_normal,
            sampled_pos=local_pos,
            sampled_vel=target_vel_local,
            sampled_normal=normal_local,
        )
        self._reset_post_hit_recovery_metrics(ids)
        self._compute_strike_timing()
        self.racket_anchor_target_pos_w[ids] = self.racket_target_pos_w[ids]
        self.racket_anchor_target_vel_w[ids] = self.racket_target_vel_w[ids]
        self.racket_anchor_target_normal_w[ids] = self.racket_target_normal_w[ids]
        self._v13b_previous_distance[ids] = torch.linalg.vector_norm(
            self.racket_pos_w[ids] - self.racket_target_pos_w[ids], dim=-1
        ).detach()

    def _reset_post_hit_recovery_metrics(self, ids: torch.Tensor) -> None:
        self._v13b_recovery_stable_steps[ids] = 0
        self._v13b_recovery_latched[ids] = False
        self._v13b_recovery_time_s[ids] = 0.0
        for name in (
            "v13b_post_hit_gate",
            "v13b_post_hit_torso_ang_vel_rms",
            "v13b_post_hit_torso_tilt_deg",
            "v13b_post_hit_root_forward_speed",
            "v13b_post_hit_recovered",
            "v13b_post_hit_time_to_recover_s",
        ):
            self.metrics[name][ids] = 0.0

    def _update_post_hit_recovery_metrics(self) -> None:
        """Log physical settling after the one public V1.3B hit event."""

        robot = self.robot
        if self._v13b_torso_body_id is None:
            ids, names = robot.find_bodies(["torso_Link"], preserve_order=True)
            if names != ["torso_Link"]:
                raise ValueError("V1.3B recovery metric cannot resolve torso_Link")
            self._v13b_torso_body_id = int(ids[0])
        torso_id = self._v13b_torso_body_id
        torso_quat_w = robot.data.body_quat_w[:, torso_id]
        torso_ang_vel_b = quat_rotate_inverse(
            torso_quat_w, robot.data.body_ang_vel_w[:, torso_id]
        )
        torso_rms = torch.linalg.vector_norm(torso_ang_vel_b[:, :2], dim=-1) / math.sqrt(2.0)
        gravity_w = torch.zeros_like(robot.data.body_pos_w[:, torso_id])
        gravity_w[:, 2] = -1.0
        gravity_b = quat_rotate_inverse(torso_quat_w, gravity_w)
        tilt_rad = torch.atan2(
            torch.linalg.vector_norm(gravity_b[:, :2], dim=-1),
            (-gravity_b[:, 2]).clamp_min(1.0e-6),
        )
        forward_speed = torch.abs(robot.data.root_lin_vel_b[:, 0])
        elapsed = torch.relu(-self.time_to_strike)
        gate_u = ((elapsed - 0.15) / 0.35).clamp(0.0, 1.0)
        gate = gate_u * gate_u * (3.0 - 2.0 * gate_u)
        post_hit = self.time_to_strike < 0.0
        self.metrics["v13b_post_hit_gate"][:] = gate
        self.metrics["v13b_post_hit_torso_ang_vel_rms"][:] = torch.where(
            post_hit, torso_rms, torch.zeros_like(torso_rms)
        )
        self.metrics["v13b_post_hit_torso_tilt_deg"][:] = torch.where(
            post_hit, torch.rad2deg(tilt_rad), torch.zeros_like(tilt_rad)
        )
        self.metrics["v13b_post_hit_root_forward_speed"][:] = torch.where(
            post_hit, forward_speed, torch.zeros_like(forward_speed)
        )

        # "Recovered" means a clean torso/velocity state continuously held
        # for 0.5 s after the follow-through window.  This is a logged audit
        # quantity, not an observation or a termination condition.
        stable = (
            post_hit
            & (elapsed >= 0.50)
            & (torso_rms <= 0.12)
            & (tilt_rad <= math.radians(5.0))
            & (forward_speed <= 0.12)
        )
        required_steps = max(1, int(round(0.50 / float(self._env.step_dt))))
        active = ~self._v13b_recovery_latched
        self._v13b_recovery_stable_steps[:] = torch.where(
            active & stable,
            self._v13b_recovery_stable_steps + 1,
            torch.where(active, torch.zeros_like(self._v13b_recovery_stable_steps), self._v13b_recovery_stable_steps),
        )
        newly_recovered = active & (self._v13b_recovery_stable_steps >= required_steps)
        self._v13b_recovery_time_s[newly_recovered] = elapsed[newly_recovered]
        self._v13b_recovery_latched |= newly_recovered
        self.metrics["v13b_post_hit_recovered"][:] = self._v13b_recovery_latched.float()
        self.metrics["v13b_post_hit_time_to_recover_s"][:] = torch.where(
            self._v13b_recovery_latched,
            self._v13b_recovery_time_s,
            torch.zeros_like(self._v13b_recovery_time_s),
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # In the training-only V1.3B branch the latest manifest supplies the
        # canonical racket-at-strike goal.  Delegate manifest modes to the
        # base command so root-frame conversion and per-motion velocity/normal
        # are preserved; otherwise the global sampler would silently replace
        # the dataset target.
        if self.cfg.target_mode in ("manifest", "manifest_perturbed"):
            super()._resample_command(env_ids)
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
            if ids.numel():
                self.metrics["v13b_goal_resample_exhausted"][ids] = 0.0
                self.metrics["v13b_goal_fallback_nominal"][ids] = 0.0
                self.metrics["v13b_goal_acceptance_rate"][ids] = 1.0
                self.metrics["v13b_reference_free"][ids] = 1.0
                self._reset_post_hit_recovery_metrics(ids)
            self.metrics["v13b_curriculum_progress"].fill_(self._progress())
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if ids.numel():
            self._sample_reference_free_goal(ids)

    def _compute_strike_timing(self):
        self.time_to_strike = self.strike_event.episode_strike_time_s - self._episode_time_s
        self.pre_strike = self.time_to_strike > 0.0
        self.strike_window = torch.abs(self.time_to_strike) <= float(self.cfg.strike_window_s)

    def _update_strike_crossing(self) -> None:
        """Consume exactly one authoritative tau crossing per episode."""
        event = self.strike_event
        current_tau = self.time_to_strike
        crossing = event.strike_armed & (self._previous_tau > 0.0) & (current_tau <= 0.0)
        self._strike_reward_now.zero_()
        if torch.any(crossing):
            self._strike_reward_now[crossing] = True
            event.strike_armed[crossing] = False
            event.strike_consumed[crossing] = True
            event.strike_event_count[crossing] += 1
        self._previous_tau[:] = current_tau
        self.metrics["v13b_strike_reward_trigger"][:] = self._strike_reward_now.float()
        self._write_event_metrics()
        if bool(getattr(self.cfg, "contract_assertions", False)):
            # Isaac may call ``compute`` once during manager construction
            # before the first reset has sampled an episode event.
            active = event.goal_sample_count > 0
            if not torch.any(active):
                return
            if torch.any(event.goal_sample_count[active] != 1):
                raise RuntimeError("V1.3B contract violation: goal_sample_count != 1")
            if torch.any(event.goal_resample_count_after_reset[active] != 0):
                raise RuntimeError("V1.3B contract violation: goal resampled during episode")
            if torch.any(event.strike_event_count[active] > 1):
                raise RuntimeError("V1.3B contract violation: multiple strike events in one episode")
            if torch.any(event.upper_prior_wrap_count[active] != 0):
                raise RuntimeError("V1.3B contract violation: upper teacher wrapped")
            aligned = active & (event.motion_id >= 0)
            if torch.any(aligned):
                error = torch.abs(
                    event.episode_strike_time_s[aligned] - event.teacher_physical_strike_time_s[aligned]
                )
                if torch.any(error > float(self._env.step_dt) + 1.0e-6):
                    raise RuntimeError("V1.3B contract violation: public/teacher strike time mismatch")

    def _update_teacher_runtime_contract(self) -> None:
        """Disable private bank execution once both action priors are gone."""
        disable_after = float(getattr(self.cfg, "private_motion_disable_progress", 0.70))
        # Workspace-expansion fine-tuning may keep the audited motion bank
        # alive solely as a sampler-side action anchor.  It still never enters
        # the public actor observation or action contract.
        keep_sampler_anchor = bool(
            getattr(self.cfg, "workspace_keep_motion_anchor_final", False)
        ) and bool(getattr(self.cfg, "workspace_expansion_enabled", False))
        disabled = (self._progress() >= disable_after) and not keep_sampler_anchor
        self._env.v13b_private_motion_disabled = bool(disabled)
        if disabled:
            # The public target sampler, student actor and rewards remain live.
            # Only the training-only private motion command / target are made
            # inert; this is the final deployment execution path.
            self.metrics["v13b_motion_goal_alignment_weight"].zero_()

    def _update_event_phase_metrics(self) -> None:
        """Expose private audit phase without changing the 98-D actor."""
        tau = self.time_to_strike
        elapsed = torch.relu(-tau)
        phase = torch.where(
            tau > 0.0,
            torch.zeros_like(tau),  # PRE_STRIKE
            torch.where(
                elapsed < 0.15,
                torch.ones_like(tau),  # FOLLOW_THROUGH
                torch.where(elapsed < 0.50, torch.full_like(tau, 2.0), torch.full_like(tau, 3.0)),
            ),
        )
        self.metrics["v13b_post_hit_phase"][:] = phase
        self.metrics["v13b_episode_time_s"][:] = self._episode_time_s
        self.metrics["v13b_episode_step"][:] = self._episode_time_s / float(self._env.step_dt)
        self.metrics["v13b_recovery_gate"][:] = self.metrics["v13b_post_hit_gate"]
        if not bool(getattr(self._env, "v13b_private_motion_disabled", False)):
            try:
                motion = self._motion()
                self.metrics["v13b_teacher_frame"][:] = motion.time_steps.float()
                self.metrics["v13b_teacher_time_s"][:] = motion.time_steps.float() / float(max(int(motion.motion.fps), 1))
                if hasattr(motion, "v13b_upper_prior_wrap_count"):
                    self.strike_event.upper_prior_wrap_count[:] = motion.v13b_upper_prior_wrap_count
                    self.metrics["v13b_upper_prior_wrap_count"][:] = motion.v13b_upper_prior_wrap_count.float()
            except Exception:
                self.metrics["v13b_teacher_frame"].fill_(-1.0)
        else:
            self.metrics["v13b_teacher_frame"].fill_(-1.0)
            self.metrics["v13b_teacher_time_s"].fill_(-1.0)

    def _update_command(self):
        if self.cfg.target_mode in ("manifest", "manifest_perturbed"):
            super()._update_command()
            self.metrics["v13b_curriculum_progress"].fill_(self._progress())
            self.metrics["v13b_reference_free"].fill_(1.0)
            self._update_post_hit_recovery_metrics()
            return
        self._update_teacher_runtime_contract()
        self._episode_time_s += float(self._env.step_dt)
        self._compute_strike_timing()
        self._update_strike_crossing()
        self._update_post_hit_recovery_metrics()
        self._update_event_phase_metrics()
        self.metrics["v13b_curriculum_progress"].fill_(self._progress())
        # This command is intentionally independent from the private motion
        # command used only by the training-time priors.
        self.metrics["v13b_reference_free"].fill_(1.0)


@configclass
class RacketTargetCommandCfg(CommandTermCfg):
    """Configuration for :class:`RacketTargetCommand`."""

    class_type: type = RacketTargetCommand

    asset_name: str = MISSING
    motion_command_name: str = "motion"

    # The target is re-sampled per swing (on clip wrap / reset), not on a fixed time schedule,
    # so disable the base CommandTerm time-based resampling.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    # --- racket mount FK ---
    racket_body_name: str = "pingpang_red_Link"
    wrist_body_name: str = "right_wrist_yaw_Link"
    mount_offset: tuple[float, float, float] = (0.210211399202899, 0.0320784994676765, 0.0320358706296689)
    # Fixed wrist->racket rotation (w, x, y, z); only used in the wrist_offset FK fallback. Identity
    # for the A3 ping-pong URDF (all mount joints are rpy=0). Set non-identity if the mount tilts the
    # paddle relative to the wrist frame.
    mount_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mount_normal_axis: int = 1  # racket-local +Y is the face normal (red/hitting face)
    mount_normal_sign: float = 1.0  # +1 = red/forehand face; -1 = black/backhand face
    racket_fk_mode: str = "auto"  # auto, body, or explicit wrist_offset

    # --- strike timing (fraction of the reference clip where the paddle meets the ball) ---
    strike_phase: float = 0.46  # HITTER clip: strike at frame 43/94 ≈ 0.46
    strike_window_s: float = 0.1  # half-window; goal-racket reward active within ±strike_window_s
    strike_time_std_s: float = 0.04  # Gaussian temporal kernel std for event-centered strike rewards
    strike_success_pos_thresh: float = 0.075  # m; "strike_success" metric = fraction of strikes with racket pos error below this
    strike_success_vel_thresh: float = 0.5  # m/s; exact-strike racket velocity acceptance threshold
    strike_success_normal_thresh_deg: float = 15.0  # deg; exact-strike face-normal acceptance threshold

    # --- nominal stance (offset of the base from the env origin) ---
    base_nominal_offset: tuple[float, float, float] = (0.0, 0.0, 0.93)

    # --- target generation mode ---
    # "uniform": independent box sampling from the *_range fields below (legacy; the boxes are
    #   PLACEHOLDERS not tied to the swing, so the imitated swing's racket may never pass through them).
    # "reference_perturbed": target = the reference swing's racket state AT the strike frame (pos/vel/
    #   normal, computed by the same FK as the actual racket) + a curriculum-scaled uniform perturbation.
    #   Reachable by construction (a perfect imitator scores exactly); the *_range fields are ignored.
    # "manifest": target = per-motion manifest impact racket state; requires motion_manifest.
    # "manifest_perturbed": the same calibrated state plus a small local
    # target offset, used for target-conditioned fixed-base training.
    target_mode: str = "reference_perturbed"
    manifest_base_aligned: bool = False

    # reference_perturbed perturbation (final half-extents; scaled 0->1 by the curriculum below).
    ref_perturb_pos: tuple[float, float, float] = (0.15, 0.20, 0.15)  # m, per-axis half-range
    ref_perturb_vel: tuple[float, float, float] = (1.0, 1.0, 0.8)  # m/s, per-axis half-range
    ref_perturb_normal: float = 0.30  # face-normal jitter magnitude (added then renormalized)
    # Curriculum: perturbation half-extents ramp from `start`*final to 1.0*final.
    # Success-gated mode (default): `_curr_perturb_scale` starts at `ref_perturb_curriculum_start` and
    # only advances (by `ref_perturb_advance_rate` per control step) once the smoothed exact-strike
    # composite success exceeds `ref_perturb_advance_threshold` — keeps the strike error inside the
    # racket reward kernel's responsive band until the policy demonstrably hits, then widens.
    # Open-loop fallback (success_gated=False): ramp over `ref_perturb_curriculum_steps` control steps
    # (env.common_step_counter); set steps<=0 to disable the ramp (always full).
    ref_perturb_curriculum_steps: int = 30000
    ref_perturb_curriculum_start: float = 0.05
    ref_perturb_success_gated: bool = True
    ref_perturb_advance_threshold: float = 0.30  # widen once smoothed exact-strike composite success > this
    ref_perturb_advance_rate: float = 1.0e-5  # scale increment per control step while above threshold

    # Manifest-centered local target conditioning.  These ranges are small on
    # purpose: the action library covers the broad workspace; residual PPO
    # learns interpolation around each motion's own strike point.
    manifest_perturb_pos: tuple[float, float, float] = (0.0, 0.02, 0.015)
    manifest_perturb_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    manifest_perturb_normal: float = 0.0
    manifest_nominal_probability: float = 0.50

    # Target residual adapter sampling.  Applied *after* the anchor snapshot,
    # in the fixed-base anchor frame.  Keep this at zero outside P0/P1/P2.
    adapter_external_offset_half_range: tuple[float, float, float] = (0.0, 0.0, 0.0)
    adapter_external_zero_probability: float = 0.20
    adapter_external_paired: bool = False
    # Floating-base replay can preserve the external delta in the base-yaw
    # frame at command receipt.  Fixed-base tasks retain the historical
    # current-base interpretation by default.
    external_delta_receipt_frame: bool = False

    # --- conditional exact-strike success metric (logging + curriculum gating) ---
    # The logged strike_*_pass_exact / strike_composite_success_exact are a sample-weighted EMA of the
    # exact-strike pass rate: acc = decay*acc + this-step-count each control step. decay ~0.99 gives a
    # ~100-step (~2 s @ 50 Hz) memory; higher = smoother but slower to reflect the current policy. The
    # rate (and the curriculum) only trust it once `exact_success_min_count` decayed samples accumulate.
    exact_success_decay: float = 0.99
    exact_success_min_count: float = 50.0

    # --- reachable racket-target workspace (offsets from the env origin, world frame, meters) ---
    # Used only by target_mode="uniform". PLACEHOLDER ranges (not the reference strike point).
    racket_pos_x_range: tuple[float, float] = (0.25, 0.55)
    racket_pos_y_range: tuple[float, float] = (-0.45, 0.45)
    racket_pos_z_range: tuple[float, float] = (0.70, 1.15)

    # --- desired racket velocity (world frame, m/s) ---
    racket_vel_x_range: tuple[float, float] = (1.5, 4.0)
    racket_vel_y_range: tuple[float, float] = (-1.0, 1.0)
    racket_vel_z_range: tuple[float, float] = (0.0, 1.5)

    # --- desired racket face normal ---
    normal_mode: str = "velocity"  # "velocity" (n = v/|v|) or "sampled"
    racket_normal_x_range: tuple[float, float] = (0.5, 1.0)
    racket_normal_y_range: tuple[float, float] = (-0.3, 0.3)
    racket_normal_z_range: tuple[float, float] = (-0.3, 0.3)

    # --- desired base XY target (offsets from the env origin, world frame, meters) ---
    base_target_x_range: tuple[float, float] = (-0.10, 0.10)
    base_target_y_range: tuple[float, float] = (-0.35, 0.35)

    # --- swing-type convention ---
    forehand_on_negative_y: bool = True  # right arm holds the paddle: target on -Y side -> forehand (+1)


@configclass
class ReferenceFreeRacketTargetCommandCfg(RacketTargetCommandCfg):
    """Config for a global target-conditioned command without motion data."""

    class_type: type[CommandTerm] = ReferenceFreeRacketTargetCommand
    target_mode: str = "reference_free_global"
    motion_command_name: str = "__disabled_motion__"
    nominal_target_local_xyz: tuple[float, float, float] = (0.42, -0.18, 0.18)
    initial_position_half_range_m: tuple[float, float, float] = (0.01, 0.01, 0.01)
    final_position_half_range_m: tuple[float, float, float] = (0.08, 0.08, 0.08)
    initial_normal_half_angle_deg: float = 2.0
    final_normal_half_angle_deg: float = 12.0
    nominal_speed_mps: float = 2.5
    initial_speed_fraction: float = 0.03
    final_speed_fraction: float = 0.20
    speed_range_mps: tuple[float, float] = (1.0, 4.0)
    nominal_time_to_hit_s: float = 0.40
    initial_time_half_range_s: float = 0.02
    final_time_half_range_s: float = 0.10
    time_to_hit_range_s: tuple[float, float] = (0.20, 0.60)
    # In the complete-prior training branch, align the public target sampler
    # to the private manifest strike state early in training.  This is a
    # sampler-side curriculum only: motion remains absent from the actor
    # observation and these defaults stay disabled for deployment.
    motion_alignment_enabled: bool = False
    motion_alignment_start_progress: float = 0.0
    motion_alignment_end_progress: float = 0.65
    motion_alignment_include_prelude_s: bool = True
    # Retained only for backwards-compatible parsing.  V1.3B never extends
    # public time-to-hit to this range; teacher playback is rephased instead.
    motion_alignment_time_range_s: tuple[float, float] = (0.20, 0.60)
    private_motion_disable_progress: float = 0.70
    contract_assertions: bool = False
    curriculum_warmup_progress: float = 0.10
    # Finish the automatic target curriculum early enough to leave a stable
    # final-distribution plateau for the last 30% of PPO updates.
    curriculum_ramp_end_progress: float = 0.70
    # Bounds are in the same base-heading local frame as
    # ``nominal_target_local_xyz``.  They are deliberately not world-height
    # bounds: the runtime target is formed as ``base_pos_w + R_yaw * local``.
    # Include the audited forehand/backhand canonical bank (forehand reaches
    # about y=-0.79 m and a small subset reaches z<0) while retaining a
    # conservative box for independently sampled deployment goals.
    workspace_local_min_xyz: tuple[float, float, float] = (0.18, -0.85, -0.05)
    workspace_local_max_xyz: tuple[float, float, float] = (0.72, 0.35, 0.45)
    # Isolated next-stage sampler.  The defaults intentionally reproduce the
    # current nominal-local V1.3B run; only the new workspace-expansion task
    # enables these fields.  ``motion_anchor`` uses the already audited
    # racket-at-strike points from the private manifest as sampler anchors,
    # while the actor remains 98-D reference-free.
    workspace_expansion_enabled: bool = False
    workspace_sampling_mode: str = "nominal_local"
    workspace_motion_anchor_probability: float = 0.70
    workspace_keep_motion_anchor_final: bool = False
    workspace_anchor_bank_enabled: bool = False
    workspace_anchor_manifest: str = ""
    workspace_anchor_sampling_enabled: bool = False
    workspace_anchor_source: str = ""
    workspace_anchor_requires_qualified: bool = False
    workspace_local_support_half_range_xyz: tuple[float, float, float] = (0.08, 0.08, 0.08)
    workspace_support_distance_tolerance_m: float = 1.0e-4
    workspace_global_probability: float = 0.0
    max_resample_attempts: int = 32
