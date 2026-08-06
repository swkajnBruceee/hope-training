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
from dataclasses import MISSING
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
        if cfg.racket_body_name in self.robot.body_names:
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
        if hasattr(self, "_ref_racket_pos_rel_by_motion"):
            motion_ids = motion_cmd.motion_ids[env_ids]
            ref_pos = self._ref_racket_pos_rel_by_motion[motion_ids]
            ref_vel = self._ref_racket_vel_w_by_motion[motion_ids]
            ref_normal = self._ref_racket_normal_w_by_motion[motion_ids]
        else:
            ref_pos = self._ref_racket_pos_rel.unsqueeze(0).expand(n, -1)
            ref_vel = self._ref_racket_vel_w.unsqueeze(0).expand(n, -1)
            ref_normal = self._ref_racket_normal_w.unsqueeze(0).expand(n, -1)
        self.racket_target_pos_w[env_ids] = origins + ref_pos + dpos

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
        self.racket_target_pos_w[env_ids] = origins + motion.strike_pos_w[motion_ids] + dpos

        dvel = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * vel_h
        dvel[nominal] = 0.0
        self.racket_target_vel_w[env_ids] = motion.strike_vel_w[motion_ids] + dvel

        dnrm = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * nrm_h
        dnrm[nominal] = 0.0
        normal = motion.strike_normal_w[motion_ids] + dnrm
        self.racket_target_normal_w[env_ids] = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
        self.metrics["ref_perturb_scale"][env_ids] = scale

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
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
            base_xy = self.racket_target_pos_w[env_ids][:, :2] - self._ref_reach_offset_xy.unsqueeze(0)
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
        if self.cfg.manifest_base_aligned:
            hit_steps = motion.hit_frame[motion_ids]
            ref_base_pos = motion._body_pos_w[motion_ids, hit_steps, 0]
            ref_reach = motion.strike_pos_w[motion_ids] - ref_base_pos
            self.racket_target_pos_w[env_ids] = self.base_pos_w[env_ids] + ref_reach
        else:
            self.racket_target_pos_w[env_ids] = origins + motion.strike_pos_w[motion_ids]
        self.racket_target_vel_w[env_ids] = motion.strike_vel_w[motion_ids]
        self.racket_target_normal_w[env_ids] = motion.strike_normal_w[motion_ids]
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

    # ------------------------------------------------------------------ #
    # Debug visualization (no-op stubs; targets are world-frame buffers).
    # ------------------------------------------------------------------ #
    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


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
