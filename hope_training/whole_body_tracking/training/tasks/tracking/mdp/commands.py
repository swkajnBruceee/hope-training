from __future__ import annotations

import math
import json
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)
from .stance_contract import validate_stance_manifest

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionLibraryLoader:
    """GPU tensor motion library loaded from a balanced training manifest."""

    def __init__(
        self,
        manifest_file: str,
        body_indexes: Sequence[int],
        device: str = "cpu",
        subset_size: int | None = None,
        expected_fps: int | None = 50,
        frame_z_offset: float = 0.0,
        ground_align: bool = False,
        validate_stance_contract: bool = False,
        stance_contract_mode: str | None = None,
    ):
        self.manifest_file = self._resolve_path(manifest_file)
        with open(self.manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if validate_stance_contract:
            validate_stance_manifest(
                manifest,
                manifest_file=self.manifest_file,
                expected_mode=stance_contract_mode,
                check_motion_paths=True,
            )

        entries = list(manifest.get("motions", []))
        if not entries:
            raise ValueError(f"motion manifest has no motions: {self.manifest_file}")
        if subset_size is not None and int(subset_size) > 0:
            entries = self._balanced_subset(entries, int(subset_size))

        self.entries = entries
        self.episode_ids = [str(e.get("episode_id", i)) for i, e in enumerate(entries)]
        self.stroke_types = [str(e.get("stroke_type", "unknown")).lower() for e in entries]
        self.stroke_ids = torch.tensor(
            [0 if s == "forehand" else 1 if s == "backhand" else -1 for s in self.stroke_types],
            dtype=torch.long,
            device=device,
        )
        self.forehand_ids = torch.where(self.stroke_ids == 0)[0]
        self.backhand_ids = torch.where(self.stroke_ids == 1)[0]

        arrays: list[dict[str, np.ndarray | float | int]] = []
        motion_paths: list[str] = []
        for entry in entries:
            motion_path = self._entry_motion_path(entry)
            motion_paths.append(motion_path)
            data = np.load(motion_path)
            fps = int(data["fps"])
            if expected_fps is not None and fps != int(expected_fps):
                raise ValueError(f"{motion_path}: fps={fps}, expected {expected_fps}")
            joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
            body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
            if joint_pos.shape[-1] != 31:
                raise ValueError(f"{motion_path}: joint_pos shape {joint_pos.shape}, expected [...,31]")
            if body_pos_w.shape[-2] != 32:
                raise ValueError(f"{motion_path}: body_pos_w shape {body_pos_w.shape}, expected [...,32,3]")
            hit_frame = int(entry.get("hit_event", {}).get("motion_hit_frame", round(0.46 * (joint_pos.shape[0] - 1))))
            if not (0 <= hit_frame < joint_pos.shape[0]):
                raise ValueError(f"{motion_path}: invalid hit_frame={hit_frame} for length={joint_pos.shape[0]}")

            target = entry.get("strike_target", {})
            normal = np.asarray(target.get("racket_normal_w", [0.0, 0.0, 1.0]), dtype=np.float32)
            n = float(np.linalg.norm(normal))
            if not np.isfinite(n) or n < 1e-6:
                raise ValueError(f"{motion_path}: invalid racket_normal_w={normal}")
            normal = normal / n

            arrays.append(
                {
                    "fps": fps,
                    "joint_pos": joint_pos,
                    "joint_vel": np.asarray(data["joint_vel"], dtype=np.float32),
                    "body_pos_w": body_pos_w,
                    "body_quat_w": np.asarray(data["body_quat_w"], dtype=np.float32),
                    "body_lin_vel_w": np.asarray(data["body_lin_vel_w"], dtype=np.float32),
                    "body_ang_vel_w": np.asarray(data["body_ang_vel_w"], dtype=np.float32),
                    "hit_frame": hit_frame,
                    "strike_pos_w": np.asarray(target.get("racket_position_m", [0.0, 0.0, 0.0]), dtype=np.float32),
                    "strike_vel_w": np.asarray(target.get("racket_velocity_mps", [0.0, 0.0, 0.0]), dtype=np.float32),
                    "strike_normal_w": normal,
                }
            )

        self.frame_z_offset = float(frame_z_offset)
        if self.frame_z_offset != 0.0:
            for a in arrays:
                a["body_pos_w"][:, :, 2] += self.frame_z_offset
                a["strike_pos_w"][2] += self.frame_z_offset

        self.ground_z_offset = 0.0
        if ground_align:
            min_z = min(float(a["body_pos_w"][:, :, 2].min()) for a in arrays)
            if min_z < 0.0:
                self.ground_z_offset = -min_z
                for a in arrays:
                    a["body_pos_w"][:, :, 2] += self.ground_z_offset
                    a["strike_pos_w"][2] += self.ground_z_offset

        self.motion_paths = motion_paths
        self.num_motions = len(arrays)
        self.motion_lengths = torch.tensor([a["joint_pos"].shape[0] for a in arrays], dtype=torch.long, device=device)
        self.time_step_total = int(self.motion_lengths.max().item())
        self.fps = int(arrays[0]["fps"])
        self._body_indexes = body_indexes

        def padded(name: str, trailing_shape: tuple[int, ...]) -> torch.Tensor:
            out = torch.zeros((self.num_motions, self.time_step_total, *trailing_shape), dtype=torch.float32)
            for i, a in enumerate(arrays):
                x = a[name]
                out[i, : x.shape[0]] = torch.from_numpy(x)
                if x.shape[0] < self.time_step_total:
                    out[i, x.shape[0] :] = out[i, x.shape[0] - 1]
            if not torch.isfinite(out).all():
                raise ValueError(f"{self.manifest_file}: non-finite values in {name}")
            return out.to(device=device)

        self.joint_pos = padded("joint_pos", arrays[0]["joint_pos"].shape[1:])
        self.joint_vel = padded("joint_vel", arrays[0]["joint_vel"].shape[1:])
        self._body_pos_w = padded("body_pos_w", arrays[0]["body_pos_w"].shape[1:])
        self._body_quat_w = padded("body_quat_w", arrays[0]["body_quat_w"].shape[1:])
        self._body_lin_vel_w = padded("body_lin_vel_w", arrays[0]["body_lin_vel_w"].shape[1:])
        self._body_ang_vel_w = padded("body_ang_vel_w", arrays[0]["body_ang_vel_w"].shape[1:])
        self.hit_frame = torch.tensor([a["hit_frame"] for a in arrays], dtype=torch.long, device=device)
        self.strike_pos_w = torch.tensor(np.stack([a["strike_pos_w"] for a in arrays]), dtype=torch.float32, device=device)
        self.strike_vel_w = torch.tensor(np.stack([a["strike_vel_w"] for a in arrays]), dtype=torch.float32, device=device)
        self.strike_normal_w = torch.tensor(
            np.stack([a["strike_normal_w"] for a in arrays]), dtype=torch.float32, device=device
        )

    @staticmethod
    def _balanced_subset(entries: list[dict], subset_size: int) -> list[dict]:
        if subset_size >= len(entries):
            return entries
        forehands = [e for e in entries if str(e.get("stroke_type", "")).lower() == "forehand"]
        backhands = [e for e in entries if str(e.get("stroke_type", "")).lower() == "backhand"]
        if forehands and backhands:
            if subset_size == 1:
                return forehands[:1]
            n_fh = subset_size // 2
            n_bh = subset_size - n_fh
            return forehands[:n_fh] + backhands[:n_bh]
        return entries[:subset_size]

    @staticmethod
    def _repo_root() -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        # .../hope_training/whole_body_tracking/training/tasks/tracking/mdp/commands.py
        return os.path.abspath(os.path.join(here, "../../../../.."))

    @classmethod
    def _workspace_root(cls) -> str:
        return os.path.abspath(os.path.join(cls._repo_root(), "../.."))

    @classmethod
    def _resolve_path(cls, path: str, manifest_dir: str | None = None) -> str:
        candidates = []
        p = os.path.expanduser(str(path))
        if os.path.isabs(p):
            candidates.append(p)
        else:
            candidates.extend(
                [
                    os.path.abspath(p),
                    os.path.join(cls._repo_root(), p),
                    os.path.join(cls._workspace_root(), p),
                ]
            )
            if manifest_dir is not None:
                candidates.append(os.path.join(manifest_dir, p))
                candidates.append(os.path.join(manifest_dir, os.path.basename(p)))
        for c in candidates:
            if os.path.isfile(c):
                return os.path.abspath(c)
        raise FileNotFoundError(f"could not resolve path '{path}', tried: {candidates}")

    def _entry_motion_path(self, entry: dict) -> str:
        manifest_dir = os.path.dirname(self.manifest_file)
        for key in ("library_motion_npz", "motion_npz"):
            value = entry.get(key)
            if value:
                try:
                    return self._resolve_path(value, manifest_dir=manifest_dir)
                except FileNotFoundError:
                    pass
        episode_id = entry.get("episode_id")
        stroke = str(entry.get("stroke_type", "")).lower()
        if episode_id and stroke:
            return self._resolve_path(os.path.join(stroke, f"{episode_id}.npz"), manifest_dir=manifest_dir)
        raise FileNotFoundError(f"no usable motion path for manifest entry: {entry}")

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, :, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, :, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, :, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, :, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        # Full motion arrays are saved in articulation body order. Body 0 is
        # the articulation root; this is distinct from cfg.body_names, which may
        # be a tracked subset for rewards/observations.
        self.motion_root_body_index = 0
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        if self.cfg.motion_manifest:
            self.motion = MotionLibraryLoader(
                self.cfg.motion_manifest,
                self.body_indexes,
                device=self.device,
                subset_size=self.cfg.manifest_subset_size,
                expected_fps=self.cfg.manifest_expected_fps,
                frame_z_offset=self.cfg.manifest_frame_z_offset,
                ground_align=self.cfg.manifest_ground_align,
                validate_stance_contract=self.cfg.validate_stance_contract,
                stance_contract_mode=self.cfg.stance_contract_mode,
            )
            self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._use_motion_library = True
            print(
                f"[MotionCommand] loaded motion manifest: {self.motion.manifest_file} "
                f"({self.motion.num_motions} motions; "
                f"forehand={len(self.motion.forehand_ids)}, backhand={len(self.motion.backhand_ids)}, "
                f"frame_z_offset={self.motion.frame_z_offset:.4f}m, "
                f"ground_z_offset={self.motion.ground_z_offset:.4f}m)",
                flush=True,
            )
        else:
            self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
            self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._use_motion_library = False
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # A task may define a static, named joint offset around the retargeted
        # motion.  Strike Stabilizer uses this for a leg ``strike-ready``
        # working point: the upper-body motion remains unchanged while hip,
        # knee, and ankle references can be scanned as one coherent plant
        # contract.  Keeping this in MotionCommand ensures reset, action
        # reference, and reference observations all see the same pose.
        self._joint_position_offset = torch.zeros(self.robot.num_joints, device=self.device)
        for joint_name, offset in dict(self.cfg.joint_position_offset).items():
            joint_ids, resolved = self.robot.find_joints([joint_name], preserve_order=True)
            if resolved != [joint_name]:
                raise ValueError(f"Unknown motion joint_position_offset joint: {joint_name!r}")
            self._joint_position_offset[int(joint_ids[0])] = float(offset)
        self._root_position_offset = torch.tensor(
            self.cfg.root_position_offset, dtype=torch.float, device=self.device
        )
        if self._root_position_offset.shape != (3,):
            raise ValueError("root_position_offset must contain exactly three values")
        # Optional terminal hold for finite task motions.  ``time_steps`` stays
        # at the final valid reference frame until the environment resets; this
        # avoids a discontinuous wrap from the end of a strike back to its
        # start. ``tail_steps`` remains observable for diagnostics.
        self.tail_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Optional physically continuous bridge used by the strike stabilizer:
        # reset from the configured ready pose, then blend upper-body targets
        # into reference frame zero before the strike phase starts.  This avoids
        # using an inconsistent "ready legs + old floating root" teleport.
        self.prelude_steps = int(self.cfg.prelude_steps)
        if self.prelude_steps < 0:
            raise ValueError("prelude_steps must be non-negative")
        self.prelude_elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Audit-only reset provenance.  These buffers are not observations and
        # do not affect control; they make it possible to verify that a task
        # configured with root randomization actually writes those sampled
        # errors into the physical reset state.
        self.last_reset_pose_offset = torch.zeros(self.num_envs, 6, device=self.device)
        self.last_reset_velocity_offset = torch.zeros(self.num_envs, 6, device=self.device)
        self.last_reset_hard_case = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.return_to_default_steps = int(self.cfg.return_to_default_steps)
        if self.return_to_default_steps < 0:
            raise ValueError("return_to_default_steps must be non-negative")
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_phase"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos_mean_abs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos_max_abs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel_mean_abs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel_max_abs"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_anchor_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["robot_anchor_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_id"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_stroke_id"] = torch.zeros(self.num_envs, device=self.device)
        for axis in ("x", "y", "z"):
            self.metrics[f"reference_anchor_pos_{axis}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"robot_anchor_pos_{axis}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"reference_anchor_lin_vel_{axis}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"robot_anchor_lin_vel_{axis}"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        if self._use_motion_library:
            reference = self.motion.joint_pos[self.motion_ids, self.time_steps]
        else:
            reference = self.motion.joint_pos[self.time_steps]
        reference = reference + self._joint_position_offset
        if self.prelude_steps > 0:
            alpha = (
                self.prelude_elapsed_steps.to(dtype=reference.dtype)
                / float(self.prelude_steps)
            ).clamp_(0.0, 1.0).unsqueeze(-1)
            reference = self.robot.data.default_joint_pos + alpha * (
                reference - self.robot.data.default_joint_pos
            )

        # A finite strike must not abruptly release the arm at its final
        # frame.  Keep the final pose long enough for the legs to absorb the
        # swing momentum, then smoothly return the whole-body reference to the
        # task's configured ready pose.  ``hold_last_frame_steps`` is the
        # first tail segment; the remaining episode time is the ready hold.
        if self.return_to_default_steps > 0:
            hold_steps = int(self.cfg.hold_last_frame_steps)
            return_elapsed = (self.tail_steps - hold_steps).clamp(
                min=0, max=self.return_to_default_steps
            ).to(dtype=reference.dtype)
            u = return_elapsed / float(self.return_to_default_steps)
            # Quintic minimum-jerk blend: position, target velocity and
            # target acceleration are all continuous at the hold/return and
            # return/ready boundaries.
            smooth_u = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)
            reference = reference + smooth_u.unsqueeze(-1) * (
                self.robot.data.default_joint_pos - reference
            )
        return reference

    @property
    def joint_vel(self) -> torch.Tensor:
        if self._use_motion_library:
            reference_vel = self.motion.joint_vel[self.motion_ids, self.time_steps]
            final_joint_pos = self.motion.joint_pos[self.motion_ids, self.time_steps]
        else:
            reference_vel = self.motion.joint_vel[self.time_steps]
            final_joint_pos = self.motion.joint_pos[self.time_steps]
        if self.return_to_default_steps <= 0:
            return reference_vel

        hold_steps = int(self.cfg.hold_last_frame_steps)
        return_elapsed = (self.tail_steps - hold_steps).clamp(
            min=0, max=self.return_to_default_steps
        ).to(dtype=reference_vel.dtype)
        u = return_elapsed / float(self.return_to_default_steps)
        # d minimum-jerk(u) / dt, using one policy/control step as dt.
        control_dt = float(self._env.cfg.decimation * self._env.cfg.sim.dt)
        smooth_rate = 30.0 * u * u * (1.0 - u) * (1.0 - u) / (
            float(self.return_to_default_steps) * control_dt
        )
        return_velocity = smooth_rate.unsqueeze(-1) * (
            self.robot.data.default_joint_pos - final_joint_pos - self._joint_position_offset
        )
        in_return_or_ready = self.tail_steps > hold_steps
        return torch.where(in_return_or_ready.unsqueeze(-1), return_velocity, reference_vel)

    @property
    def body_pos_w(self) -> torch.Tensor:
        if self._use_motion_library:
            reference = self.motion.body_pos_w[self.motion_ids, self.time_steps]
        else:
            reference = self.motion.body_pos_w[self.time_steps]
        return reference + self._env.scene.env_origins[:, None, :] + self._root_position_offset

    @property
    def body_quat_w(self) -> torch.Tensor:
        if self._use_motion_library:
            return self.motion.body_quat_w[self.motion_ids, self.time_steps]
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        if self._use_motion_library:
            return self.motion.body_lin_vel_w[self.motion_ids, self.time_steps]
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        if self._use_motion_library:
            return self.motion.body_ang_vel_w[self.motion_ids, self.time_steps]
        return self.motion.body_ang_vel_w[self.time_steps]

    def _motion_root_state_w(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return root body state from the full motion arrays.

        Do not use ``self.body_*[:, 0]`` for this. ``self.body_*`` is filtered
        by cfg.body_names and may start at torso/right arm in native-strike
        tasks.
        """
        root_idx = self.motion_root_body_index
        if self._use_motion_library:
            root_pos = self.motion._body_pos_w[self.motion_ids, self.time_steps, root_idx].clone()
            root_ori = self.motion._body_quat_w[self.motion_ids, self.time_steps, root_idx].clone()
            root_lin_vel = self.motion._body_lin_vel_w[self.motion_ids, self.time_steps, root_idx].clone()
            root_ang_vel = self.motion._body_ang_vel_w[self.motion_ids, self.time_steps, root_idx].clone()
        else:
            root_pos = self.motion._body_pos_w[self.time_steps, root_idx].clone()
            root_ori = self.motion._body_quat_w[self.time_steps, root_idx].clone()
            root_lin_vel = self.motion._body_lin_vel_w[self.time_steps, root_idx].clone()
            root_ang_vel = self.motion._body_ang_vel_w[self.time_steps, root_idx].clone()
        root_pos += self._env.scene.env_origins + self._root_position_offset
        return root_pos, root_ori, root_lin_vel, root_ang_vel

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        if self._use_motion_library:
            reference = (
                self.motion.body_pos_w[self.motion_ids, self.time_steps, self.motion_anchor_body_index]
            )
        else:
            reference = self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
        return reference + self._env.scene.env_origins + self._root_position_offset

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        if self._use_motion_library:
            return self.motion.body_quat_w[self.motion_ids, self.time_steps, self.motion_anchor_body_index]
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        if self._use_motion_library:
            return self.motion.body_lin_vel_w[self.motion_ids, self.time_steps, self.motion_anchor_body_index]
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        if self._use_motion_library:
            return self.motion.body_ang_vel_w[self.motion_ids, self.time_steps, self.motion_anchor_body_index]
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        anchor_pos_err = self.anchor_pos_w - self.robot_anchor_pos_w
        anchor_rot_err = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        anchor_lin_vel_err = self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w
        anchor_ang_vel_err = self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w

        self.metrics["error_anchor_pos"] = torch.norm(anchor_pos_err, dim=-1)
        self.metrics["error_anchor_rot"] = anchor_rot_err
        self.metrics["error_anchor_lin_vel"] = torch.norm(anchor_lin_vel_err, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(anchor_ang_vel_err, dim=-1)
        self.metrics["error_anchor_rot_deg"] = anchor_rot_err * (180.0 / math.pi)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        joint_pos_err = self.joint_pos - self.robot_joint_pos
        joint_vel_err = self.joint_vel - self.robot_joint_vel
        self.metrics["error_joint_pos"] = torch.norm(joint_pos_err, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(joint_vel_err, dim=-1)
        self.metrics["error_joint_pos_mean_abs"] = torch.mean(torch.abs(joint_pos_err), dim=-1)
        self.metrics["error_joint_pos_max_abs"] = torch.max(torch.abs(joint_pos_err), dim=-1).values
        self.metrics["error_joint_vel_mean_abs"] = torch.mean(torch.abs(joint_vel_err), dim=-1)
        self.metrics["error_joint_vel_max_abs"] = torch.max(torch.abs(joint_vel_err), dim=-1).values

        # Log anchor states in an env-origin-relative frame so cross-env averages remain meaningful.
        anchor_ref_rel = self.anchor_pos_w - self._env.scene.env_origins
        anchor_robot_rel = self.robot_anchor_pos_w - self._env.scene.env_origins
        for axis_idx, axis in enumerate(("x", "y", "z")):
            self.metrics[f"reference_anchor_pos_{axis}"] = anchor_ref_rel[:, axis_idx]
            self.metrics[f"robot_anchor_pos_{axis}"] = anchor_robot_rel[:, axis_idx]
            self.metrics[f"reference_anchor_lin_vel_{axis}"] = self.anchor_lin_vel_w[:, axis_idx]
            self.metrics[f"robot_anchor_lin_vel_{axis}"] = self.robot_anchor_lin_vel_w[:, axis_idx]

        self.metrics["reference_anchor_speed"] = torch.norm(self.anchor_lin_vel_w, dim=-1)
        self.metrics["robot_anchor_speed"] = torch.norm(self.robot_anchor_lin_vel_w, dim=-1)
        if self._use_motion_library:
            lengths = self.motion.motion_lengths[self.motion_ids].clamp(min=1)
            self.metrics["motion_phase"] = self.time_steps.float() / (lengths - 1).clamp(min=1).float()
            self.metrics["motion_id"] = self.motion_ids.float()
            self.metrics["motion_stroke_id"] = self.motion.stroke_ids[self.motion_ids].float()
        else:
            self.metrics["motion_phase"] = self.time_steps.float() / max(self.motion.time_step_total - 1, 1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if not self.cfg.sample_random_start_phase:
            self.time_steps[env_ids] = 0
            self.metrics["sampling_entropy"][:] = 0.0
            self.metrics["sampling_top1_prob"][:] = 1.0
            self.metrics["sampling_top1_bin"][:] = 0.0
            return

        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _sample_motion_ids(self, count: int) -> torch.Tensor:
        if not self._use_motion_library:
            return torch.zeros(count, dtype=torch.long, device=self.device)
        if (
            self.cfg.manifest_balance_strokes
            and len(self.motion.forehand_ids) > 0
            and len(self.motion.backhand_ids) > 0
        ):
            choose_fh = torch.rand(count, device=self.device) < 0.5
            out = torch.empty(count, dtype=torch.long, device=self.device)
            fh_pick = torch.randint(len(self.motion.forehand_ids), (int(choose_fh.sum()),), device=self.device)
            bh_pick = torch.randint(len(self.motion.backhand_ids), (int((~choose_fh).sum()),), device=self.device)
            out[choose_fh] = self.motion.forehand_ids[fh_pick]
            out[~choose_fh] = self.motion.backhand_ids[bh_pick]
            return out
        return torch.randint(self.motion.num_motions, (count,), dtype=torch.long, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self.tail_steps[env_ids] = 0
        self.prelude_elapsed_steps[env_ids] = 0
        hard_case_probability = float(self.cfg.hard_case_probability)
        if not 0.0 <= hard_case_probability <= 1.0:
            raise ValueError("hard_case_probability must be in [0, 1]")
        hard_case = torch.rand(len(env_ids), device=self.device) < hard_case_probability
        self.last_reset_hard_case[env_ids] = hard_case
        if self._use_motion_library:
            self.motion_ids[env_ids] = self._sample_motion_ids(len(env_ids))
            if torch.any(hard_case):
                hard_ids = tuple(int(x) for x in self.cfg.hard_case_motion_ids)
                if not hard_ids:
                    raise ValueError("hard_case_motion_ids must be non-empty when hard_case_probability > 0")
                if min(hard_ids) < 0 or max(hard_ids) >= self.motion.num_motions:
                    raise ValueError(
                        f"hard_case_motion_ids={hard_ids} outside motion library size {self.motion.num_motions}"
                    )
                choices = torch.tensor(hard_ids, dtype=torch.long, device=self.device)
                self.motion_ids[env_ids[hard_case]] = choices[
                    torch.randint(len(hard_ids), (int(hard_case.sum()),), device=self.device)
                ]
        self._adaptive_sampling(env_ids)
        if self._use_motion_library:
            lengths = self.motion.motion_lengths[self.motion_ids[env_ids]]
            self.time_steps[env_ids] = torch.minimum(self.time_steps[env_ids], lengths - 1)

        if self.cfg.reset_to_default_pose:
            # ``default_*`` includes the task-configured strike-ready pose.
            # It is the only valid reset source for a prelude; mixing ready
            # leg joints with a root taken from the old swing trace is not a
            # physically coherent contact state.
            root_state = self.robot.data.default_root_state.clone()
            root_pos = root_state[:, :3] + self._env.scene.env_origins
            root_ori = root_state[:, 3:7]
            root_lin_vel = root_state[:, 7:10]
            root_ang_vel = root_state[:, 10:13]
            joint_pos = self.robot.data.default_joint_pos.clone()
            joint_vel = self.robot.data.default_joint_vel.clone()
        else:
            root_pos, root_ori, root_lin_vel, root_ang_vel = self._motion_root_state_w()
            joint_pos = self.joint_pos.clone()
            joint_vel = self.joint_vel.clone()

        # Apply reset randomization after choosing its physical source.  In
        # particular, a strike-ready reset must not sample a perturbation and
        # then overwrite it with default_root_state; that silently turns a
        # configured robustness curriculum into a deterministic reset.
        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        perturb_probability = float(self.cfg.reset_perturbation_probability)
        if not 0.0 <= perturb_probability <= 1.0:
            raise ValueError("reset_perturbation_probability must be in [0, 1]")
        perturb_active = (torch.rand(len(env_ids), device=self.device) < perturb_probability) | hard_case
        rand_samples *= perturb_active.unsqueeze(-1)
        self.last_reset_pose_offset[env_ids] = rand_samples
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        rand_samples *= perturb_active.unsqueeze(-1)
        if torch.any(hard_case):
            hard_ranges = [
                self.cfg.hard_case_velocity_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ]
            hard_ranges = torch.tensor(hard_ranges, device=self.device)
            rand_samples[hard_case] = sample_uniform(
                hard_ranges[:, 0], hard_ranges[:, 1], (int(hard_case.sum()), 6), device=self.device
            )
        self.last_reset_velocity_offset[env_ids] = rand_samples
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        advance_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if self.prelude_steps > 0:
            prelude_active = self.prelude_elapsed_steps < self.prelude_steps
            self.prelude_elapsed_steps[prelude_active] += 1
            # Do not advance the finite strike reference while it is being
            # blended in.  Each environment enters the real phase-0 strike
            # only after its configured prelude completes.
            advance_mask = ~prelude_active
        hold_steps = int(self.cfg.hold_last_frame_steps)
        if hold_steps > 0:
            if self._use_motion_library:
                motion_lengths = self.motion.motion_lengths[self.motion_ids]
            else:
                motion_lengths = torch.full_like(self.time_steps, self.motion.time_step_total)
            final_steps = motion_lengths - 1
            advancing = advance_mask & (self.time_steps < final_steps)
            self.time_steps[advancing] += 1
            holding = advance_mask & ~advancing
            self.tail_steps[holding] += 1
            # A finite strike episode is terminated by the environment timeout.
            # Do not resample here: even one reference-frame wrap would create
            # an artificial post-swing impulse and invalidate the settling tail.
            env_ids = torch.empty(0, dtype=torch.long, device=self.device)
        else:
            self.time_steps[advance_mask] += 1
            if self._use_motion_library:
                motion_lengths = self.motion.motion_lengths[self.motion_ids]
                env_ids = torch.where(advance_mask & (self.time_steps >= motion_lengths))[0]
            else:
                env_ids = torch.where(advance_mask & (self.time_steps >= self.motion.time_step_total))[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str | None = None
    motion_manifest: str | None = None
    manifest_subset_size: int | None = None
    manifest_expected_fps: int = 50
    manifest_balance_strokes: bool = True
    # Explicit frame adapter for using HOPE competition-frame motions in tracking scenes whose ground is z=0.
    # HOPE competition frame uses z=0 at table surface and floor at z=-0.76, so the manifest task sets +0.76.
    manifest_frame_z_offset: float = 0.0
    manifest_ground_align: bool = False
    # Opt-in validation for prepositioned stance metadata. Fixed-base manifests
    # remain unchanged unless this is explicitly enabled.
    validate_stance_contract: bool = False
    stance_contract_mode: str | None = None
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    # Static named reference offsets.  Empty preserves the retargeted motion.
    # Strike Stabilizer scans leg-only offsets here rather than encoding a
    # joint-action template in the learned actor or reward.
    joint_position_offset: dict[str, float] = {}
    # Translational counterpart of a static joint working-point adjustment.
    # A knee-flexed ready pose normally requires a lower pelvis reference to
    # preserve foot contact; this is scanned together with joint offsets.
    root_position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sample_random_start_phase: bool = True
    # Hold a finite motion's final reference frame until the environment reset.
    # The task episode length defines the required settling-tail duration. Zero
    # preserves legacy looping behaviour.
    hold_last_frame_steps: int = 0
    # After ``hold_last_frame_steps`` at the final motion pose, blend the
    # complete joint reference back to ``robot.init_state`` over this many
    # control steps.  The remaining episode time holds that ready pose, so a
    # task can require post-return stability rather than hiding a fall at the
    # end of the strike clip.
    return_to_default_steps: int = 0
    # A pre-swing bridge from ``robot.init_state`` into motion frame zero.
    # It is disabled for legacy tracking tasks.
    prelude_steps: int = 0
    reset_to_default_pose: bool = False
    # Reset curriculum for task-conditioned stabilizers.  A hard case is a
    # physically sampled initial condition, not a hand-authored leg action.
    reset_perturbation_probability: float = 1.0
    hard_case_probability: float = 0.0
    hard_case_motion_ids: tuple[int, ...] = ()
    hard_case_velocity_range: dict[str, tuple[float, float]] = {}

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
