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
    quat_rotate_inverse,
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
        require_upper_momentum: bool = False,
        expected_root_quaternion_wxyz: Sequence[float] | None = None,
        root_quaternion_tolerance: float = 1.0e-4,
    ):
        self.manifest_file = self._resolve_path(manifest_file)
        with open(self.manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self._strict_payload_contract = bool(
            manifest.get("payload_contract_strict", False)
            or str(manifest.get("schema_version", "")).startswith("p5d3a_")
        )

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
        # Reference-conditioned P5D sampling metadata.  These fields are
        # audit/runtime metadata only; none is exposed through actor
        # observations.  Keep the loader generic so manifests from other
        # tracking tasks continue to work without a p5d2_dataset block.
        self.regions = [
            str(e.get("p5d2_dataset", {}).get("region", e.get("region", "unknown")))
            for e in entries
        ]
        self.region_to_ids = {
            region: torch.tensor([i for i, r in enumerate(self.regions) if r == region], dtype=torch.long, device=device)
            for region in sorted(set(self.regions))
        }
        raw_difficulty = []
        raw_sample_weights = []
        for e in entries:
            meta = e.get("p5d2_dataset", {})
            value = meta.get("difficulty_weight", meta.get("projection_max_rad_after_reoptimization", 0.0))
            try:
                raw_difficulty.append(max(0.0, float(value)))
            except (TypeError, ValueError):
                raw_difficulty.append(0.0)
            sample_weight = e.get("sample_weight", 1.0)
            try:
                sample_weight = float(sample_weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid sample_weight={sample_weight!r} in motion manifest entry") from exc
            if not np.isfinite(sample_weight) or sample_weight < 0.0:
                raise ValueError(f"sample_weight must be finite and non-negative, got {sample_weight!r}")
            raw_sample_weights.append(sample_weight)
        difficulty = torch.tensor(raw_difficulty, dtype=torch.float32, device=device)
        self.difficulty_weights = 1.0 + difficulty / 0.01
        self.sample_weights = torch.tensor(raw_sample_weights, dtype=torch.float32, device=device)
        if not torch.any(self.sample_weights > 0.0):
            raise ValueError(f"{self.manifest_file}: all motion sample_weight values are zero")

        arrays: list[dict[str, np.ndarray | float | int]] = []
        motion_paths: list[str] = []
        expected_root_q = None
        if expected_root_quaternion_wxyz is not None:
            expected_root_q = np.asarray(expected_root_quaternion_wxyz, dtype=np.float64).reshape(-1)
            if expected_root_q.shape != (4,) or not np.isfinite(expected_root_q).all():
                raise ValueError("expected_root_quaternion_wxyz must be a finite 4-vector")
            expected_root_q /= max(float(np.linalg.norm(expected_root_q)), 1.0e-12)
            if root_quaternion_tolerance <= 0.0:
                raise ValueError("root_quaternion_tolerance must be positive")
        for entry in entries:
            motion_path = self._entry_motion_path(entry)
            motion_paths.append(motion_path)
            data = np.load(motion_path)
            fps = int(data["fps"])
            if expected_fps is not None and fps != int(expected_fps):
                raise ValueError(f"{motion_path}: fps={fps}, expected {expected_fps}")
            joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
            body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
            if expected_root_q is not None:
                root_q = np.asarray(data["body_quat_w"], dtype=np.float64)[0, 0]
                root_q /= max(float(np.linalg.norm(root_q)), 1.0e-12)
                root_error = min(float(np.linalg.norm(root_q - expected_root_q)), float(np.linalg.norm(root_q + expected_root_q)))
                if root_error > float(root_quaternion_tolerance):
                    raise ValueError(
                        f"{motion_path}: root quaternion {root_q.tolist()} disagrees with the task root "
                        f"{expected_root_q.tolist()} (error={root_error:.3e} > tolerance={root_quaternion_tolerance:.3e})"
                    )
            if joint_pos.shape[-1] != 31:
                raise ValueError(f"{motion_path}: joint_pos shape {joint_pos.shape}, expected [...,31]")
            if body_pos_w.shape[-2] != 32:
                raise ValueError(f"{motion_path}: body_pos_w shape {body_pos_w.shape}, expected [...,32,3]")
            hit_frame = int(entry.get("hit_event", {}).get("motion_hit_frame", round(0.46 * (joint_pos.shape[0] - 1))))
            if not (0 <= hit_frame < joint_pos.shape[0]):
                raise ValueError(f"{motion_path}: invalid hit_frame={hit_frame} for length={joint_pos.shape[0]}")

            target = entry.get("strike_target", {})

            # P5 scene-placed reference payloads carry the authoritative goal
            # in the immutable initial-base-heading frame.  Older manifests
            # also contain a ``strike_target`` dictionary, but for generated
            # forehand clips that dictionary was originally authored in the
            # source mocap scene and can be metres away from the placed robot.
            # Prefer the canonical payload when all of its frame metadata is
            # present; otherwise retain the legacy manifest target contract.
            canonical_scene_goal = (
                "canonical_goal_position_b0_m" in data.files
                and "canonical_goal_linear_velocity_b0_mps" in data.files
                and "canonical_goal_normal_b0" in data.files
                and "scene_root_anchor_w_m" in data.files
            )
            # The dense A3 candidate bank uses the newer compact contract:
            # canonical_* are expressed in the initial root-heading frame,
            # while body_pos_w/body_quat_w contain the reference root pose.
            # Keep this separate from the older scene-placed canonical_goal_*
            # contract; falling through to the manifest's raw strike_target
            # would incorrectly interpret a root-relative point as world data.
            canonical_root_goal = (
                "canonical_position" in data.files
                and "canonical_velocity" in data.files
                and "canonical_normal" in data.files
            )

            def _rotate_z_np(vector: np.ndarray, heading_rad: float) -> np.ndarray:
                c = float(np.cos(heading_rad))
                s = float(np.sin(heading_rad))
                x, y, z = [float(v) for v in np.asarray(vector).reshape(3)]
                return np.asarray([c * x - s * y, s * x + c * y, z], dtype=np.float32)

            def _yaw_from_quat_wxyz(quaternion: np.ndarray) -> float:
                w, x, y, z = [float(v) for v in np.asarray(quaternion).reshape(4)]
                denominator = 1.0 - 2.0 * (y * y + z * z)
                numerator = 2.0 * (w * z + x * y)
                return float(np.arctan2(numerator, denominator))

            if canonical_scene_goal:
                anchor = np.asarray(data["scene_root_anchor_w_m"], dtype=np.float32).reshape(3)
                heading = float(np.asarray(data.get("scene_root_heading_w_rad", [0.0])).reshape(-1)[0])
                canonical_position_b = np.asarray(data["canonical_goal_position_b0_m"], dtype=np.float32).reshape(3)
                canonical_velocity_b = np.asarray(data["canonical_goal_linear_velocity_b0_mps"], dtype=np.float32).reshape(3)
                canonical_normal_b = np.asarray(data["canonical_goal_normal_b0"], dtype=np.float32).reshape(3)
                strike_position = anchor + _rotate_z_np(canonical_position_b, heading)
                strike_velocity = _rotate_z_np(canonical_velocity_b, heading)
                strike_normal = _rotate_z_np(canonical_normal_b, heading)
                strike_position_b0 = np.zeros(3, dtype=np.float32)
                strike_velocity_b0 = np.zeros(3, dtype=np.float32)
                strike_normal_b0 = np.zeros(3, dtype=np.float32)
                strike_target_is_root_relative = False
            elif canonical_root_goal:
                root_anchor = np.asarray(body_pos_w[0, 0], dtype=np.float32).reshape(3)
                root_heading = _yaw_from_quat_wxyz(np.asarray(data["body_quat_w"])[0, 0])
                strike_position_b0 = np.asarray(data["canonical_position"], dtype=np.float32).reshape(3)
                strike_velocity_b0 = np.asarray(data["canonical_velocity"], dtype=np.float32).reshape(3)
                strike_normal_b0 = np.asarray(data["canonical_normal"], dtype=np.float32).reshape(3)
                strike_position = root_anchor + _rotate_z_np(strike_position_b0, root_heading)
                strike_velocity = _rotate_z_np(strike_velocity_b0, root_heading)
                strike_normal = _rotate_z_np(strike_normal_b0, root_heading)
                strike_target_is_root_relative = True
            else:
                strike_position = np.asarray(target.get("racket_position_m", [0.0, 0.0, 0.0]), dtype=np.float32)
                strike_velocity = np.asarray(target.get("racket_velocity_mps", [0.0, 0.0, 0.0]), dtype=np.float32)
                strike_normal = np.asarray(target.get("racket_normal_w", [0.0, 0.0, 1.0]), dtype=np.float32)
                strike_position_b0 = np.zeros(3, dtype=np.float32)
                strike_velocity_b0 = np.zeros(3, dtype=np.float32)
                strike_normal_b0 = np.zeros(3, dtype=np.float32)
                strike_target_is_root_relative = False

            normal = strike_normal
            n = float(np.linalg.norm(normal))
            if not np.isfinite(n) or n < 1e-6:
                raise ValueError(f"{motion_path}: invalid racket_normal_w={normal}")
            normal = normal / n
            if require_upper_momentum:
                required = {"upper_momentum_pelvis", "upper_mass_kg", "upper_length_scale_m"}
                missing = sorted(required - set(data.files))
                if missing:
                    raise ValueError(f"{motion_path}: missing required canonical momentum fields {missing}")
                upper_momentum = np.asarray(data["upper_momentum_pelvis"], dtype=np.float32)
                if upper_momentum.shape != (joint_pos.shape[0], 6):
                    raise ValueError(
                        f"{motion_path}: upper_momentum_pelvis shape {upper_momentum.shape}, "
                        f"expected {(joint_pos.shape[0], 6)}"
                    )
                upper_mass = float(np.asarray(data["upper_mass_kg"]).reshape(-1)[0])
                upper_length_scale = float(np.asarray(data["upper_length_scale_m"]).reshape(-1)[0])
                if not np.isfinite(upper_momentum).all() or upper_mass <= 0.0 or upper_length_scale <= 0.0:
                    raise ValueError(f"{motion_path}: invalid canonical momentum metadata")
            else:
                upper_momentum = np.zeros((joint_pos.shape[0], 6), dtype=np.float32)
                upper_mass = 1.0
                upper_length_scale = 1.0

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
                    "strike_pos_w": strike_position,
                    "strike_vel_w": strike_velocity,
                    "strike_normal_w": normal,
                    "strike_pos_b0": strike_position_b0,
                    "strike_vel_b0": strike_velocity_b0,
                    "strike_normal_b0": strike_normal_b0,
                    "strike_target_is_root_relative": strike_target_is_root_relative,
                    "upper_momentum_pelvis": upper_momentum,
                    "upper_mass_kg": upper_mass,
                    "upper_length_scale_m": upper_length_scale,
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
        self.upper_momentum_pelvis = padded("upper_momentum_pelvis", (6,))
        self.upper_mass_kg = torch.tensor(
            [a["upper_mass_kg"] for a in arrays], dtype=torch.float32, device=device
        )
        self.upper_length_scale_m = torch.tensor(
            [a["upper_length_scale_m"] for a in arrays], dtype=torch.float32, device=device
        )
        self.has_canonical_upper_momentum = bool(require_upper_momentum)
        self.hit_frame = torch.tensor([a["hit_frame"] for a in arrays], dtype=torch.long, device=device)
        self.strike_pos_w = torch.tensor(np.stack([a["strike_pos_w"] for a in arrays]), dtype=torch.float32, device=device)
        self.strike_vel_w = torch.tensor(np.stack([a["strike_vel_w"] for a in arrays]), dtype=torch.float32, device=device)
        self.strike_normal_w = torch.tensor(
            np.stack([a["strike_normal_w"] for a in arrays]), dtype=torch.float32, device=device
        )
        self.strike_pos_b0 = torch.tensor(
            np.stack([a["strike_pos_b0"] for a in arrays]), dtype=torch.float32, device=device
        )
        self.strike_vel_b0 = torch.tensor(
            np.stack([a["strike_vel_b0"] for a in arrays]), dtype=torch.float32, device=device
        )
        self.strike_normal_b0 = torch.tensor(
            np.stack([a["strike_normal_b0"] for a in arrays]), dtype=torch.float32, device=device
        )
        # The canonical motion payload stores the geometric red-face normal
        # (+Y of the racket frame) for both stroke families.  The motion
        # family itself is still distinct, but the contact-face semantic must
        # be applied explicitly: forehand=red (+1), backhand=black (-1).
        self.face_sign = torch.where(
            self.stroke_ids == 0,
            torch.ones_like(self.stroke_ids, dtype=torch.float32),
            torch.where(
                self.stroke_ids == 1,
                -torch.ones_like(self.stroke_ids, dtype=torch.float32),
                torch.ones_like(self.stroke_ids, dtype=torch.float32),
            ),
        )
        self.strike_normal_w = self.strike_normal_w * self.face_sign.unsqueeze(-1)
        self.strike_normal_b0 = self.strike_normal_b0 * self.face_sign.unsqueeze(-1)
        self.strike_target_is_root_relative = torch.tensor(
            [a["strike_target_is_root_relative"] for a in arrays],
            dtype=torch.bool,
            device=device,
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
        # ``library_motion_npz`` used to silently win over ``motion_npz``.
        # That made a generated-candidate manifest replay its provenance clip
        # whenever the generator forgot to rewrite both fields.  A candidate
        # screen can therefore look numerically plausible while testing the
        # wrong trajectory.  If both explicit paths resolve, require them to
        # identify the same payload and fail closed on disagreement.
        explicit_paths: dict[str, str] = {}
        for key in ("library_motion_npz", "motion_npz"):
            value = entry.get(key)
            if value:
                try:
                    explicit_paths[key] = self._resolve_path(value, manifest_dir=manifest_dir)
                except FileNotFoundError:
                    # Qualification manifests are deliberately self-contained,
                    # but their provenance keeps the source machine's absolute
                    # ``motion_npz`` path.  When replaying such a package on a
                    # different host, prefer its colocated payload over the
                    # legacy stroke/episode fallback below.  This never masks
                    # a valid explicit path: it is considered only after that
                    # path failed to resolve.
                    packaged_path = os.path.join(
                        manifest_dir,
                        "motion_npz",
                        os.path.basename(str(value)),
                    )
                    if os.path.isfile(packaged_path):
                        explicit_paths[key] = os.path.abspath(packaged_path)
        # Legacy packaged manifests intentionally keep a provenance
        # ``motion_npz`` while ``library_motion_npz`` points at the colocated
        # replay payload.  Preserve that established precedence.  Candidate
        # manifests opt into strict mode through their schema or by carrying
        # an explicit canonical payload field; those must never disagree.
        strict_entry = self._strict_payload_contract or bool(entry.get("canonical_motion_npz"))
        if len(explicit_paths) == 2 and strict_entry:
            library_path = os.path.realpath(explicit_paths["library_motion_npz"])
            motion_path = os.path.realpath(explicit_paths["motion_npz"])
            if library_path != motion_path:
                raise ValueError(
                    "motion manifest entry has conflicting explicit payloads: "
                    f"library_motion_npz={library_path!r} != motion_npz={motion_path!r}. "
                    "The loader refuses to guess which trajectory is authoritative."
                )
        for key in ("library_motion_npz", "motion_npz"):
            if key in explicit_paths:
                return explicit_paths[key]
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
        # Keep an explicit environment handle for the unified next-action
        # recovery query; IsaacLab CommandTerm versions differ in whether
        # they expose the protected handle under the same name.
        self._env = env

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
                require_upper_momentum=self.cfg.require_upper_momentum,
                expected_root_quaternion_wxyz=self.cfg.expected_root_quaternion_wxyz or None,
                root_quaternion_tolerance=self.cfg.root_quaternion_tolerance,
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
        # V1.3B CompletePriors may rephase a private teacher so its physical
        # hit matches the public short time-to-hit.  These are bookkeeping
        # only; they are never exposed to the public actor.
        self.v13b_teacher_start_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.v13b_teacher_hit_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.v13b_teacher_rephased = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.v13b_upper_prior_wrap_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Monotonic within-episode strike generation.  Episode resets return
        # this to zero; begin_next_shot() increments it without touching the
        # physical articulation state.  Runtime controllers use this explicit
        # signal instead of mistaking a reference phase rewind for a reset.
        self.shot_cycle = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
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
        # V27 may use a task-local bent READY without mutating the robot asset
        # default shared by frozen V25/V26. Empty overrides preserve the exact
        # historical reset, bridge and finite-return contract.
        self._ready_joint_position_overrides: dict[int, float] = {}
        for joint_name, value in dict(self.cfg.ready_joint_positions).items():
            joint_ids, resolved = self.robot.find_joints(
                [joint_name], preserve_order=True
            )
            if resolved != [joint_name]:
                raise ValueError(
                    f"Unknown ready_joint_positions joint: {joint_name!r}"
                )
            self._ready_joint_position_overrides[int(joint_ids[0])] = float(value)
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
        self.prelude_settle_steps = int(self.cfg.prelude_settle_steps)
        self.prelude_launch_steps = int(self.cfg.prelude_launch_steps)
        if not 0 <= self.prelude_settle_steps <= self.prelude_steps:
            raise ValueError("prelude_settle_steps must be within [0, prelude_steps]")
        if self.prelude_launch_steps < 0:
            raise ValueError("prelude_launch_steps must be non-negative")
        self._prelude_waist_pitch_id: int | None = None
        if self.cfg.prelude_waist_pitch_anchor_rad is not None:
            joint_ids, resolved = self.robot.find_joints(["waist_pitch_joint"], preserve_order=True)
            if resolved != ["waist_pitch_joint"]:
                raise ValueError("Prelude waist-pitch anchor joint could not be resolved")
            self._prelude_waist_pitch_id = int(joint_ids[0])
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
    def ready_joint_pos(self) -> torch.Tensor:
        """Return the live lower-body READY plus task-local joint overrides.

        The staggered-stance reset event updates ``default_joint_pos`` after
        command construction.  Rebuilding from that live tensor preserves the
        validated lower-body handoff while V27 replaces only named arm joints.
        """
        ready = self.robot.data.default_joint_pos.clone()
        for joint_id, value in self._ready_joint_position_overrides.items():
            ready[:, joint_id] = value
        return ready

    @property
    def joint_pos(self) -> torch.Tensor:
        if self._use_motion_library:
            reference = self.motion.joint_pos[self.motion_ids, self.time_steps]
        else:
            reference = self.motion.joint_pos[self.time_steps]
        reference = reference + self._joint_position_offset
        if self.prelude_steps > 0:
            in_prelude = self.prelude_elapsed_steps < self.prelude_steps
            if not self._prelude_position_bridge_enabled():
                # Exact legacy path: the original reference is linearly
                # blended throughout prelude.  Established checkpoints were
                # trained under this timing and must not silently change.
                alpha = (
                    self.prelude_elapsed_steps.to(dtype=reference.dtype)
                    / float(self.prelude_steps)
                ).clamp_(0.0, 1.0).unsqueeze(-1)
                return self.ready_joint_pos + alpha * (
                    reference - self.ready_joint_pos
                )
            endpoint = self._prelude_endpoint(reference)
            if self.cfg.prelude_quintic_hermite:
                bridge, _ = self._prelude_quintic_bridge(reference.dtype)
            else:
                alpha, _ = self._prelude_blend(reference.dtype)
                bridge = self.ready_joint_pos + alpha.unsqueeze(-1) * (
                    endpoint - self.ready_joint_pos
                )
            reference = torch.where(in_prelude.unsqueeze(-1), bridge, reference)

            # After the bridge reaches a zero-velocity safe anchor, blend the
            # first known swing frames in rather than jumping directly from
            # frame zero to frame one.  Motion phase still advances normally,
            # so hit timing and task targets remain unchanged.
            launch_steps = self.prelude_launch_steps
            launch_active = (
                (~in_prelude)
                & (self.tail_steps == 0)
                & (self.time_steps < launch_steps)
            )
            if launch_steps > 0:
                u = (self.time_steps.to(dtype=reference.dtype) / float(launch_steps)).clamp(0.0, 1.0)
                smooth = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)
                launch = endpoint + smooth.unsqueeze(-1) * (reference - endpoint)
                reference = torch.where(launch_active.unsqueeze(-1), launch, reference)

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
                self.ready_joint_pos - reference
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
        reference_pos = self._initial_joint_reference_at_current_phase()
        # Position shaping and velocity shaping are intentionally independent.
        # The legacy contract keeps its original reference velocity throughout
        # prelude unless this explicit opt-in is set.
        if self.prelude_steps > 0 and self.cfg.prelude_continuous_velocity_reference:
            in_prelude = self.prelude_elapsed_steps < self.prelude_steps
            endpoint = self._prelude_endpoint(self._initial_joint_reference())
            if self.cfg.prelude_quintic_hermite:
                _, prelude_velocity = self._prelude_quintic_bridge(
                    reference_vel.dtype
                )
            else:
                _, blend_rate = self._prelude_blend(reference_vel.dtype)
                prelude_velocity = blend_rate.unsqueeze(-1) * (
                    endpoint - self.ready_joint_pos
                )
            launch_steps = self.prelude_launch_steps
            launch_active = (
                (~in_prelude)
                & (self.tail_steps == 0)
                & (self.time_steps < launch_steps)
            )
            if launch_steps > 0:
                control_dt = float(self._env.cfg.decimation * self._env.cfg.sim.dt)
                u = (self.time_steps.to(dtype=reference_vel.dtype) / float(launch_steps)).clamp(0.0, 1.0)
                smooth = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)
                smooth_rate = 30.0 * u * u * (1.0 - u) * (1.0 - u) / (float(launch_steps) * control_dt)
                launch_velocity = (
                    smooth.unsqueeze(-1) * reference_vel
                    + smooth_rate.unsqueeze(-1) * (reference_pos - endpoint)
                )
                reference_vel = torch.where(launch_active.unsqueeze(-1), launch_velocity, reference_vel)
            reference_vel = torch.where(in_prelude.unsqueeze(-1), prelude_velocity, reference_vel)

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
            self.ready_joint_pos - final_joint_pos - self._joint_position_offset
        )
        in_return_or_ready = self.tail_steps > hold_steps
        return torch.where(in_return_or_ready.unsqueeze(-1), return_velocity, reference_vel)

    def _initial_joint_reference(self) -> torch.Tensor:
        """Return frame-zero motion joints in runtime articulation order."""
        if self._use_motion_library:
            reference = self.motion.joint_pos[self.motion_ids, 0]
        else:
            reference = self.motion.joint_pos[0].unsqueeze(0).expand(self.num_envs, -1)
        return reference + self._joint_position_offset

    def _prelude_position_bridge_enabled(self) -> bool:
        """Whether this task intentionally replaces the legacy linear prelude."""
        return bool(
            self.cfg.prelude_minimum_jerk
            or self.cfg.prelude_quintic_hermite
            or self.prelude_settle_steps > 0
            or self.prelude_launch_steps > 0
            or self.cfg.prelude_waist_pitch_anchor_rad is not None
        )

    def _initial_joint_reference_at_current_phase(self) -> torch.Tensor:
        """Return current raw motion joints, including configured static offsets."""
        if self._use_motion_library:
            reference = self.motion.joint_pos[self.motion_ids, self.time_steps]
        else:
            reference = self.motion.joint_pos[self.time_steps]
        return reference + self._joint_position_offset

    def _prelude_endpoint(self, frame_zero_reference: torch.Tensor) -> torch.Tensor:
        """Build the safe, per-joint prelude endpoint without altering the motion clip."""
        endpoint = frame_zero_reference.clone()
        if self._prelude_waist_pitch_id is not None:
            anchor = float(self.cfg.prelude_waist_pitch_anchor_rad)
            endpoint[:, self._prelude_waist_pitch_id] = torch.minimum(
                endpoint[:, self._prelude_waist_pitch_id],
                torch.full((self.num_envs,), anchor, dtype=endpoint.dtype, device=self.device),
            )
        return endpoint

    def _prelude_blend(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Return bridge blend and d(blend)/dt with an optional ready settle."""
        transition_steps = self.prelude_steps - self.prelude_settle_steps
        if transition_steps <= 0:
            zero = torch.zeros(self.num_envs, dtype=dtype, device=self.device)
            return zero, zero
        elapsed = (self.prelude_elapsed_steps - self.prelude_settle_steps).clamp(
            min=0, max=transition_steps
        ).to(dtype=dtype)
        u = elapsed / float(transition_steps)
        if not self.cfg.prelude_minimum_jerk:
            blend = u
            rate = torch.full_like(u, 1.0 / (float(transition_steps) * self._env.cfg.decimation * self._env.cfg.sim.dt))
            rate = torch.where((u <= 0.0) | (u >= 1.0), torch.zeros_like(rate), rate)
            return blend, rate
        blend = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)
        control_dt = float(self._env.cfg.decimation * self._env.cfg.sim.dt)
        rate = 30.0 * u * u * (1.0 - u) * (1.0 - u) / (float(transition_steps) * control_dt)
        return blend, rate

    def _prelude_quintic_bridge(
        self, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match READY q/0/0 to frame-zero q/qdot/qddot analytically."""
        if self.prelude_steps < 2:
            raise ValueError("prelude_quintic_hermite requires at least 2 steps")
        if self.prelude_settle_steps != 0 or self.prelude_launch_steps != 0:
            raise ValueError(
                "prelude_quintic_hermite owns the complete bridge; "
                "settle/launch steps must be zero"
            )
        if self._use_motion_library:
            positions = self.motion.joint_pos[self.motion_ids, :3]
        else:
            positions = self.motion.joint_pos[:3].unsqueeze(0).expand(
                self.num_envs, -1, -1
            )
        positions = positions.to(dtype=dtype)
        positions = positions + self._joint_position_offset.view(1, 1, -1)
        endpoint = self._prelude_endpoint(positions[:, 0])
        control_dt = float(self._env.cfg.decimation * self._env.cfg.sim.dt)
        duration = float(self.prelude_steps) * control_dt
        endpoint_velocity = (positions[:, 1] - positions[:, 0]) / control_dt
        endpoint_acceleration = (
            positions[:, 2] - 2.0 * positions[:, 1] + positions[:, 0]
        ) / (control_dt * control_dt)

        matrix = torch.tensor(
            (
                (duration**3, duration**4, duration**5),
                (3.0 * duration**2, 4.0 * duration**3, 5.0 * duration**4),
                (6.0 * duration, 12.0 * duration**2, 20.0 * duration**3),
            ),
            dtype=dtype,
            device=self.device,
        )
        rhs = torch.stack(
            (
                endpoint - self.ready_joint_pos.to(dtype=dtype),
                endpoint_velocity,
                endpoint_acceleration,
            ),
            dim=1,
        )
        coefficients = torch.einsum(
            "ij,bjk->bik", torch.linalg.inv(matrix), rhs
        )
        a3, a4, a5 = coefficients.unbind(dim=1)
        elapsed = self.prelude_elapsed_steps.to(dtype=dtype) * control_dt
        elapsed = elapsed.clamp(min=0.0, max=duration).unsqueeze(-1)
        position = (
            self.ready_joint_pos.to(dtype=dtype)
            + a3 * elapsed**3
            + a4 * elapsed**4
            + a5 * elapsed**5
        )
        velocity = (
            3.0 * a3 * elapsed**2
            + 4.0 * a4 * elapsed**3
            + 5.0 * a5 * elapsed**4
        )
        return position, velocity

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
        if bool(getattr(self._env, "v13b_private_motion_disabled", False)):
            # Do not dereference motion tensors in the final deployment-path
            # portion of V1.3B training.  Keep existing metric buffers finite
            # for generic loggers without inventing a new reference phase.
            for value in self.metrics.values():
                if isinstance(value, torch.Tensor):
                    value.zero_()
            return
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
        fixed_motion_id = self.cfg.fixed_motion_id
        if not self._use_motion_library:
            if fixed_motion_id not in (None, 0):
                raise ValueError(
                    "fixed_motion_id requires a motion manifest, or must be 0 for a single-motion command"
                )
            return torch.zeros(count, dtype=torch.long, device=self.device)
        if fixed_motion_id is not None:
            motion_id = int(fixed_motion_id)
            if not 0 <= motion_id < self.motion.num_motions:
                raise ValueError(
                    f"fixed_motion_id={motion_id} is outside [0, {self.motion.num_motions})"
                )
            return torch.full((count,), motion_id, dtype=torch.long, device=self.device)

        def weighted_pick(candidate_ids: torch.Tensor, pick_count: int, extra_weights: torch.Tensor | None = None):
            if pick_count <= 0:
                return torch.empty(0, dtype=torch.long, device=self.device)
            if candidate_ids.numel() == 0:
                raise RuntimeError("cannot sample a motion from an empty candidate set")
            weights = self.motion.sample_weights[candidate_ids]
            if extra_weights is not None:
                weights = weights * extra_weights
            if not torch.isfinite(weights).all() or not torch.any(weights > 0.0):
                return candidate_ids[torch.randint(candidate_ids.numel(), (pick_count,), device=self.device)]
            return candidate_ids[torch.multinomial(weights, pick_count, replacement=True)]

        mode = str(getattr(self.cfg, "reference_sampling_mode", "uniform")).strip().lower()
        if mode not in {"uniform", "balanced_by_region", "difficulty_weighted", "curriculum"}:
            raise ValueError(
                "reference_sampling_mode must be one of uniform, balanced_by_region, "
                f"difficulty_weighted, curriculum; got {mode!r}"
            )
        candidate_ids = torch.arange(self.motion.num_motions, device=self.device, dtype=torch.long)
        if mode == "curriculum":
            sizes = tuple(int(x) for x in getattr(self.cfg, "reference_curriculum_sizes", ()))
            stage = int(getattr(self.cfg, "reference_curriculum_stage", 0))
            if sizes:
                if stage < 0 or stage >= len(sizes):
                    raise ValueError(f"reference_curriculum_stage={stage} outside {len(sizes)} stages")
                active_count = min(max(1, sizes[stage]), self.motion.num_motions)
                candidate_ids = candidate_ids[:active_count]
            # Curriculum stages use region balancing over the active prefix;
            # this prevents an easy anchor prefix from monopolising Stage 1.
            mode = "balanced_by_region"
        if mode == "balanced_by_region":
            active_regions = {}
            for region, ids in self.motion.region_to_ids.items():
                ids = ids[ids < candidate_ids.numel()]
                if ids.numel() > 0:
                    active_regions[region] = ids
            if active_regions:
                region_names = tuple(sorted(active_regions))
                chosen_regions = torch.randint(len(region_names), (count,), device=self.device)
                out = torch.empty(count, dtype=torch.long, device=self.device)
                for ridx, region in enumerate(region_names):
                    mask = chosen_regions == ridx
                    n = int(mask.sum())
                    if n:
                        ids = active_regions[region]
                        out[mask] = weighted_pick(ids, n)
                return out
        if mode == "difficulty_weighted":
            return weighted_pick(candidate_ids, count, self.motion.difficulty_weights[candidate_ids])
        if candidate_ids.numel() != self.motion.num_motions:
            return weighted_pick(candidate_ids, count)
        if (
            self.cfg.manifest_balance_strokes
            and len(self.motion.forehand_ids) > 0
            and len(self.motion.backhand_ids) > 0
        ):
            choose_fh = torch.rand(count, device=self.device) < 0.5
            out = torch.empty(count, dtype=torch.long, device=self.device)
            out[choose_fh] = weighted_pick(self.motion.forehand_ids, int(choose_fh.sum()))
            out[~choose_fh] = weighted_pick(self.motion.backhand_ids, int((~choose_fh).sum()))
            return out
        return weighted_pick(candidate_ids, count)

    def select_nearest_strike_motion_ids(
        self, target_position_b: torch.Tensor | Sequence[float] | Sequence[Sequence[float]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose the admitted control anchor nearest an external target.

        The input is a racket position in the base-yaw frame at command
        receipt.  Each candidate is expressed in that same frame using its
        own reference root pose at motion step zero.  A task can add a
        measured stabilized-plant offset and mask unsafe motions through the
        external-control fields in :class:`MotionCommandCfg`; absent those
        fields, this is exactly the legacy manifest-anchor selector.
        Returns one motion id and its anchor distance per target row, plus the
        complete ``[motion, 3]`` control-anchor table for diagnostics.
        """
        if not self._use_motion_library:
            raise RuntimeError("nearest-anchor selection requires a motion manifest")
        target = torch.as_tensor(target_position_b, dtype=torch.float32, device=self.device)
        if target.shape == (3,):
            target = target.unsqueeze(0)
        if target.ndim != 2 or target.shape[1] != 3:
            raise ValueError(
                "target_position_b must have shape (3,) or (N, 3) for nearest-anchor selection"
            )
        if not torch.isfinite(target).all():
            raise ValueError("target_position_b must contain only finite values")

        root_pos = self.motion._body_pos_w[:, 0, self.motion_root_body_index]
        root_pos = root_pos + self._root_position_offset.unsqueeze(0)
        root_quat = self.motion._body_quat_w[:, 0, self.motion_root_body_index]
        anchors_b = quat_rotate_inverse(
            yaw_quat(root_quat), self.motion.strike_pos_w - root_pos
        )
        offsets = torch.as_tensor(
            self.cfg.external_control_anchor_offset_b_by_motion,
            dtype=torch.float32,
            device=self.device,
        )
        if offsets.numel() > 0:
            if offsets.shape != anchors_b.shape or not torch.isfinite(offsets).all():
                raise ValueError(
                    "external_control_anchor_offset_b_by_motion must have finite "
                    "shape (num_manifest_motions, 3)"
                )
            anchors_b = anchors_b + offsets
        enabled = torch.as_tensor(
            self.cfg.external_control_anchor_enabled_by_motion,
            dtype=torch.bool,
            device=self.device,
        )
        if enabled.numel() == 0:
            enabled = torch.ones(self.motion.num_motions, dtype=torch.bool, device=self.device)
        elif enabled.shape != (self.motion.num_motions,):
            raise ValueError(
                "external_control_anchor_enabled_by_motion must have one entry "
                "per manifest motion"
            )
        if not torch.any(enabled):
            raise ValueError("external target selector has no admitted control anchors")
        distances = torch.linalg.vector_norm(
            target.unsqueeze(1) - anchors_b.unsqueeze(0), dim=-1
        )
        # An unsafe/unvalidated motion must never win an otherwise-nearest
        # target selection.  Its unmasked anchor remains in the diagnostics
        # table so the rejection can still be explained to callers.
        distances[:, ~enabled] = float("inf")
        nearest_distance, motion_ids = torch.min(distances, dim=1)
        return motion_ids, nearest_distance, anchors_b

    def begin_next_shot(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        motion_ids: Sequence[int] | torch.Tensor | int | None = None,
    ) -> None:
        """Start another finite strike without resetting robot physics.

        This is intentionally separate from ``_resample_command``: that reset
        path writes root and joint state into simulation, which would hide
        inter-shot instability.  Callers must first verify that the robot is
        settled and ready for a new strike.
        """
        env_ids_tensor = torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        ).flatten()
        if env_ids_tensor.numel() == 0:
            return
        if torch.any((env_ids_tensor < 0) | (env_ids_tensor >= self.num_envs)):
            raise IndexError("begin_next_shot env_ids are outside the environment batch")
        if not self._use_motion_library and motion_ids is not None:
            requested = torch.as_tensor(
                motion_ids, dtype=torch.long, device=self.device
            ).flatten()
            if torch.any(requested != 0):
                raise ValueError("Single-motion commands only accept motion_id=0")
        if self._use_motion_library:
            if motion_ids is None:
                selected = self._sample_motion_ids(env_ids_tensor.numel())
            else:
                selected = torch.as_tensor(
                    motion_ids, dtype=torch.long, device=self.device
                ).flatten()
                if selected.numel() == 1:
                    selected = selected.expand(env_ids_tensor.numel())
                if selected.numel() != env_ids_tensor.numel():
                    raise ValueError(
                        "begin_next_shot requires one motion_id per environment"
                    )
                if torch.any(
                    (selected < 0) | (selected >= self.motion.num_motions)
                ):
                    raise ValueError(
                        "begin_next_shot motion_ids are outside the motion library"
                    )
            self.motion_ids[env_ids_tensor] = selected

        self.time_steps[env_ids_tensor] = 0
        self.tail_steps[env_ids_tensor] = 0
        self.prelude_elapsed_steps[env_ids_tensor] = 0
        self.shot_cycle[env_ids_tensor] += 1

    def configure_v13b_episode_strike(
        self, env_ids: Sequence[int] | torch.Tensor, teacher_start_frames: torch.Tensor
    ) -> None:
        """Align teacher playback to a latched V1.3B public strike event.

        ``teacher_start_frame`` is selected so that
        ``(hit_frame - start_frame) / fps`` equals the public sampled
        time-to-hit within one motion frame.  The legacy READY prelude is
        deliberately skipped: it is not added to either clock and therefore
        cannot be double counted.
        """
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        starts = torch.as_tensor(teacher_start_frames, dtype=torch.long, device=self.device).flatten()
        if starts.numel() != ids.numel():
            raise ValueError("teacher_start_frames must have one entry per environment")
        if not self._use_motion_library:
            raise RuntimeError("V1.3B teacher rephasing requires a motion manifest")
        lengths = self.motion.motion_lengths[self.motion_ids[ids]]
        starts = torch.minimum(torch.clamp(starts, min=0), lengths - 1)
        self.time_steps[ids] = starts
        self.v13b_teacher_start_frame[ids] = starts
        self.v13b_teacher_hit_frame[ids] = self.motion.hit_frame[self.motion_ids[ids]]
        self.v13b_teacher_rephased[ids] = True
        self.v13b_upper_prior_wrap_count[ids] = 0
        # Do not run a second READY->frame0 bridge after choosing a nonzero
        # teacher start frame.  The event clock is now authoritative.
        self.prelude_elapsed_steps[ids] = self.prelude_steps
        self.tail_steps[ids] = 0

    def can_begin_next_shot(self, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """Return the unified physical recovery gate for next-action admission.

        This is deliberately a query separate from ``begin_next_shot`` so
        callers can report a rejected transition without mutating command
        timing.  Missing physical state is a hard failure rather than an
        implicit approval.
        """
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if ids.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool, device=self.device)
        from training.tasks.tracking.mdp.fall_state import unified_fall_state

        state = unified_fall_state(self._env)
        if state.recovery_ready.shape != (self.num_envs,):
            raise RuntimeError("unified recovery gate returned an invalid shape")
        return state.recovery_ready[ids] & (~state.confirmed_fall[ids]) & (~state.predicted_unrecoverable[ids])

    def export_v29_rsi_state(self, env_ids: Sequence[int] | torch.Tensor) -> dict[str, torch.Tensor]:
        """Export all reference-cycle state needed for a V29 recovery RSI load."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        fields = (
            "motion_ids",
            "time_steps",
            "shot_cycle",
            "tail_steps",
            "prelude_elapsed_steps",
            "last_reset_pose_offset",
            "last_reset_velocity_offset",
            "last_reset_hard_case",
        )
        return {
            "schema_version": torch.tensor(3, dtype=torch.int64),
            "snapshot_phase": "post_physics_pre_observation",
            **{name: getattr(self, name)[ids].detach().clone() for name in fields},
        }

    def restore_v29_rsi_state(
        self, state: dict[str, torch.Tensor], env_ids: Sequence[int] | torch.Tensor
    ) -> None:
        """Restore a V29 reference-cycle snapshot; reject incomplete contracts."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        required = (
            "motion_ids", "time_steps", "shot_cycle", "tail_steps",
            "prelude_elapsed_steps", "last_reset_pose_offset",
            "last_reset_velocity_offset", "last_reset_hard_case",
        )
        missing = [name for name in required if name not in state]
        if (
            int(state.get("schema_version", torch.tensor(-1)).item()) != 3
            or state.get("snapshot_phase") != "post_physics_pre_observation"
            or missing
        ):
            raise ValueError(f"Invalid V29 MotionCommand RSI snapshot; missing={missing}")
        for name in required:
            value = state[name].to(device=self.device)
            if value.shape[0] != ids.numel():
                raise ValueError(f"V29 RSI field {name!r} has incompatible batch size")
            getattr(self, name)[ids] = value

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if bool(getattr(self._env, "v13b_private_motion_disabled", False)):
            # Final V1.3B iterations must still reset the physical plant to
            # the shared READY, but must not select/read a motion-bank clip.
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
            self.time_steps[ids] = 0
            self.tail_steps[ids] = 0
            self.prelude_elapsed_steps[ids] = 0
            self.v13b_teacher_rephased[ids] = False
            self.v13b_teacher_start_frame[ids] = 0
            self.v13b_teacher_hit_frame[ids] = 0
            self.v13b_upper_prior_wrap_count[ids] = 0
            root_state = self.robot.data.default_root_state.clone()
            root_state[ids, :3] += self._env.scene.env_origins[ids]
            joint_pos = self.ready_joint_pos[ids].clone()
            joint_vel = self.robot.data.default_joint_vel[ids].clone()
            limits = self.robot.data.soft_joint_pos_limits[ids]
            joint_pos = torch.clip(joint_pos, limits[:, :, 0], limits[:, :, 1])
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
            self.robot.write_root_state_to_sim(root_state[ids], env_ids=ids)
            return
        self.shot_cycle[env_ids] = 0
        self.tail_steps[env_ids] = 0
        self.prelude_elapsed_steps[env_ids] = 0
        self.v13b_teacher_rephased[env_ids] = False
        self.v13b_teacher_start_frame[env_ids] = 0
        self.v13b_teacher_hit_frame[env_ids] = 0
        self.v13b_upper_prior_wrap_count[env_ids] = 0
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
            joint_pos = self.ready_joint_pos.clone()
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
        if bool(getattr(self._env, "v13b_private_motion_disabled", False)):
            # The command remains registered for a single training run, but
            # final reference-free iterations must not read or advance motion
            # bank data once both priors are annealed out.
            return
        previous_steps = self.time_steps.clone()
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
        # CompletePriors is finite/one-shot: a rephased teacher is never
        # allowed to rewind before environment reset.  Count and fail through
        # the public event assertion if a future refactor reintroduces wrap.
        wrapped = self.v13b_teacher_rephased & (self.time_steps < previous_steps)
        self.v13b_upper_prior_wrap_count[wrapped] += 1

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
    # P5D-2 reference sampler.  This is reset-time data sampling only and is
    # intentionally not part of the actor observation or task goal contract.
    reference_sampling_mode: str = "uniform"
    reference_curriculum_stage: int = 0
    reference_curriculum_sizes: tuple[int, ...] = ()
    # Optional fixed manifest route for focused identification or recovery
    # training. This affects only reset-time random sampling; manual replay
    # and external-target selection retain their own contracts.
    fixed_motion_id: int | None = None
    # Optional execution-time Cartesian centres for external target selection.
    # These offsets are measured relative to the manifest strike anchors in
    # the command-receipt yaw-heading frame.  Keeping them separate from the
    # motion data makes it possible to select against the actual stabilized
    # plant without rewriting reference demonstrations.
    external_control_anchor_offset_b_by_motion: tuple[tuple[float, float, float], ...] = ()
    # A motion must be explicitly admitted before the external-target
    # selector may choose it.  Empty preserves legacy all-manifest selection.
    # It is intentionally independent of a manual ``motion_id`` replay.
    external_control_anchor_enabled_by_motion: tuple[bool, ...] = ()
    # Per-motion verified half-ranges around the external control centres.
    # Empty leaves the caller-level range policy unchanged; nonempty entries
    # are an execution safety ceiling and cannot be widened by play.py.
    external_control_local_half_range_by_motion: tuple[tuple[float, float, float], ...] = ()
    # Explicit frame adapter for using HOPE competition-frame motions in tracking scenes whose ground is z=0.
    # HOPE competition frame uses z=0 at table surface and floor at z=-0.76, so the manifest task sets +0.76.
    manifest_frame_z_offset: float = 0.0
    manifest_ground_align: bool = False
    # Opt-in validation for prepositioned stance metadata. Fixed-base manifests
    # remain unchanged unless this is explicitly enabled.
    validate_stance_contract: bool = False
    stance_contract_mode: str | None = None
    # V19+ must fail closed when a legacy NPZ without canonical FK-derived
    # upper momentum is selected.
    require_upper_momentum: bool = False
    # A fixed-root floating tracker must not silently mix motions authored at
    # different headings.  When set, every manifest payload root quaternion is
    # checked against the task READY quaternion before the first physics step.
    expected_root_quaternion_wxyz: tuple[float, ...] = ()
    root_quaternion_tolerance: float = 1.0e-4
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
    # Task-local READY overrides. Empty preserves robot.init_state exactly.
    # V27 uses this for a bent arm without changing V25/V26 asset defaults.
    ready_joint_positions: dict[str, float] = {}
    sample_random_start_phase: bool = True
    # Hold a finite motion's final reference frame until the environment reset.
    # The task episode length defines the required settling-tail duration. Zero
    # preserves legacy looping behaviour.
    hold_last_frame_steps: int = 0
    # Full-cycle admission: a motion tail is not a recovery verdict.  These
    # values are consumed by cycle/audit tooling and are intentionally kept
    # in the command contract so timeout/guard semantics cannot drift.
    post_hit_guard_steps: int = 75
    recovery_timeout_steps: int = 250
    recovery_ready_hold_steps: int = 15
    # After ``hold_last_frame_steps`` at the final motion pose, blend the
    # complete joint reference back to ``robot.init_state`` over this many
    # control steps.  The remaining episode time holds that ready pose, so a
    # task can require post-return stability rather than hiding a fall at the
    # end of the strike clip.
    return_to_default_steps: int = 0
    # A pre-swing bridge from ``robot.init_state`` into motion frame zero.
    # It is disabled for legacy tracking tasks.
    prelude_steps: int = 0
    # Optional three-part ready-to-swing bridge.  Legacy tasks retain the
    # prior linear bridge because all controls default to disabled.
    prelude_settle_steps: int = 0
    prelude_launch_steps: int = 0
    prelude_minimum_jerk: bool = False
    prelude_quintic_hermite: bool = False
    prelude_continuous_velocity_reference: bool = False
    prelude_waist_pitch_anchor_rad: float | None = None
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
