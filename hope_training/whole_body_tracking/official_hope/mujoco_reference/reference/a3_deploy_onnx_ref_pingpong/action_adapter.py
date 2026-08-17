# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Shared ActionAdapter: raw policy output -> joint-position targets.

The adapter is a pure, deterministic numeric transform (a joint-position residual
plus a clamp). It is NOT a rejection filter and emits no failure status:

    q_des = default_q + stance_offset + raw_action * action_scale
    q_des = clip(q_des, clamp_lower, clamp_upper)

The runtime may load either the neutral public example
(``config/action_adapter.yaml``) or the exact Unitree-style ``deploy.yaml`` shipped
with a published model bundle.  In both cases the transform is resolved into the
same SDK/MuJoCo joint order.

Vendor hard limits, motor protection, and e-stop remain the responsibility of the
robot backend; this transform neither probes nor bypasses them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .joint_order import JOINT_NAMES, NUM_JOINTS


@dataclass
class ActionAdapter:
    default_q: np.ndarray     # (31,) neutral/upright joint positions, rad
    action_scale: np.ndarray  # (31,) per-column residual scale (uniform in the example)
    clamp_lower: np.ndarray   # (31,) lower joint-position clamp, rad
    clamp_upper: np.ndarray   # (31,) upper joint-position clamp, rad
    stance_offset: np.ndarray | None = None  # (31,) fixed stance offset, rad
    raw_action_clip: float = 20.0

    def __post_init__(self) -> None:
        for field in ("default_q", "action_scale", "clamp_lower", "clamp_upper"):
            v = np.asarray(getattr(self, field), dtype=np.float64).reshape(-1)
            if v.shape[0] != NUM_JOINTS:
                raise ValueError(f"{field} must be length {NUM_JOINTS}, got {v.shape[0]}")
            setattr(self, field, v)
        if self.stance_offset is None:
            self.stance_offset = np.zeros(NUM_JOINTS, dtype=np.float64)
        else:
            self.stance_offset = np.asarray(self.stance_offset, dtype=np.float64).reshape(-1)
            if self.stance_offset.shape[0] != NUM_JOINTS:
                raise ValueError(
                    f"stance_offset must be length {NUM_JOINTS}, got {self.stance_offset.shape[0]}"
                )
        if not np.all(np.isfinite(self.stance_offset)):
            raise ValueError("stance_offset must contain only finite values")
        if np.any(self.clamp_lower > self.clamp_upper):
            raise ValueError("action_adapter clamp_lower must be <= clamp_upper for every joint")

    @property
    def stance_q(self) -> np.ndarray:
        """Fixed target posture used by the deployed affine action contract."""
        return self.default_q + self.stance_offset

    def decode(self, raw_action: np.ndarray) -> np.ndarray:
        """Map raw policy output to q_des using HOPE's deploy contract.

        The native runner first clips the residual action to +/-20, then applies
        the affine decode and finally the exported safe q_des interval.
        """
        raw = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        if raw.shape[0] != NUM_JOINTS:
            raise ValueError(f"raw_action must be length {NUM_JOINTS}, got {raw.shape[0]}")
        raw = np.clip(raw, -float(self.raw_action_clip), float(self.raw_action_clip))
        q_des = self.stance_q + raw * self.action_scale
        return np.clip(q_des, self.clamp_lower, self.clamp_upper)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ActionAdapter":
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)

        # A published model bundle carries the authoritative Unitree-style
        # deploy.yaml used by the native runner. Reuse its resolved action arrays
        # directly so the lightweight MuJoCo runner does not need a second copy of
        # model-specific defaults, scales, or clamps.
        if doc.get("format") == "unitree_rl_lab.deploy":
            return cls._from_deploy_yaml(doc)

        default_q = _resolve_per_joint(doc["default_q"], "default_q")

        scale_spec = doc["action_scale"]
        if isinstance(scale_spec, (int, float)):
            action_scale = np.full(NUM_JOINTS, float(scale_spec), dtype=np.float64)
        else:
            action_scale = _resolve_per_joint(scale_spec, "action_scale")

        clamp = doc["joint_position_clamp"]
        clamp_lower = _resolve_per_joint(clamp["lower"], "joint_position_clamp.lower")
        clamp_upper = _resolve_per_joint(clamp["upper"], "joint_position_clamp.upper")
        return cls(default_q, action_scale, clamp_lower, clamp_upper)

    @classmethod
    def _from_deploy_yaml(cls, doc: dict) -> "ActionAdapter":
        action = doc["actions"]["JointPositionAction"]
        names = tuple(str(name) for name in action["joint_names"])
        if len(names) != NUM_JOINTS or set(names) != set(JOINT_NAMES):
            raise ValueError("deploy.yaml JointPositionAction must name all 31 A3 joints once")

        def reorder(values, field_name: str) -> np.ndarray:
            if len(values) != NUM_JOINTS:
                raise ValueError(f"deploy.yaml {field_name} must be length {NUM_JOINTS}")
            by_name = {name: value for name, value in zip(names, values)}
            return np.asarray([by_name[name] for name in JOINT_NAMES], dtype=np.float64)

        clips = action["clip"]
        if len(clips) != NUM_JOINTS or any(len(pair) != 2 for pair in clips):
            raise ValueError("deploy.yaml JointPositionAction.clip must contain 31 [lo, hi] pairs")
        lower = reorder([pair[0] for pair in clips], "JointPositionAction.clip.lower")
        upper = reorder([pair[1] for pair in clips], "JointPositionAction.clip.upper")
        return cls(
            default_q=reorder(action["offset"], "JointPositionAction.offset"),
            action_scale=reorder(action["scale"], "JointPositionAction.scale"),
            clamp_lower=lower,
            clamp_upper=upper,
            stance_offset=reorder(
                action.get("stance_offset", [0.0] * NUM_JOINTS),
                "JointPositionAction.stance_offset",
            ),
            raw_action_clip=20.0,
        )


def _resolve_per_joint(spec, field_name: str) -> np.ndarray:
    """Accept either an ordered length-31 list or a ``{joint_name: value}`` map."""
    if isinstance(spec, dict):
        missing = [n for n in JOINT_NAMES if n not in spec]
        if missing:
            raise ValueError(f"{field_name} is missing joints: {missing[:3]}...")
        return np.array([float(spec[n]) for n in JOINT_NAMES], dtype=np.float64)
    arr = np.asarray(spec, dtype=np.float64).reshape(-1)
    if arr.shape[0] != NUM_JOINTS:
        raise ValueError(f"{field_name} must be length {NUM_JOINTS}, got {arr.shape[0]}")
    return arr
