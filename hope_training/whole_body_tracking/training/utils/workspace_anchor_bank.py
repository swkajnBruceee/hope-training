"""Lightweight audited strike-anchor metadata for V1.3B workspace expansion.

This module deliberately does not import Isaac, load an NPZ, instantiate a
motion command, or load any teacher checkpoint.  The expansion sampler only
needs the canonical 10-D strike metadata already written in the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


class WorkspaceStrikeAnchorBank:
    """Immutable manifest-backed anchor bank in the locked local goal frame."""

    def __init__(
        self,
        manifest_path: str | Path,
        device: torch.device | str,
        *,
        require_qualified: bool = False,
        nominal_local: tuple[float, float, float] = (0.42, -0.18, 0.18),
        support_half_range: tuple[float, float, float] = (0.08, 0.08, 0.08),
        support_tolerance_m: float = 1.0e-4,
    ):
        path = Path(manifest_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"workspace anchor manifest does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("motions") if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not rows:
            raise ValueError("workspace anchor manifest must contain a non-empty motions list")

        self.manifest_path = str(path)
        self.device = torch.device(device)
        self.require_qualified = bool(require_qualified)
        self.source_anchor_count = len(rows)
        self.qualified_anchor_count = 0
        self.rejected_anchor_count = 0
        self.qualification_reason_counts: dict[str, int] = {}
        if self.require_qualified:
            if not isinstance(payload, dict) or payload.get("physics_qualified") is not True or payload.get("training_admission") is not True:
                raise RuntimeError(
                    "qualified workspace anchor manifest must have top-level physics_qualified=true "
                    "and training_admission=true"
                )
            declared = payload.get("qualified_anchor_count")
            if declared is None or int(declared) != len(rows):
                raise RuntimeError("qualified_anchor_count does not equal manifest motions length")

        positions: list[list[float]] = []
        velocities: list[list[float]] = []
        normals: list[list[float]] = []
        timings: list[float] = []
        source_ids: list[str] = []
        for row in rows:
            reasons: list[str] = []
            if isinstance(row, dict):
                if row.get("physics_qualified") is not True:
                    reasons.append("physics_pending")
                if row.get("training_admission") is not True:
                    reasons.append("training_pending")
            else:
                reasons.append("row_not_mapping")
            if self.require_qualified and reasons:
                self.rejected_anchor_count += 1
                for reason in reasons:
                    self.qualification_reason_counts[reason] = self.qualification_reason_counts.get(reason, 0) + 1
                continue
            goal = row.get("canonical_goal_10d") if isinstance(row, dict) else None
            if not isinstance(goal, dict):
                self.rejected_anchor_count += 1
                self.qualification_reason_counts["missing_canonical_goal"] = self.qualification_reason_counts.get("missing_canonical_goal", 0) + 1
                continue
            p = goal.get("position_m")
            v = goal.get("linear_velocity_mps")
            n = goal.get("normal_w")
            t = goal.get("time_to_hit_s")
            if not (isinstance(p, list) and len(p) == 3 and isinstance(v, list) and len(v) == 3
                    and isinstance(n, list) and len(n) == 3 and isinstance(t, (int, float))):
                self.rejected_anchor_count += 1
                self.qualification_reason_counts["invalid_canonical_goal"] = self.qualification_reason_counts.get("invalid_canonical_goal", 0) + 1
                continue
            positions.append([float(x) for x in p])
            velocities.append([float(x) for x in v])
            nn = torch.tensor(n, dtype=torch.float32)
            nn = nn / torch.linalg.vector_norm(nn).clamp_min(1.0e-6)
            normals.append(nn.tolist())
            timings.append(float(t))
            source_ids.append(str(row.get("motion_id", len(source_ids))))
        if not positions:
            raise ValueError(f"manifest contains no valid canonical_goal_10d rows: {path}")

        self.position_local = torch.tensor(positions, dtype=torch.float32, device=self.device)
        self.velocity_local = torch.tensor(velocities, dtype=torch.float32, device=self.device)
        self.normal_local = torch.tensor(normals, dtype=torch.float32, device=self.device)
        self.time_to_hit_s = torch.tensor(timings, dtype=torch.float32, device=self.device)
        self.source_ids = tuple(source_ids)
        self.anchor_count = int(self.position_local.shape[0])
        self.qualified_anchor_count = self.anchor_count if self.require_qualified else sum(
            1 for row in rows if isinstance(row, dict) and row.get("physics_qualified") is True and row.get("training_admission") is True
        )
        if self.require_qualified and self.rejected_anchor_count:
            raise RuntimeError(
                f"qualified workspace anchor manifest contains {self.rejected_anchor_count} unqualified rows: "
                f"{self.qualification_reason_counts}"
            )
        nominal = torch.tensor(nominal_local, device=self.device, dtype=self.position_local.dtype)
        half = torch.tensor(support_half_range, device=self.device, dtype=self.position_local.dtype)
        if torch.any(half < 0):
            raise ValueError("support_half_range must be non-negative")
        self.support_half_range = tuple(float(x) for x in support_half_range)
        excess = torch.clamp(torch.abs(self.position_local - nominal) - half, min=0.0)
        self.support_distance_m = torch.linalg.vector_norm(excess, dim=-1)
        self.support_inside_mask = self.support_distance_m <= float(support_tolerance_m)
        self.support_tolerance_m = float(support_tolerance_m)
        self.sorted_support_ids = torch.argsort(self.support_distance_m)

    def statistics(self, nominal_local: tuple[float, float, float]) -> dict[str, object]:
        nominal = torch.tensor(nominal_local, device=self.device, dtype=self.position_local.dtype)
        distance = torch.linalg.vector_norm(self.position_local - nominal, dim=-1)
        def row(x: torch.Tensor) -> dict[str, float]:
            q = torch.quantile(x, torch.tensor([0.0, 0.01, 0.25, 0.50, 0.75, 0.99, 1.0], device=x.device))
            return {k: float(v) for k, v in zip(("min", "p01", "p25", "p50", "p75", "p99", "max"), q.cpu())}
        return {
            "manifest": self.manifest_path,
            "anchor_count": self.anchor_count,
            "source_anchor_count": self.source_anchor_count,
            "qualified_anchor_count": self.qualified_anchor_count,
            "rejected_anchor_count": self.rejected_anchor_count,
            "qualification_reason_counts": dict(self.qualification_reason_counts),
            "local_support_half_range_m": list(self.support_half_range),
            "inside_current_support_count": int(self.support_inside_mask.sum().item()),
            "inside_current_support_fraction": float(self.support_inside_mask.float().mean().item()),
            "position_xyz_m": {axis: row(self.position_local[:, i]) for i, axis in enumerate(("x", "y", "z"))},
            "distance_from_nominal_m": row(distance),
            "support_distance_m": row(self.support_distance_m),
        }

    def sample(
        self,
        count: int,
        progress: float,
        nominal_local: tuple[float, float, float],
        support_half_range: tuple[float, float, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Sample a smooth near-to-far active set and return local goal centers."""
        if count <= 0:
            raise ValueError("anchor sample count must be positive")
        p = max(0.0, min(1.0, float(progress)))
        # Eligibility is based on distance outside the old learned support,
        # not nearest-to-nominal row count. Distances are cached at init.
        knots_p = (0.00, 0.10, 0.25, 0.40, 0.60, 1.00)
        max_distance = float(self.support_distance_m.max().item())
        knots_d = (0.0, 0.05, 0.10, 0.20, 0.35, max_distance)
        threshold = knots_d[-1]
        for left_p, right_p, left_d, right_d in zip(knots_p[:-1], knots_p[1:], knots_d[:-1], knots_d[1:]):
            if p <= right_p:
                u = 0.0 if right_p <= left_p else (p - left_p) / (right_p - left_p)
                u = max(0.0, min(1.0, u))
                smooth = u * u * (3.0 - 2.0 * u)
                threshold = left_d + smooth * (right_d - left_d)
                break
        eligible_ids = torch.nonzero(self.support_distance_m <= threshold + self.support_tolerance_m, as_tuple=False).flatten()
        if eligible_ids.numel() == 0:
            eligible_ids = torch.nonzero(self.support_inside_mask, as_tuple=False).flatten()
        if eligible_ids.numel() == 0:
            raise RuntimeError("workspace anchor bank has no anchors inside the current support")
        eligible = int(eligible_ids.numel())
        picks = eligible_ids[torch.randint(eligible, (count,), device=self.device)]
        return (
            self.position_local[picks], self.velocity_local[picks], self.normal_local[picks],
            self.time_to_hit_s[picks], picks.to(dtype=torch.long), eligible,
        )
