"""Pure tensor helpers for the actuator torque-headroom reward."""

from __future__ import annotations

import torch


def torque_headroom_penalty(
    utilization: torch.Tensor,
    *,
    safe_fraction: float = 0.9,
    topk: int = 2,
    topk_blend: float = 0.7,
    penalty_cap: float = 9.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return a soft torque-rail debt and per-environment utilization metrics.

    ``utilization`` is ``abs(computed_torque) / effort_limit`` with shape ``[env, joint]``.
    The debt is zero below ``safe_fraction``, quadratic through the first full-rail interval,
    then linear with a finite cap.  The top-k component prevents one saturated actuator from
    disappearing in an all-joint mean, while the cap prevents scratch-training reward explosions.
    """
    safe = float(safe_fraction)
    blend = float(topk_blend)
    k = int(topk)
    cap = float(penalty_cap)
    if not (0.0 <= safe < 1.0):
        raise ValueError(f"safe_fraction must satisfy 0 <= safe_fraction < 1, got {safe_fraction}")
    if k <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"topk_blend must be in [0, 1], got {topk_blend}")
    if not torch.isfinite(torch.as_tensor(cap)) or cap <= 0.0:
        raise ValueError(f"penalty_cap must be finite and positive, got {penalty_cap}")
    if utilization.ndim != 2 or utilization.shape[-1] == 0:
        raise ValueError(f"utilization must have shape [env, joint], got {tuple(utilization.shape)}")

    finite_utilization = torch.nan_to_num(
        utilization,
        nan=0.0,
        posinf=10.0,
        neginf=0.0,
    ).clamp_min(0.0)
    x = ((finite_utilization - safe) / (1.0 - safe)).clamp_min(0.0)
    debt = torch.where(x <= 1.0, x.square(), 2.0 * x - 1.0).clamp_max(cap)
    k = min(k, int(debt.shape[-1]))
    topk_mean = torch.topk(debt, k=k, dim=-1).values.mean(dim=-1)
    all_mean = debt.mean(dim=-1)
    penalty = blend * topk_mean + (1.0 - blend) * all_mean

    with torch.no_grad():
        metrics = {
            "utilization_mean": finite_utilization.mean(dim=-1),
            "utilization_p95": torch.quantile(finite_utilization, 0.95, dim=-1),
            "utilization_p99": torch.quantile(finite_utilization, 0.99, dim=-1),
            "utilization_max": finite_utilization.max(dim=-1).values,
            "over_safe_fraction": (finite_utilization > safe).float().mean(dim=-1),
            "saturation_fraction": (finite_utilization > 1.0).float().mean(dim=-1),
            "headroom_penalty": penalty.detach(),
        }
    return penalty, metrics
