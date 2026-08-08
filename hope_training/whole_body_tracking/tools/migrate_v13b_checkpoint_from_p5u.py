#!/usr/bin/env python3
"""Migrate a P5U-1 (519-D/26-D) checkpoint into the V1.3B contract.

This is an explicit semantic migration, not a raw ``resume``.  The P5U
reference-preview features are not copied into the new actor.  Only shared
proprioception, actual racket state, and the three matching target channels
are mapped into the V1.3B first layer; the new goal column for signed time is
mapped from P5U's time-to-strike.  The 26-D output head is shape-compatible
and is copied as an initialization prior, while optimizer state is discarded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import torch


@dataclass(frozen=True)
class ObservationSlice:
    """Local copy keeps this offline checkpoint tool independent of IsaacSim."""

    name: str
    start: int
    end: int


def _slices(spec: list[tuple[str, int]]) -> dict[str, ObservationSlice]:
    out: dict[str, ObservationSlice] = {}
    cursor = 0
    for name, width in spec:
        out[name] = ObservationSlice(name, cursor, cursor + width)
        cursor += width
    return out


OLD_TERMS = _slices(
    [
        ("base_lin_vel", 3), ("base_ang_vel", 3),
        ("joint_pos", 22), ("joint_vel", 22), ("actions", 26),
        ("projected_gravity", 3), ("feet_contact", 2),
        ("reference_joint_pos", 22), ("reference_joint_vel", 22),
        ("reference_joint_pos_error", 22), ("reference_joint_vel_error", 22),
        ("reference_joint_pos_8", 22), ("reference_joint_vel_8", 22),
        ("reference_joint_pos_16", 22), ("reference_joint_vel_16", 22),
        ("reference_joint_pos_1", 22), ("reference_joint_vel_1", 22),
        ("reference_joint_pos_3", 22), ("reference_joint_vel_3", 22),
        ("reference_joint_pos_6", 22), ("reference_joint_vel_6", 22),
        ("reference_joint_pos_12", 22), ("reference_joint_vel_12", 22),
        ("strike_phase", 1), ("strike_phase_sin", 1), ("strike_phase_cos", 1),
        ("time_to_strike", 1), ("marked_hit_step", 1),
        ("racket_target_pos_b", 3), ("racket_target_vel_b", 3),
        ("racket_target_normal_b", 3), ("racket_pos_b", 3),
        ("racket_lin_vel_b", 3), ("racket_normal_b", 3),
        ("racket_target_error_pos_b", 3), ("racket_target_error_vel_b", 3),
        ("racket_target_error_normal_b", 3),
        ("reference_racket_pos_b", 3), ("reference_racket_vel_b", 3),
        ("reference_racket_normal_b", 3),
        ("reference_racket_pos_b_1", 3), ("reference_racket_vel_b_1", 3),
        ("reference_racket_normal_b_1", 3),
        ("reference_racket_pos_b_3", 3), ("reference_racket_vel_b_3", 3),
        ("reference_racket_normal_b_3", 3),
        ("reference_racket_pos_b_6", 3), ("reference_racket_vel_b_6", 3),
        ("reference_racket_normal_b_6", 3),
        ("reference_racket_pos_b_12", 3), ("reference_racket_vel_b_12", 3),
        ("reference_racket_normal_b_12", 3),
        ("reference_racket_pos_error_b", 3), ("reference_racket_vel_error_b", 3),
        ("reference_racket_normal_error_b", 3),
    ]
)

NEW_TERMS = _slices(
    [
        ("base_lin_vel", 3), ("base_ang_vel", 3), ("joint_pos", 22),
        ("joint_vel", 22), ("actions", 26), ("projected_gravity", 3),
        ("racket_pos_b", 3), ("racket_lin_vel_b", 3),
        ("racket_normal_b", 3), ("strike_goal_10d", 10),
    ]
)

# P5U's first 12 public outputs were lower balance residuals around the
# model_3396 target; its next 10 were 0.035-rad upper residuals.  V1.3B uses
# the qualified direct-action envelope.  Rescale the output rows so the
# migrated mean has approximately the same physical joint authority instead
# of inheriting a 0.035-rad residual as a direct 0.44-rad command.
OLD_ACTION_SCALE = torch.tensor(
    (0.048, 0.140, 0.184, 0.040, 0.060, 0.028) * 2 + (0.035,) * 10 + (1.0,) * 4,
    dtype=torch.float32,
)
NEW_ACTION_SCALE = torch.tensor(
    # Must exactly match cfg/target_conditioned/direct_action_scale_v13b_annealed_prior.yaml.
    (0.192, 0.048, 0.192, 0.192, 0.144, 0.192, 0.192, 0.048, 0.096, 0.072, 0.144, 0.192)
    + (0.440, 0.022, 0.110, 0.440, 0.0132, 0.110, 0.440, 0.440, 0.440, 0.440)
    + (1.0,) * 4,
    dtype=torch.float32,
)


def _copy_columns(old_weight: torch.Tensor, new_width: int, mapping: list[tuple[ObservationSlice, ObservationSlice]]) -> torch.Tensor:
    result = torch.zeros((old_weight.shape[0], new_width), dtype=old_weight.dtype)
    for source, target in mapping:
        width = min(source.end - source.start, target.end - target.start)
        result[:, target.start : target.start + width] = old_weight[:, source.start : source.start + width]
    return result


def _map_vector(old: torch.Tensor, new_width: int, mapping: list[tuple[ObservationSlice, ObservationSlice]], *, fill: float) -> torch.Tensor:
    result = torch.full((1, new_width), fill, dtype=old.dtype)
    for source, target in mapping:
        width = min(source.end - source.start, target.end - target.start)
        result[:, target.start : target.start + width] = old[:, source.start : source.start + width]
    return result


def _mapping() -> list[tuple[ObservationSlice, ObservationSlice]]:
    pairs = []
    for name in ("base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel", "actions", "projected_gravity", "racket_pos_b", "racket_lin_vel_b", "racket_normal_b"):
        pairs.append((OLD_TERMS[name], NEW_TERMS[name]))
    goal = NEW_TERMS["strike_goal_10d"]
    pairs.extend([
        (OLD_TERMS["racket_target_pos_b"], ObservationSlice("goal_pos", goal.start, goal.start + 3)),
        (OLD_TERMS["racket_target_vel_b"], ObservationSlice("goal_vel", goal.start + 3, goal.start + 6)),
        (OLD_TERMS["racket_target_normal_b"], ObservationSlice("goal_normal", goal.start + 6, goal.start + 9)),
        (OLD_TERMS["time_to_strike"], ObservationSlice("signed_time_to_hit", goal.start + 9, goal.start + 10)),
    ])
    return pairs


def migrate(src: Path, dst: Path) -> dict:
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    old = checkpoint.get("model_state_dict")
    if not isinstance(old, dict):
        raise RuntimeError(f"no model_state_dict in {src}")
    if tuple(old["actor.0.weight"].shape) != (512, 519) or tuple(old["actor.6.weight"].shape) != (26, 128):
        raise RuntimeError("source is not the expected P5U 519-D/26-D actor contract")

    pairs = _mapping()
    state: dict[str, torch.Tensor] = {}
    # Migrate the actor only.  The critic/privileged observation contract is
    # intentionally reinitialized by the V1.3B runner.
    for key, value in old.items():
        if key.startswith("actor.") and key not in {"actor.0.weight", "actor.6.weight", "actor.6.bias"}:
            state[key] = value.clone()
    state["actor.0.weight"] = _copy_columns(old["actor.0.weight"], 98, pairs)
    output_scale = OLD_ACTION_SCALE / NEW_ACTION_SCALE
    state["actor.6.weight"] = old["actor.6.weight"] * output_scale.to(old["actor.6.weight"].device).unsqueeze(-1)
    state["actor.6.bias"] = old["actor.6.bias"] * output_scale.to(old["actor.6.bias"].device)
    # Start exploration from the V1.3B safe direct-action setting; old P5U
    # per-channel noise belongs to a different residual/action contract.
    state["std"] = torch.full_like(old["std"], 0.15)

    old_norm = checkpoint.get("obs_norm_state_dict", {})
    norm = {}
    if all(k in old_norm for k in ("_mean", "_var", "_std")):
        for key in ("_mean", "_var", "_std"):
            fill = 0.0 if key == "_mean" else 1.0
            norm[key] = _map_vector(old_norm[key], 98, pairs, fill=fill)
        norm["count"] = old_norm.get("count", torch.tensor(0)).clone()
    else:
        norm = {"_mean": torch.zeros(1, 98), "_var": torch.ones(1, 98), "_std": torch.ones(1, 98), "count": torch.tensor(0)}

    # The new critic has only one privileged time-left scalar.  Keep its
    # normalizer independent of the old 532-D privileged reference contract.
    priv_norm = {
        "_mean": torch.zeros(1, 99), "_var": torch.ones(1, 99),
        "_std": torch.ones(1, 99), "count": torch.tensor(0),
    }
    migrated = {
        "model_state_dict": state,
        "obs_norm_state_dict": norm,
        "privileged_obs_norm_state_dict": priv_norm,
        "iter": 0,
        "infos": {"migration": "p5u_519_to_v13b_98_explicit_semantic", "source": str(src)},
        "v13b_migrated_from_p5u": True,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, dst)
    summary = {
        "source": str(src), "destination": str(dst), "source_actor_obs": 519,
        "destination_actor_obs": 98, "action_dim": 26,
        "mapped_terms": [f"{s.name}->{t.name}" for s, t in pairs],
        "observation_index_mapping": [
            {
                "source_term": source.name,
                "source_indices": list(range(source.start, source.end)),
                "target_term": target.name,
                "target_indices": list(range(target.start, target.end)),
            }
            for source, target in pairs
        ],
        "critic": "reinitialized_by_v13b_runner",
        "optimizer_state": "discarded",
        "action_mean_row_scale_old_over_new": output_scale.tolist(),
        "action_index_mapping": [
            {
                "target_index": index,
                "source_index": index,
                "old_scale_rad": float(OLD_ACTION_SCALE[index]),
                "new_scale_rad": float(NEW_ACTION_SCALE[index]),
                "output_row_ratio_old_over_new": float(output_scale[index]),
            }
            for index in range(26)
        ],
        "dropped_reference_features": [name for name in OLD_TERMS if name not in {s.name for s, _ in pairs}],
    }
    dst.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
