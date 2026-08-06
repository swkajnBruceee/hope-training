"""Source/runtime audit for the physical meaning of RacketCommand.position."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .strike_goal import STRIKE_GOAL_CONTRACT_VERSION, STRIKE_GOAL_POSITION_SEMANTICS


class PlannerPositionContractError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_planner_position_contract(
    workspace_root: Path, runtime_probe_path: Path
) -> dict[str, Any]:
    """Fail closed unless source dataflow and runtime evidence both say ball centre."""

    workspace_root = workspace_root.expanduser().resolve()
    files = {
        "strike_definition": workspace_root
        / "hope_ws/src/trajectory/include/ball_trajectory_predictor.h",
        "hit_plan_assignment": workspace_root / "hope_ws/src/solver/src/hit_plan_solver.cpp",
        "message_publication": workspace_root / "hope_ws/src/solver/src/solver_node.cpp",
    }
    required_fragments = {
        "strike_definition": (
            "Eigen::Vector3d p_ball;",
            "predicted ball position at strike",
        ),
        "hit_plan_assignment": ("plan.p_hit = strike.p_ball;",),
        "message_publication": ("fillPoint(out.position, plan.p_hit);",),
    }
    source_evidence = {}
    for label, path in files.items():
        if not path.is_file():
            raise PlannerPositionContractError(f"missing Planner source file: {path}")
        text = path.read_text(encoding="utf-8")
        missing = [fragment for fragment in required_fragments[label] if fragment not in text]
        if missing:
            raise PlannerPositionContractError(
                f"Planner position dataflow changed in {path}: missing {missing}"
            )
        source_evidence[label] = {
            "path": str(path),
            "sha256": _sha256(path),
            "required_fragments": list(required_fragments[label]),
        }

    runtime_probe_path = runtime_probe_path.expanduser().resolve()
    runtime = json.loads(runtime_probe_path.read_text(encoding="utf-8"))
    cases = runtime.get("cases", [])
    if not cases:
        raise PlannerPositionContractError("runtime Planner probe contains no cases")
    if not all(case.get("position_equals_predicted_ball_position") is True for case in cases):
        raise PlannerPositionContractError(
            "runtime Planner probe does not confirm position equals predicted ball position"
        )
    raw_positions = [case.get("goal_10d_raw", [])[:3] for case in cases]
    if any(len(position) != 3 for position in raw_positions):
        raise PlannerPositionContractError("runtime Planner probe has an invalid 10D position")

    return {
        "contract_version": STRIKE_GOAL_CONTRACT_VERSION,
        "racket_command_position_semantics": STRIKE_GOAL_POSITION_SEMANTICS,
        "verdict": "predicted_ball_center_at_strike",
        "excluded_semantics": [
            "racket_link_origin",
            "racket_face_center",
            "ball_racket_contact_point",
            "implicitly_offset_virtual_point",
        ],
        "source_dataflow": [
            "trajectory::StrikeTarget.p_ball",
            "HitPlan.p_hit = strike.p_ball",
            "RacketCommand.position = HitPlan.p_hit",
        ],
        "source_evidence": source_evidence,
        "runtime_evidence": {
            "path": str(runtime_probe_path),
            "sha256": _sha256(runtime_probe_path),
            "case_count": len(cases),
            "all_positions_equal_predicted_ball_position": True,
            "observed_positions": raw_positions,
        },
        "policy_mapping_requirement": (
            "A separately versioned ball-centre/contact/TCP transform is required; "
            "it must not be applied when a future Planner contract changes semantics."
        ),
    }
