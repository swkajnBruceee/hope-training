from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from training.utils.planner_position_contract import (  # noqa: E402
    PlannerPositionContractError,
    audit_planner_position_contract,
)


def test_actual_planner_source_and_runtime_probe_lock_position_to_ball_center():
    report = audit_planner_position_contract(
        WORKSPACE_ROOT,
        ROOT / "eval_outputs/strike_goal_p1/planner_ros_delay_probe.json",
    )
    assert report["verdict"] == "predicted_ball_center_at_strike"
    assert report["runtime_evidence"]["case_count"] == 3
    assert "racket_link_origin" in report["excluded_semantics"]


def test_contract_fails_closed_when_source_dataflow_changes(tmp_path: Path):
    fake_root = tmp_path
    paths = {
        "hope_ws/src/trajectory/include/ball_trajectory_predictor.h": (
            "Eigen::Vector3d p_ball; // predicted ball position at strike"
        ),
        "hope_ws/src/solver/src/hit_plan_solver.cpp": "plan.p_hit = some_tcp;",
        "hope_ws/src/solver/src/solver_node.cpp": "fillPoint(out.position, plan.p_hit);",
    }
    for relative, content in paths.items():
        path = fake_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    runtime = fake_root / "runtime.json"
    runtime.write_text(
        '{"cases":[{"position_equals_predicted_ball_position":true,'
        '"goal_10d_raw":[0,0,0,0,0,0,0,0,0,0]}]}',
        encoding="utf-8",
    )
    with pytest.raises(PlannerPositionContractError, match="dataflow changed"):
        audit_planner_position_contract(fake_root, runtime)
