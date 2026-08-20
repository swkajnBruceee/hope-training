import itertools
import ast
import numpy as np
from pathlib import Path
from hope_training.whole_body_tracking.deployment_v2 import load_canonical_metadata, map_normalized_velocity
from hope_training.whole_body_tracking.deployment_v2_isaac.action_space_feasibility import *
from hope_training.whole_body_tracking.deployment_v2_isaac.v2_one_ball_env import MODEL


def test_tier_counts_and_unique_union():
    t1=tier1_actions(); t2=tier2_additional_actions(); full=full_grid_actions()
    assert len(t1)==27 and len(t2)==98 and len(full)==125
    assert len(set(t1+t2))==125


def test_all_actions_in_domain_center_once_and_all_corners():
    full=full_grid_actions()
    assert all(all(-1 <= x <= 1 for x in a) for a in full)
    assert full.count((0.,0.,0.))==1
    assert set(itertools.product((-1.,1.),repeat=3)) <= set(full)


def test_deterministic_vx_vy_vz_order():
    assert tier1_actions()[:4] == ((-1.,-1.,-1.),(-1.,-1.,0.),(-1.,-1.,1.),(-1.,0.,-1.))
    assert tier1_actions() == tier1_actions()


def test_frozen_bh_mapping_low_center_high():
    m=load_canonical_metadata(MODEL)
    assert np.allclose(map_normalized_velocity([-1]*3,-1,m),[1.55,-.18,.40])
    assert np.allclose(map_normalized_velocity([0]*3,-1,m),[2.035,.055,.86])
    assert np.allclose(map_normalized_velocity([1]*3,-1,m),[2.52,.29,1.32])


def test_first_legal_selection_rule():
    rows=[{"id":0,"LEGAL_RETURN":False},{"id":1,"LEGAL_RETURN":True},{"id":2,"LEGAL_RETURN":True}]
    assert first_legal_result(rows)["id"]==1


def test_negative_semantics_never_claims_infeasible():
    s=negative_search_semantics([{"LEGAL_RETURN":False}])
    assert s["NO_WITNESS_FOUND_WITHIN_SEARCH_PROTOCOL"] and not s["ACTION_SPACE_INFEASIBLE"]


def test_repeatability_gate_exactly_five_clean_legal_stable():
    good={"LEGAL_RETURN":True,"SIMULATION_STABILITY":True,"SIMULATION_NONFINITE_COUNT":0}
    assert repeatability_pass([good.copy() for _ in range(5)])
    assert not repeatability_pass([good.copy() for _ in range(4)])
    bad=good.copy(); bad["LEGAL_RETURN"]=False
    assert not repeatability_pass([good.copy() for _ in range(4)]+[bad])


def test_runner_import_order_and_fail_closed_markers():
    source=(Path(__file__).parents[1]/"scripts"/"v2b21_action_space_legal_return_feasibility_v1.py").read_text()
    assert "from a3_deploy_onnx_ref_pingpong.lifecycle import SwingLifecycle" not in source
    assert "lifecycle_type=type(ex.lifecycle)" in source
    assert "ex.lifecycle=lifecycle_type(ex.lifecycle_cfg)" in source
    for marker in ("MAIN_ENTER","EXECUTOR_CREATED","SEARCH_BEGIN","FIRST_CASE_BEGIN","WITNESS_FOUND","WITNESS_REPEAT_BEGIN","SEARCH_COMPLETE","APP_CLOSE_BEGIN","APP_CLOSE_END"):
        assert f"B21_PHASE={marker}" in source
    assert "B21_SCIENTIFIC_COMPLETE=TRUE" in source
    assert "B21_SCIENTIFIC_COMPLETE=FALSE" in source
    assert "B21_TEARDOWN_EXCEPTION_AFTER_SCIENTIFIC_COMPLETE=TRUE" in source
    assert "traceback.print_exc()" in source and source.rstrip().endswith("raise")


def test_completion_markers_are_mutually_exclusive_by_control_flow():
    source=(Path(__file__).parents[1]/"scripts"/"v2b21_action_space_legal_return_feasibility_v1.py").read_text()
    tree=ast.parse(source)
    guards=[node for node in ast.walk(tree) if isinstance(node,ast.If) and isinstance(node.test,ast.Name) and node.test.id=="SCIENTIFIC_COMPLETE"]
    assert len(guards)==1
    guarded=guards[0]
    true_text=ast.get_source_segment(source,guarded.body[0]); false_text="\n".join(ast.get_source_segment(source,x) for x in guarded.orelse)
    assert "TEARDOWN_EXCEPTION_AFTER_SCIENTIFIC_COMPLETE=TRUE" in true_text
    assert "B21_SCIENTIFIC_COMPLETE=FALSE" not in true_text
    assert "B21_RUNTIME_EXCEPTION=TRUE" in false_text and "B21_SCIENTIFIC_COMPLETE=FALSE" in false_text


def test_runner_uses_fresh_assembler_per_case_and_protocol_markers():
    source=(Path(__file__).parents[1]/"scripts"/"v2b21_action_space_legal_return_feasibility_v1.py").read_text()
    assert "fresh_reset(); assembler=OneDecisionCommandAssembler();" in source
    assert source.count("assembler=OneDecisionCommandAssembler()") == 1
    for marker in ("TIER1_CASES=27","TIER2_ADDITIONAL_CASES=98","MAX_UNIQUE_SEARCH_CASES=125","WITNESS_SELECTION_RULE=FIRST_LEGAL_IN_PREDEFINED_GRID_ORDER","WITNESS_REPEAT_REQUIRED=5"):
        assert marker in source
