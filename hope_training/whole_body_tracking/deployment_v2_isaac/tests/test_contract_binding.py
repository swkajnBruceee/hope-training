import numpy as np
import pytest
from hope_training.whole_body_tracking.deployment_v2_isaac.v2_one_ball_env import OneDecisionCommandAssembler
from hope_training.whole_body_tracking.deployment_v2_isaac.predictor_observation import PredictorCommand

def cmd(y=-0.44): return PredictorCommand((0.0,y,1.0),(-3.0,0.2,-0.5),10.0,0.5)

@pytest.mark.parametrize("action", [(-1,-1,-1),(0,0,0),(1,1,1)])
@pytest.mark.parametrize("y,base,expected", [(-0.44,( -0.58,0.0),1),(-0.09,(-0.58,0.04),-1)])
def test_six_contract_cases(action,y,base,expected):
    out=OneDecisionCommandAssembler().build(flight_id=1,revision_id=1,command_seq=1,predictor_command=cmd(y),source_now_s=10.1,producer_wall_s=100.0,current_base_xy=base,normalized_action=action)
    assert out.observation.shape==(7,) and out.normalized_action.shape==(3,)
    assert out.swing_sign==expected and out.schema2.shape==(19,)

def test_flight_side_and_action_lock():
    a=OneDecisionCommandAssembler(); kw=dict(flight_id=4,command_seq=1,predictor_command=cmd(),source_now_s=10.0,producer_wall_s=100.0,current_base_xy=(-.58,0),normalized_action=(0,0,0))
    first=a.build(revision_id=1,**kw)
    second=a.build(revision_id=2,**{**kw,"command_seq":2,"predictor_command":cmd(-.09)})
    assert first.swing_sign==second.swing_sign
    with pytest.raises(ValueError): a.build(revision_id=3,**{**kw,"command_seq":3,"normalized_action":(.1,0,0)})
