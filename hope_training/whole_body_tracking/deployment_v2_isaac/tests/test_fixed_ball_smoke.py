import numpy as np
from hope_training.whole_body_tracking.deployment_v2_isaac.fixed_ball_smoke import *
from hope_training.whole_body_tracking.deployment_v2 import load_canonical_metadata, select_nearest_station_side
from hope_training.whole_body_tracking.deployment_v2_isaac.v2_one_ball_env import MODEL

def test_fixed_source_and_hope_intercept():
    x=predict_fixed_intercept()
    assert np.allclose(BALL_INITIAL_POSITION,[2.22,-.5375,.43])
    assert x.position[0]==0 and x.flight_time_s>0 and x.bounce_count==1 and np.isfinite(np.r_[x.position,x.velocity]).all()

def test_fixed_incoming_selects_bh_at_canonical_station():
    x=predict_fixed_intercept(); meta=load_canonical_metadata(MODEL)
    sign,_=select_nearest_station_side(x.position[:2],[-.5,-.7625],meta)
    assert sign==-1
