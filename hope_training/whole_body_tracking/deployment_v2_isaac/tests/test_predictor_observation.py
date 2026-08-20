import numpy as np
import pytest
from hope_training.whole_body_tracking.deployment_v2_isaac.predictor_observation import *

def test_raw_layout_and_age():
    o=build_v2_observation(PredictorCommand((0,-.2,1.1),(-3,.1,-.4),10,0.6),10.2,-1)
    assert o.dtype==np.float32 and o.tolist()==pytest.approx([-.2,1.1,-3,.1,-.4,.4,-1])

def test_reject_bad():
    with pytest.raises(Exception): build_v2_observation(PredictorCommand((0,0,1),(1,2,3),0,0),1,1)
