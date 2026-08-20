import numpy as np
from hope_training.whole_body_tracking.deployment_v2_isaac.model21800_executor import bind_target_to_observation

def test_exact_target_slices():
    o=bind_target_to_observation(np.zeros(110),target_position_world=(1,2,3),base_position_world=(.5,.5,1),target_velocity_world=(2,.3,.8),time_to_strike_s=.4)
    assert np.allclose(o[103:106],[.5,1.5,2]) and np.allclose(o[106:109],[2,.3,.8]) and np.isclose(o[109],.4)
