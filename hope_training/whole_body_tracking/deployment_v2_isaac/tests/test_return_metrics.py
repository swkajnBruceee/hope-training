import numpy as np
from hope_training.whole_body_tracking.deployment_v2_isaac.return_metrics import (
    NET_X, detect_opponent_table_bounce, detect_outgoing_net_crossing, legal_return,
)


def test_pre_contact_bounce_is_not_landing():
    assert not detect_opponent_table_bounce([2.0, -.5, .02], [2., 0., 1.], contact_seen=False)


def test_post_contact_gate_accepts_stage5_bounce():
    assert detect_opponent_table_bounce([2.0, -.5, .02], [2., 0., 1.], contact_seen=True)


def test_net_crossing_interpolation_and_clearance():
    event = detect_outgoing_net_crossing([1.36, -.5, .16], [1.38, -.4, .20], [2., 0., 0.], contact_seen=True)
    assert event is not None and np.allclose(event.position, [NET_X, -.45, .18])
    assert np.isclose(event.clearance, .0075) and event.clears_net


def test_opponent_table_bounds_are_inclusive_and_reject_outside():
    assert detect_opponent_table_bounce([1.40, -1.495, -.005], [.21, 0., .051], contact_seen=True)
    assert not detect_opponent_table_bounce([1.399, -.5, .02], [2., 0., 1.], contact_seen=True)


def test_legal_return_boolean_composition():
    assert legal_return(True, True, True)
    for values in ((False, True, True), (True, False, True), (True, True, False)):
        assert not legal_return(*values)
