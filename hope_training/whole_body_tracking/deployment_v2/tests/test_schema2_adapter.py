import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from hope_training.whole_body_tracking.deployment_v2.hope_open_source_contract import load_canonical_metadata
from hope_training.whole_body_tracking.deployment_v2.schema2_adapter import (
    AdapterStationMirror,
    FlightRevisionManager,
    IdentityError,
    TimingError,
    age_command_timing,
    build_schema2_packet,
    reanchor_to_wall,
)

MODEL = Path("/home/a104/hope_training_repo/hope-deploy-baseline/a3_deploy/a3_deploy_example/models/model_21800/policy/exported/policy.onnx")


def test_timing_aging_expiry_and_reanchor():
    assert age_command_timing(100.0, 100.2, 0.8) == pytest.approx(0.6)
    assert age_command_timing(100.0, 99.9, 0.8) == pytest.approx(0.8)
    with pytest.raises(TimingError):
        age_command_timing(100.0, 101.0, 0.8)
    strike, sec, nsec = reanchor_to_wall(0.6, 1_800_000_000.125)
    encoded = sec + nsec * 1e-9
    assert abs(strike - (1_800_000_000.125 + 0.6)) <= 1e-9
    assert abs(strike - (encoded + 0.6)) <= 1e-9


def test_flight_revision_monotonic_duplicate_and_reordered():
    manager = FlightRevisionManager(shot_reuse_tolerance_s=0.05)
    first = manager.observe_valid(100.0)
    second = manager.observe_valid(100.02)
    third = manager.observe_valid(101.0)
    assert (first.flight_id, first.revision_id) == (1, 1)
    assert (second.flight_id, second.revision_id) == (1, 2)
    assert (third.flight_id, third.revision_id) == (2, 1)
    guard = FlightRevisionManager()
    guard.validate_received_identity(1, 1, 1)
    with pytest.raises(IdentityError): guard.validate_received_identity(1, 1, 2)
    guard.validate_received_identity(2, 1, 2)
    with pytest.raises(IdentityError): guard.validate_received_identity(3, 1, 2)


def test_flight_side_lock_and_station_update_only_after_accept():
    metadata = load_canonical_metadata(MODEL)
    mirror = AdapterStationMirror(metadata)
    side, station = mirror.candidate_for(1, metadata.reach_offsets[0], [0, 0])
    assert side == 1
    assert mirror.held_station_xy is None
    # Same flight target strongly favors BH, but side remains locked FH.
    side2, station2 = mirror.candidate_for(1, metadata.reach_offsets[1], [0, 0])
    assert side2 == 1
    mirror.accept_candidate(1, side2, station2)
    np.testing.assert_allclose(mirror.held_station_xy, station2)


def make_packet():
    return build_schema2_packet(
        valid=True, swing_sign=1, position=[0.58, -0.44, 1.0],
        velocity=[2.0, 0.31, 0.88], control_tts_s=0.6,
        producer_wall_s=1_800_000_000.125, command_seq=1,
        flight_id=1, revision_id=1, estimator_sample_count=0,
        estimator_span_s=0.0,
    )


def test_schema_packet_shape_dtype_indices_and_zero_metadata():
    p = make_packet()
    assert p.shape == (19,) and p.dtype == np.float64
    np.testing.assert_allclose(p[:12], [2, 1, 1, 0.58, -0.44, 1.0, 2.0, 0.31, 0.88, 0.6, p[10], 0])
    assert tuple(p[14:19]) == (1.0, 1.0, 1.0, 0.0, 0.0)
    assert abs(p[10] - (p[12] + p[13] * 1e-9 + p[9])) <= 1e-9


def test_nan_inf_rejected():
    with pytest.raises(ValueError): age_command_timing(np.nan, 1, 1)
    with pytest.raises(ValueError): reanchor_to_wall(1, np.inf)
    kwargs = dict(valid=True, swing_sign=1, position=[0, 0, np.nan], velocity=[1, 1, 1],
                  control_tts_s=1, producer_wall_s=100, command_seq=1,
                  flight_id=1, revision_id=1)
    with pytest.raises(ValueError): build_schema2_packet(**kwargs)


def test_reference_parser_head_semantics_and_coverage_limit():
    reference = Path("/home/a104/hope_training_repo/hope-deploy-baseline/a3_deploy/a3_deploy_example/reference")
    sys.path.insert(0, str(reference))
    try:
        from a3_deploy_onnx_ref_pingpong.ros_command_source import parse_flat_racket_command
        parsed = parse_flat_racket_command(make_packet())
        assert parsed is not None
        assert parsed.task_id == 1 and parsed.task_revision == 1
        assert parsed.swing_sign == 1
        assert parsed.position == pytest.approx((0.58, -0.44, 1.0))
        assert parsed.velocity == pytest.approx((2.0, 0.31, 0.88))
        assert parsed.time_to_strike == pytest.approx(0.6)
        # Reference parser deliberately ignores frame, producer timestamp,
        # command_seq, estimator count/span and absolute strike time.
    finally:
        sys.path.remove(str(reference))
