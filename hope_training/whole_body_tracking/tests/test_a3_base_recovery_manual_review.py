import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "manual_review", ROOT / "tools/analyze_a3_base_recovery_manual_review.py"
)
manual_review = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(manual_review)


def test_stats_and_rolling_rms_use_only_numpy_synthetic_data():
    values = np.arange(1, 6, dtype=np.float32)
    summary = manual_review.stats(values)
    assert summary["count"] == 5
    assert summary["median"] == 3.0
    assert summary["mad"] == 1.0
    assert summary["iqr"] == 2.0

    matrix = np.array([[3.0], [4.0], [0.0]], dtype=np.float32)
    rms = manual_review.rolling_rms(matrix, 2)
    np.testing.assert_allclose(rms[:, 0], [3.0, np.sqrt(12.5), np.sqrt(8.0)])


def test_episode_events_detect_recovery_exit_and_reentry():
    dt = 0.1
    values = np.full((12, 1), 2.0, dtype=np.float32)
    values[2:6] = 0.5
    values[6:8] = 1.6
    values[8:] = 0.5
    arrays = {"abs_pelvis_roll_rad": values}
    envelope = {
        "abs_pelvis_roll_rad": {"enter_threshold": 1.0, "exit_threshold": 1.5}
    }
    event = manual_review.episode_events(
        arrays, np.ones((12, 1), dtype=bool), envelope, dt, dwell_s=0.2
    )[0]
    assert event["first_envelope_entry_step"] == 2
    assert event["first_dwell_completion_step"] == 3
    assert event["first_recovery_s"] == 0.30000000000000004
    assert event["recovery_time_s"] == event["recovery_confirmed_s"]
    assert event["post_recovery_exit_count"] == 1
    assert event["longest_exit_steps"] == 3
    assert event["recovered_again"] is True
    assert event["final_1s_inside_fraction"] == 0.8
    assert event["classification"] == "core_body_reinstability"


def test_spike_classification_requires_aux_only_and_at_most_two_steps():
    channel = "abs_joint_velocity_rad_s/right_ankle_pitch_joint"
    envelope = {channel: {"enter_threshold": 1.0, "exit_threshold": 1.5}}
    active = np.ones((10, 2), dtype=bool)
    values = np.full((10, 2), 0.5, dtype=np.float32)
    values[4:6, 0] = 2.0
    values[4:7, 1] = 2.0
    events = manual_review.episode_events(
        {channel: values}, active, envelope, 0.1, dwell_s=0.2
    )
    assert events[0]["classification"] == "numerical_or_contact_spike_contact_unverified"
    assert events[1]["classification"] == "ankle_velocity_only"


def test_hysteresis_changes_exits_but_not_first_entry():
    channel = "abs_pelvis_roll_rad"
    values = np.array([[2.0], [0.5], [0.5], [1.1], [0.5], [0.5]], dtype=np.float32)
    active = np.ones_like(values, dtype=bool)
    no_hysteresis = {channel: {"enter_threshold": 1.0, "exit_threshold": 1.0}}
    light = {channel: {"enter_threshold": 1.0, "exit_threshold": 1.2}}
    first = manual_review.episode_events(
        {channel: values}, active, no_hysteresis, 0.1, dwell_s=0.2
    )[0]
    second = manual_review.episode_events(
        {channel: values}, active, light, 0.1, dwell_s=0.2
    )[0]
    assert first["first_envelope_entry_step"] == second["first_envelope_entry_step"] == 1
    assert first["post_recovery_exit_count"] == 1
    assert second["post_recovery_exit_count"] == 0


def test_manifest_entry_has_replay_window_and_camera_focus():
    entry = manual_review._manifest_entry(
        "candidate",
        {
            "env_id": 3,
            "first_exit_s": 2.0,
            "exit_trigger_metrics": ["abs_pelvis_pitch_rad"],
        },
        42,
        "core_body_top10",
    )
    assert entry["time_window_s"] == [1.0, 4.0]
    assert entry["suggested_camera_focus"] == "pelvis"
    assert entry["trace_index"] == 42


def _one_channel_events(values, *, channel="abs_pelvis_roll_rad", dwell_s=0.2):
    matrix = np.asarray(values, dtype=np.float32)[:, None]
    return manual_review.episode_events(
        {channel: matrix},
        np.ones_like(matrix, dtype=bool),
        {channel: {"enter_threshold": 1.0, "exit_threshold": 1.5}},
        0.1,
        dwell_s=dwell_s,
    )[0]


def test_outside_band_crossings_do_not_create_extra_exit_without_dwell():
    # Recover at step 1, exit at 2, then cross exit repeatedly without two
    # consecutive enter-envelope samples.
    event = _one_channel_events([0.5, 0.5, 2.0, 1.2, 2.0, 0.5, 1.2, 2.0])
    assert event["post_recovery_exit_count"] == 1
    assert event["exit_events"][0]["exit_threshold_violation_steps"] == 3
    assert event["exit_events"][0]["recovered_again"] is False
    assert event["exit_events"][0]["outside_state_steps"] == 6
    assert event["durable_recovery"] is False


def test_completed_redwell_allows_second_exit_cycle():
    event = _one_channel_events(
        [0.5, 0.5, 2.0, 0.5, 0.5, 1.2, 2.0, 0.5, 0.5]
    )
    assert event["post_recovery_exit_count"] == 2
    assert event["exit_events"][0]["recovery_confirmed_step"] == 4
    assert event["exit_events"][1]["recovery_confirmed_step"] == 8
    assert event["durable_recovery"] is True
    assert event["durable_recovery_step"] == 8


def test_exit_events_are_classified_independently_for_ankle_and_core():
    ankle = "abs_joint_velocity_rad_s/right_ankle_pitch_joint"
    core = "abs_pelvis_roll_rad"
    arrays = {
        ankle: np.array([[0.5], [0.5], [2.0], [0.5], [0.5], [0.5], [0.5]]),
        core: np.array([[0.5], [0.5], [0.5], [0.5], [0.5], [2.0], [2.0]]),
    }
    envelope = {
        name: {"enter_threshold": 1.0, "exit_threshold": 1.5}
        for name in arrays
    }
    event = manual_review.episode_events(
        arrays, np.ones((7, 1), dtype=bool), envelope, 0.1, dwell_s=0.2
    )[0]
    assert [item["classification"] for item in event["exit_events"]] == [
        "numerical_or_contact_spike_contact_unverified",
        "core_body_reinstability",
    ]
    assert event["classification"] == "multiple_exit_categories"


def test_no_exit_and_completion_time_and_final_stability_semantics():
    event = _one_channel_events([2.0, 0.5, 0.5, 0.5, 0.5])
    assert event["classification"] == "no_post_recovery_exit"
    assert event["exit_events"] == []
    assert event["dwell_start_s"] == 0.1
    assert event["recovery_time_s"] == 0.2
    assert event["durable_recovery"] is True
    assert event["durable_recovery_time_s"] == 0.2
    assert event["final_1s_stable"] is False  # 4/5 = 0.80, below 0.95.

    summary = manual_review.summarize_events([event], 0)
    assert summary["transient_recovery_rate"] == 1.0
    assert summary["durable_recovery_rate"] == 1.0
    assert summary["final_1s_stable_rate"] == 0.0
    assert summary["exit_cycle_count"] == 0
