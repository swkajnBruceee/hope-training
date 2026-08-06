import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_recovery_trace_is_deterministic_balanced_and_fail_closed(tmp_path):
    tool = ROOT / "tools/build_a3_base_recovery_disturbance_trace.py"
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first_manifest = tmp_path / "first.json"
    second_manifest = tmp_path / "second.json"
    common = ["--samples-per-profile", "16", "--seed", "7"]
    subprocess.run(
        [sys.executable, tool, "--output", first, "--manifest", first_manifest, *common],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, tool, "--output", second, "--manifest", second_manifest, *common],
        check=True,
        capture_output=True,
        text=True,
    )
    first_payload = np.load(first, allow_pickle=False)
    second_payload = np.load(second, allow_pickle=False)
    for key in first_payload.files:
        np.testing.assert_array_equal(first_payload[key], second_payload[key])
    manifest = json.loads(first_manifest.read_text())
    assert manifest["training_distribution_approved"] is False
    assert manifest["deployment_approved"] is False
    for profile in manifest["profiles"][1:]:
        assert set(profile["roll_pitch_sign_quadrants"].values()) == {4}
        assert set(profile["angular_velocity_sign_quadrants"].values()) == {4}
