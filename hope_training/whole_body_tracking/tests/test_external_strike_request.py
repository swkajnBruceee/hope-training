import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.utils.external_strike_request import load_external_strike_request


def test_loads_minimal_position_and_optional_hit_time(tmp_path: Path):
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(
            {
                "request_id": "mocap-17",
                "target_position_b": [0.5, -0.1, 0.2],
                "hit_time_s": 2.06,
            }
        ),
        encoding="utf-8",
    )
    request = load_external_strike_request(path)
    assert request["request_id"] == "mocap-17"
    assert request["target_position_b"] == [0.5, -0.1, 0.2]
    assert request["hit_time_s"] == pytest.approx(2.06)


def test_rejects_unknown_fields_and_bad_target_shape(tmp_path: Path):
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps({"target_position_b": [0.5, 0.1], "unexpected": 1}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported fields"):
        load_external_strike_request(path)

    path.write_text(json.dumps({"target_position_b": [0.5, 0.1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="three finite numbers"):
        load_external_strike_request(path)
