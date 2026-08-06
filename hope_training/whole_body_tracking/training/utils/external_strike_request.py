"""Strict file contract for a single external strike request."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def load_external_strike_request(path: str | Path) -> dict[str, Any]:
    """Load one target-only request produced by a tracking/prediction module.

    The file deliberately has a small, version-free schema:

    ``{"target_position_b": [x, y, z], "hit_time_s": optional, "request_id": optional}``
    """
    request_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"external strike request does not exist: {request_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"external strike request is not valid JSON: {request_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("external strike request must be a JSON object")
    unknown = set(payload) - {"target_position_b", "hit_time_s", "request_id"}
    if unknown:
        raise ValueError(
            "external strike request has unsupported fields: "
            f"{sorted(unknown)}"
        )
    if "target_position_b" not in payload:
        raise ValueError("external strike request requires target_position_b")
    target = payload["target_position_b"]
    if (
        not isinstance(target, list)
        or len(target) != 3
        or any(not isinstance(value, (int, float)) for value in target)
        or not all(math.isfinite(float(value)) for value in target)
    ):
        raise ValueError(
            "external strike request target_position_b must be three finite numbers"
        )
    hit_time_s = payload.get("hit_time_s")
    if hit_time_s is not None and (
        not isinstance(hit_time_s, (int, float))
        or not math.isfinite(float(hit_time_s))
        or float(hit_time_s) <= 0.0
    ):
        raise ValueError(
            "external strike request hit_time_s must be a positive finite number"
        )
    request_id = payload.get("request_id")
    if request_id is not None and not isinstance(request_id, (str, int, float)):
        raise ValueError("external strike request request_id must be a string or number")
    return {
        "request_path": str(request_path),
        "request_id": request_id,
        "target_position_b": [float(value) for value in target],
        "hit_time_s": None if hit_time_s is None else float(hit_time_s),
    }
