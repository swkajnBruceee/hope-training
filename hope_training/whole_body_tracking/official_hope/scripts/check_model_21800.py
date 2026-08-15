"""Host-only validation for the model_21800 checkpoint."""

from __future__ import annotations

import hashlib
import pathlib
import sys


EXPECTED_SHA256 = "69ad47f206bb9da263102488b243bf3b750f09608078a354beee663c79f0fb6b"
EXPECTED_SHAPES = {
    "actor.0.weight": (512, 110),
    "actor.2.weight": (256, 512),
    "actor.4.weight": (128, 256),
    "actor.6.weight": (31, 128),
}


def _root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    checkpoint = _root() / "checkpoints" / "model_21800.pt"
    if not checkpoint.is_file():
        print(f"missing checkpoint: {checkpoint}", file=sys.stderr)
        return 2

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        print(f"sha256 mismatch: {digest}", file=sys.stderr)
        return 3

    try:
        import torch
    except ImportError as exc:
        print(f"checkpoint hash is valid, but torch is unavailable: {exc}", file=sys.stderr)
        return 4

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = payload.get("model_state_dict", payload)
    missing = []
    mismatched = []
    for name, expected in EXPECTED_SHAPES.items():
        tensor = actor.get(name)
        if tensor is None:
            missing.append(name)
        elif tuple(tensor.shape) != expected:
            mismatched.append(f"{name}: got {tuple(tensor.shape)}, expected {expected}")
    if missing or mismatched:
        print(f"actor layout mismatch; missing={missing}, mismatched={mismatched}", file=sys.stderr)
        return 5

    print(f"checkpoint: {checkpoint}")
    print(f"sha256: {digest}")
    print("actor: obs_dim=110 action_dim=31 hidden=[512,256,128]")
    print(f"iteration: {payload.get('iter', 'unknown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
