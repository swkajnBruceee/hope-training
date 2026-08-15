"""Admission smoke test for the self-contained three-model runtime."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from .v13b_runtime import ThreeModelRuntime


def verify_sha256(root: Path) -> None:
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split(maxsplit=1)
        path = root / relative
        # The repository manifest was generated from the repository root and
        # therefore contains ``model_deployment/<package>/...``.  The same
        # manifest is also shipped inside the package zip, where paths are
        # naturally relative to this directory.
        if not path.is_file():
            parts = Path(relative).parts
            if root.name in parts:
                path = root.joinpath(*parts[parts.index(root.name) + 1 :])
        if not path.is_file():
            raise FileNotFoundError(f"asset listed by SHA256SUMS does not exist: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"SHA256 mismatch: {relative}: {digest} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="v13b_three_model_runtime_20260810 directory",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    root = Path(args.package_root).expanduser().resolve()
    verify_sha256(root)
    runtime = ThreeModelRuntime(root, device=args.device)
    outputs = runtime.infer(
        torch.zeros(1, 98),
        torch.zeros(1, 126),
        torch.zeros(1, 56),
    )
    print("SHA256: OK")
    for policy in (runtime.student, runtime.lower, runtime.upper):
        print(policy.describe())
    print("zero-observation inference: OK")
    print("student action:", tuple(outputs.student_action.shape))
    print("lower prior action:", tuple(outputs.lower_prior_action.shape))
    print("upper prior action:", tuple(outputs.upper_prior_action.shape))


if __name__ == "__main__":
    main()
