"""Runtime path fixes for pip-installed Isaac Sim training environments.

This module is auto-imported by Python when the whole_body_tracking repo root is
present on ``sys.path``. ``setup_train_env.sh`` already prepends that repo root
to ``PYTHONPATH``, so this is the most reliable place to patch Isaac Sim's
namespace-package search path without depending on the old source layout.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path


def _site_packages() -> Path:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return Path(sys.prefix) / "lib" / version / "site-packages"


def _candidate_package_roots() -> list[str]:
    base = _site_packages()
    roots = (
        base / "isaacsim" / "exts",
        base / "isaacsim" / "extsDeprecated",
        base / "isaacsim" / "extscache",
        base / "isaacsim" / "extsPhysics",
        base / "omni" / "exts",
        base / "omni" / "extscore",
    )
    paths: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for entry in sorted(glob.glob(str(root / "*"))):
            pkg_root = Path(entry)
            if (pkg_root / "isaacsim").is_dir():
                paths.append(str(pkg_root / "isaacsim"))
            if (pkg_root / "omni").is_dir():
                paths.append(str(pkg_root / "omni"))
            if (pkg_root / "pxr").is_dir() or (pkg_root / "usdrt").is_dir():
                paths.append(str(pkg_root))

            include_dir = pkg_root / "include"
            if (include_dir / "isaacsim").is_dir():
                paths.append(str(include_dir / "isaacsim"))
            if (include_dir / "omni").is_dir():
                paths.append(str(include_dir / "omni"))
            if (include_dir / "pxr").is_dir() or (include_dir / "usdrt").is_dir():
                paths.append(str(include_dir))
    return paths


def _extend_package_path(pkg_name: str, extra_paths: list[str]) -> None:
    try:
        pkg = __import__(pkg_name)
    except Exception:
        return

    pkg_path = list(getattr(pkg, "__path__", []))
    for path in extra_paths:
        if path not in pkg_path:
            pkg_path.append(path)
        if path not in sys.path:
            sys.path.insert(0, path)
    pkg.__path__ = pkg_path


if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").strip().lower() in {"yes", "true", "1", "y"}:
    extra_paths = _candidate_package_roots()
    _extend_package_path("isaacsim", extra_paths)
    _extend_package_path("omni", extra_paths)
