"""Runtime path fixes for pip-installed Isaac Sim on the training server.

This module is imported automatically by Python when it is present on
``sys.path``. The HOPE training launcher puts this directory first, so we can
teach the pip-installed Isaac Sim tree about its extension packages before the
training code imports :mod:`isaacsim.core.*`.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path


def _site_packages() -> Path:
    return Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"


def _candidate_package_roots() -> list[str]:
    base = _site_packages()
    roots = [
        base / "isaacsim" / "exts",
        base / "isaacsim" / "extsDeprecated",
        base / "isaacsim" / "extscache",
        base / "isaacsim" / "extsPhysics",
        base / "omni" / "exts",
        base / "omni" / "extscore",
    ]
    paths: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for entry in sorted(glob.glob(str(root / "*"))):
            p = Path(entry)
            if (p / "isaacsim").is_dir():
                paths.append(str(p / "isaacsim"))
            if (p / "omni").is_dir():
                paths.append(str(p / "omni"))
            if (p / "pxr").is_dir():
                paths.append(str(p))
            if (p / "usdrt").is_dir():
                paths.append(str(p))
            include_dir = p / "include"
            if (include_dir / "isaacsim").is_dir():
                paths.append(str(include_dir / "isaacsim"))
            if (include_dir / "omni").is_dir():
                paths.append(str(include_dir / "omni"))
            if (include_dir / "pxr").is_dir():
                paths.append(str(include_dir))
            if (include_dir / "usdrt").is_dir():
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
    _extra_paths = _candidate_package_roots()
    _extend_package_path("isaacsim", _extra_paths)
    _extend_package_path("omni", _extra_paths)
