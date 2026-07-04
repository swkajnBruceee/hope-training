"""
Python module serving as a project/extension template.
"""

# Isaac Sim 4.5+ extension submodules (e.g. usdrt.Usd, omni.physics.tensors)
# are not guaranteed to be available during bare Python imports. Keep the
# package importable in headless training setups and let downstream code touch
# the optional module only when it is actually present.
try:  # pragma: no cover - optional runtime dependency
    import usdrt.Usd  # noqa: F401
except ModuleNotFoundError:
    pass

# Headless pip installs of Isaac Sim 4.5 pull in UI extensions when importing
# omni.replicator.core, which crashes in a headless AppLauncher context. The
# stub below provides only the replicator API that IsaacLab actually calls.
from ._isaacsim_headless_stub import *  # noqa: F401,F403
