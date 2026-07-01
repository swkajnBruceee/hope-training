"""
Python module serving as a project/extension template.
"""

# Isaac Sim 4.5+ extension submodules (e.g. usdrt.Usd, omni.physics.tensors)
# are namespace packages that are not auto-loaded by Python. Import them here
# before any IsaacLab/Isaac Sim module touches them.
import usdrt.Usd  # noqa: F401

# Headless pip installs of Isaac Sim 4.5 pull in UI extensions when importing
# omni.replicator.core, which crashes in a headless AppLauncher context. The
# stub below provides only the replicator API that IsaacLab actually calls.
from ._isaacsim_headless_stub import *  # noqa: F401,F403

# Register Gym environments.
from .tasks import *
