"""Package containing task implementations for various robotic environments."""

##
# Register Gym environments.
##

# ``import_packages`` walks this tree and eagerly imports every submodule,
# which in turn pulls in ``isaaclab.managers`` -> ``omni.kit.app``. The pip
# metapackage ``isaacsim`` does NOT ship ``omni.kit`` (it ships only the
# headless runtime), so an eager sweep fails on standard training boxes with
# ``ModuleNotFoundError: No module named 'omni.kit'`` BEFORE the simulator
# has even launched. Treat the sweep as a best-effort task registry; the
# user's main entry point (``scripts/train.py``) imports ``training`` only
# after ``AppLauncher`` has started the simulator, at which point ``omni.kit``
# is available.
try:
    from isaaclab_tasks.utils import import_packages
except ImportError:  # pragma: no cover -- depends on host Isaac install
    import_packages = None  # type: ignore[assignment]

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
if import_packages is not None:
    import_packages(__name__, _BLACKLIST_PKGS)
