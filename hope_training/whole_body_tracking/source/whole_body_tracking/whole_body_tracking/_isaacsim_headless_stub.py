"""Headless-only stub for omni.replicator.core.

Isaac Sim 4.5's pip distribution of omni.replicator.core eagerly imports UI
extensions (omni.kit.window.material_graph -> omni.kit.context_menu -> ...)
when the module is imported. In a headless AppLauncher context those UI
extensions are not loaded and the import chain crashes with:

    TypeError: expected str, bytes or os.PathLike object, not NoneType

IsaacLab only needs rep.set_global_seed() from this module inside
ManagerBasedEnv.seed(). We provide a minimal stub so the seed path succeeds
without pulling in the broken UI imports.
"""

import sys
import types


def set_global_seed(seed: int) -> None:
    """No-op replicator seed stub for headless runs."""
    pass


# Only install the stub if the real module has not been imported yet.
if "omni.replicator.core" not in sys.modules:
    # Ensure the parent namespace package exists so that
    # `import omni.replicator.core as rep` resolves.
    if "omni.replicator" not in sys.modules:
        replicator_pkg = types.ModuleType("omni.replicator")
        replicator_pkg.__path__ = []
        sys.modules["omni.replicator"] = replicator_pkg
    sys.modules["omni.replicator.core"] = sys.modules[__name__]
