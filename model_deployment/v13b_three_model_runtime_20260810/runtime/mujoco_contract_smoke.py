"""Deterministic MuJoCo contract smoke for the v13b adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .mujoco_adapter import (
    MotionManifestReferenceProvider,
    MujocoV13BAdapter,
    StrikeTarget,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--motion-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--qualify-standing",
        action="store_true",
        help="assert the guarded MuJoCo standing envelope over all requested steps",
    )
    parser.add_argument(
        "--low-level-profile",
        choices=("official_pd", "isaac_passive_stable"),
        default="isaac_passive_stable",
        help="MuJoCo-only drive/balance profile; never used for hardware gains",
    )
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    default_xml = package / "models/a3_v13b_isaac_compatible/mjcf/a3_v13b_isaac_compatible.xml"
    if not default_xml.is_file():
        # The portable package is shared with the mc_ltl visual runtime.  Use
        # its grounded model when the package-local optional model copy is not
        # present, while still allowing --xml to select any explicit asset.
        sibling_xml = package.parents[2] / "mc_ltl/runtime/mujoco_v13b/model/mjcf/a3_pingpong_grounded.xml"
        if sibling_xml.is_file():
            default_xml = sibling_xml
    xml = args.xml or default_xml
    reference = None
    if args.manifest is not None:
        reference = MotionManifestReferenceProvider(args.manifest, motion_index=args.motion_index)
    adapter = MujocoV13BAdapter(
        xml,
        package,
        reference=reference,
        enable_priors=reference is not None,
        low_level_profile=args.low_level_profile,
    )
    adapter.reset()
    allowed_bodies = {"left_ankle_roll_Link", "right_ankle_roll_Link", "world"}
    for index in range(adapter.data.ncon):
        contact = adapter.data.contact[index]
        body_names = {
            adapter.model.body(int(adapter.model.geom_bodyid[contact.geom1])).name,
            adapter.model.body(int(adapter.model.geom_bodyid[contact.geom2])).name,
        }
        if not body_names.issubset(allowed_bodies):
            raise AssertionError(f"unexpected reset contact bodies: {body_names}")
    if reference is None:
        target = StrikeTarget(
            position_world=np.asarray((-0.2, -0.5, 1.1), dtype=np.float32),
            velocity_world=np.zeros(3, dtype=np.float32),
            normal_world=np.asarray((0.0, 1.0, 0.0), dtype=np.float32),
            hit_time_s=1.0,
        )
    else:
        target = reference.target_for(adapter)
    min_root_z = float("inf")
    max_tilt_rad = 0.0
    for _ in range(max(1, args.steps)):
        result = adapter.step(target)
        assert result.student_observation.shape == (98,)
        assert result.lower_observation.shape == (126,)
        assert result.upper_observation.shape == (56,)
        assert np.isfinite(result.target_joint_positions).all()
        assert np.isfinite(adapter.data.qpos).all()
        assert np.isfinite(adapter.data.qvel).all()
        min_root_z = min(min_root_z, float(adapter.data.qpos[2]))
        gravity = adapter._root_state()[2]
        max_tilt_rad = max(max_tilt_rad, float(np.arccos(np.clip(-gravity[2], -1.0, 1.0))))
    if args.qualify_standing:
        if adapter.low_level_config.name != "isaac_passive_stable":
            raise AssertionError("standing qualification requires isaac_passive_stable")
        if min_root_z < 0.85 or max_tilt_rad > 0.60:
            raise AssertionError(
                f"standing qualification failed: min_root_z={min_root_z:.3f} "
                f"max_tilt_rad={max_tilt_rad:.3f}"
            )
    print(
        "mujoco contract PASS: "
        f"xml={xml} policy_dt={adapter.control_dt:.6f}s decimation={adapter.control_decimation} "
        f"priors={'complete' if adapter.enable_priors else 'disabled-for-wiring-smoke'} "
        f"low_level={adapter.low_level_config.name} "
        f"reference={getattr(reference, 'motion_path', None)} "
        f"steps={adapter.step_index} root={adapter.data.qpos[:3].tolist()} "
        f"min_root_z={min_root_z:.3f} max_tilt_rad={max_tilt_rad:.3f}"
    )


if __name__ == "__main__":
    main()
