# v13b runtime contents

This directory is the portable inference layer for the three checkpoint
bundle. It does not depend on IsaacLab and does not silently invent robot
state. Install PyTorch and run:

```bash
python -m runtime.dry_run
```

The smoke test verifies all four asset hashes, reconstructs the actor MLPs,
loads all three observation normalizers, and performs zero-observation
inference:

```text
model_3396: 126 -> 14
model_900:    56 -> 10
model_5000:   98 -> 26
```

`ThreeModelRuntime.infer()` expects observations that already obey
`contracts/observation_contract.yaml`. `blend_targets()` expects physical
prior targets, not raw prior actions. The caller is responsible for creating
those targets from the MuJoCo/robot state, including:

- model3396's historical 126D private observation and normalizer;
- model900's private 56D observation, 50-step READY prelude, shoulder
  lookahead and task-phase velocity feed-forward;
- pelvis/root-local 10D goal and signed time-to-hit;
- joint limits, velocity/torque limits, rate limits and emergency stop.

The MuJoCo adapter is included in `mujoco_adapter.py`. A contract smoke
against the supplied A3 XML is:

```bash
PYTHONPATH=model_deployment/v13b_three_model_runtime_20260810 \
  python -m runtime.mujoco_contract_smoke
```

It checks real MuJoCo state extraction, 31-DOF mapping, racket FK, all three
observation widths, READY initialization, prior gating, limits and finite
physics using the bundled Isaac-compatible XML. Pass `--xml` to compare
another model. With no reference file, the smoke deliberately disables prior
target blending; it still loads and evaluates all three actors, but does not
pretend READY is the private motion reference. Construct
`NpzReferenceProvider` and pass `enable_priors=True` for the complete-prior
path. This is a wiring smoke, not a dynamic standing qualification.

For the project's real reference bank, use `MotionManifestReferenceProvider`.
It reads the existing manifest and lazily loads one selected 50 Hz motion
payload, including the 50-step READY prelude and 8/12/16-step previews:

```bash
PYTHONPATH=model_deployment/v13b_three_model_runtime_20260810 \
  python -m runtime.mujoco_contract_smoke \
  --manifest model_deployment/v13b_three_model_runtime_20260810/references/training_reference_bank_merged_20260807/training_manifest.json \
  --motion-index 0 --steps 5
```

The smoke defaults to the MuJoCo-only `isaac_passive_stable` low-level
profile. It uses the source-snapshot passive-stability leg/foot gains, a
bounded support-feedback layer evaluated before every `mj_step`, and a
0.12-rad lower-body READY envelope for plant qualification. To inspect the
unassisted official gain map, pass `--low-level-profile official_pd`.
Neither profile changes `hardware_command()`: simulation balance torque and
MuJoCo bias compensation never enter hardware `tau_ff`.

`--motion-index` gives deterministic replay; the provider also supports
nearest-target selection when a planner supplies the target before the swing.
Missing arrays or an incompatible joint order raise an error. The bundled
`ReadyHoldReference` remains only for reset/stand wiring checks.

The exact IsaacLab-side implementation used as the contract reference is
copied under `source_snapshot/`; see its README for the relevant entry
points. It is intentionally kept separate from this portable loader because
it imports the larger training environment.

The guarded candidate replay with manifest motion 0 has been checked for 250
50-Hz control steps: the root remains near 1.04 m with the passive-stable
profile. This is a low-level/plant qualification case, not proof that every
motion, target, or real robot route is dynamically stable.
