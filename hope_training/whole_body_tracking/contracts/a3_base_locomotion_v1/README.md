# A3 Base Locomotion v1 contracts

These files are the dependency-free Phase 0 interface candidate for the A3
Base Policy, Strike v2 reference, and unique 31-DOF Command Composer.

Current status: **structurally valid; approved only for the bounded deterministic
Stand smoke described by `stand_fixture_gate_v1.json`; not approved for long
training, locomotion, or deployment**.  Candidate action scales, waist-pitch
residual limit, command-frame signs, final transport, timeout behavior, and
normalization still require their later gates.

`stand_authority_candidate_gate_v1.json` separately permits one exact
100-iteration simulation diagnostic with only waist-pitch Kp/Kd changed to
350/7. It does not revise the v1 execution contract or promote that gain.

`stand_clip_candidate_gate_v1.json` independently permits one exact
100-iteration simulation diagnostic with the original gains and only the
normalized action clip changed from 0.25 to 0.5. It likewise cannot promote
the action contract.

`stand_authority_clip_candidate_gate_v1.json` permits the fourth and final
cell of the bounded 2x2 authority diagnostic. It explicitly stops further
authority parameter enumeration after the deterministic comparison.

`stand_authority_factorial_decision_v1.json` records that comparison. The
combined cell is retained only for follow-up diagnostics; neither gain nor
action authority is approved, and an extended smoke requires a new gate.

`stand_causal_audit_decision_v1.json` records the follow-up failure timeline,
reward accounting, waist-freeze, contact-geometry, and zero-action evidence.
It inserts a static-working-point qualification before any additional PPO.
The current URDF importer uses convex-hull collision rather than raw triangle
mesh contact; a foot-only conservative sole box fixes the initial load
asymmetry but does not fix the dynamic failure.  All training promotion flags
remain false.

The matching unapproved PPO candidate is
`cfg/algo/ppo_a3_base_stand_low_noise_candidate.yaml`.  It changes only the
initial exploration standard deviation from `1.0` to `0.15`; the global A3 PPO
configuration is intentionally untouched.  `scripts/train.py` rejects all
additional Base Stand PPO while the final decision's
`additional_ppo_smoke_approved` flag is false.

Run structural and source-asset validation:

```bash
cd hope_training/whole_body_tracking
python3 tools/validate_a3_base_contract.py
```

The training gate is deliberately fail-closed until approval flags are changed
after Phase 0 evidence:

```bash
python3 tools/validate_a3_base_contract.py --require-training-approved
# expected now: exit code 2
```

Generate a reference 31-DOF command without Isaac Lab or the deployment SDK:

```bash
python3 tools/compose_a3_command_reference.py \
  --base-action '[0,0,0,0,0,0,0,0,0,0,0,0,0,0]' \
  --strike-q-reference '[0,0,0.3,-0.12,0,0.8,0,0,0]'
```

The Python Composer is a semantic reference, not the production command
publisher. The future Isaac Action Term and C++ Composer must pass the same
`golden_composer_vectors.json` cases.

Generate the frozen Phase 0 experiment matrix:

```bash
python3 tools/build_a3_base_calibration_matrix.py --output /tmp/a3_base_phase0_matrix.json
```

Generate one candidate joint-step payload:

```bash
python3 tools/build_a3_base_calibration_command.py \
  --matrix /tmp/a3_base_phase0_matrix.json \
  --case-id step__a0.10__left_hip_pitch_joint__pos__r01 \
  --output /tmp/a3_base_left_hip_pitch_step.npz
```

The matrix does not command a simulator or approve any scale. It freezes case
IDs, amplitudes, repeats, transport variants, and the result schema before a
runner is allowed to collect evidence.

Command generation also does not authorize execution. The builder deliberately
rejects `command_basis` because no trained Base policy exists yet, and rejects
`target_transport` because that comparison must run at native simulator
substeps. A faster external publisher is not treated as simulator interpolation.
Any accepted payload still requires a resettable isolated simulator and exactly
one command publisher; it is not approved for hardware.

Run one native-substep fixture case without ROS/AimRT/network transport:

```bash
python3 tools/run_a3_base_mujoco_calibration.py \
  --matrix /tmp/a3_base_phase0_matrix.json \
  --case-id step__a0.10__left_hip_pitch_joint__pos__r01 \
  --model ../../agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml \
  --output /tmp/a3_base_left_hip_pitch_native.json
```

The native runner requires MuJoCo Python bindings exactly matching `3.1.6`.
Its `single_joint_fixture_v1` output is actuator/transport diagnostic evidence,
not free-base balance evidence. See `docs/A3_BASE_PHASE0_MUJOCO_PILOT.md`.
`stand_passive_stable_decision_v1.json` closes the zero-command PPO loop: the
Base14 PD_STAND plant is deterministically stable, but all learned Stand smoke
checkpoints are rejected for excessive raw-action clipping.  Further policy
learning moves to a separately gated small-disturbance recovery task.
