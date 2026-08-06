# A3 Strike Master

> **Strategy transition (2026-07-18):** this document remains authoritative for
> immutable Strike targets, provenance, diagnostic replay, task-sample evidence,
> and strike qualification gates. It no longer defines the future normal-control
> balance/motion owner. Official MOTION is retained only as a comparison or
> startup/recovery facility, not as the normal Base controller. New Base/Strike
> ownership, the Strike v2 waist split, the unique 31-DOF Composer, and all future
> Stand/StrikeSupport/Locomotion development are governed by
> [`A3_BASE_LOCOMOTION_MASTER_PLAN.md`](A3_BASE_LOCOMOTION_MASTER_PLAN.md).
> Historical contracts below are not silently reinterpreted; data crosses into
> the new route only through an explicit versioned conversion and requalification.

Current status: Phase 0A inventory, Phase 1 documentation consolidation, and Phase 2–3 qualification infrastructure are complete. No K8/K12/K24 checkpoint is valid for warm start; no manifest is admitted to PPO until it carries `a3_strike_gate_v3` provenance and passes the official-standalone executor path.

Current blocker: the first direct body-drive fixed-stand contract has been frozen, but it still needs ten-repeat official `RobotIOBackend` standalone evidence, zero-compensation identity, and a positive/negative basis scan. It is a diagnostic contract, not a feedback-balance or real-robot deployment contract.

Prepare a review-only target packet from traceable source evidence:

```bash
python tools/extract_a3_review_target_candidate.py \
  --source-target-spec <source_target_spec.json> \
  --source-manifest <source_manifest.json> --episode-id <episode_id> \
  --source-dataset <dataset_identity> \
  --racket-mount-contract-id a3_pingpong_right_racket_site_red_face_y_v1 \
  --output artifacts/pilots/<episode_id>/review_target_candidate.json
```

After human review, copy only `proposed_target_input` into a separate reviewed
JSON and create the immutable target with `tools/create_a3_target_spec.py`.
The candidate packet is not a target specification and must not be supplied to
the builder, recorder, evaluator, or training entry point.

## Single pipeline

```text
immutable source target
  -> 31-DOF frozen command builder
  -> a3_deploy_example RobotIOBackend / standalone body-drive
  -> raw state + IMU + actual racket task samples
  -> qualification report
  -> low-dimensional smooth command compensation
  -> FH/BH pilot gates
  -> K8 rebuild and ball-outcome admission
  -> residual PPO
```

The target is never overwritten from executor output. Isaac remains a cheap screening/training environment, not the final pilot authority.

## Contracts and sources

- Executor contract: [`../contracts/a3_t2d5_body_drive_fixed_stand_diag_v1/executor_contract.json`](../contracts/a3_t2d5_body_drive_fixed_stand_diag_v1/executor_contract.json)
- Gate version: `a3_strike_gate_v3`
- Source target schema: `schema_version=1`, base-frame position/velocity/normal, source identity, hit time, racket mount ID, and `source_target_sha256`.
- Command hash: `canonical_command_payload_sha256`; it hashes joint names/order, timestamps, `q_des`, `dq_des`, `tau_ff`, `kp`, and `kd`, rather than NPZ container bytes.
- Project-side standalone runner: [`../standalone_replay/README.md`](../standalone_replay/README.md). It is injected into the vendor build through `CMAKE_PROJECT_INCLUDE`, so vendor sources remain unchanged.
- Official source hashes and Phase 0 inventory: `artifacts/_archive_not_for_training/20260718_pre_executor_contract/ARCHIVE_MANIFEST.json`.
- Legacy document archive (hash verified after extraction): `artifacts/_archive_not_for_training/20260718_pre_executor_contract/docs.tar.zst`; use `DOCS_ARCHIVE_MANIFEST.json` in the same directory for recovery.

The fixed-stand contract owns all 31 command fields. Waist 3 + right arm 7 are strike-optimized; left arm, legs, and head have explicit baseline ownership. Any native MOTION or feedback-balance path requires a new contract and full requalification.

## Operational gates

Training accepts only `dataset_status=active_training_candidate` manifests containing: `source_target_sha256`, `command_sha256`, `executor_contract_id`, `gate_version`, and `provenance_version`. Paths containing `_archive_not_for_training`, `diagnostic`, `relabel`, or `invalid` are rejected.

First-round qualification requires all of:

1. 10 deterministic standalone repeats, with position/velocity-vector/normal measurement noise below 10% of the bootstrap thresholds (`0.075 m`, `0.5 m/s`, `15 deg`);
2. explicit raw state, backend sync, and 50 Hz command timing; the three rates are reported separately;
3. zero-compensation canonical payload identity;
4. stable `+epsilon/-epsilon` responses for each retained basis;
5. immutable target hash throughout.

Only after a forehand and backhand pilot pass may a low-dimensional CMA-ES/CEM search be selected. Do not run frame-wise ILC, residual PPO, K8 rebuild, or TTMD6 expansion before then.

The runner writes raw synchronized state evidence and command timing. It intentionally does not invent racket FK: an official-model task exporter must write actual base-frame racket position, velocity, and normal before the qualification tool can consider repeatability.

### SIL connectivity evidence (not qualification)

On 2026-07-18, the local A3 MuJoCo/AimRT ROS2 simulator completed one
fixed-stand connectivity replay using the immutable smoke-stand reference.
The runner performed its 3 s PD-STAND gateway followed by 120 static 50 Hz
command samples (2.38 s). The evidence is ignored from training at
`artifacts/sil_connectivity/run_20260718_static_stand/` and records 3,199
aligned 31-DOF state rows, 120 command rows, a 500.12 Hz median state rate,
and zero reported synchronization skew. The command's canonical payload hash
is `34faa1c213d30af3033593b21640e7d5100d5c1e65428e7038b3e1b465222697`.

This proves only local transport, state alignment, PD-STAND, command replay,
and raw evidence conversion. It is not a task repeatability result: it has no
actual racket task samples, no ten-repeat analysis, no sensitivity scan, and
no ball-outcome gate.

On the same local SIL path, the official-model task-sample recorder was then
validated with a second static-stand replay. It wrote 280 paired pelvis/racket
samples over 2.808 s (100.00 Hz median), all finite with strictly increasing
capture-relative timestamps. The nearest sample to 0.460 s was at 0.457844 s
(2.156 ms alignment error); the largest recorded pose-pair timestamp skew was
9.997 ms. The output and matching state sidecar are at
`artifacts/sil_connectivity/run_20260718_static_stand_task_samples/`.
This only proves the official MuJoCo pose-to-base-frame exporter and its
qualification-file schema. It contains no immutable reviewed strike target, so
it is intentionally not admissible as one of the ten target-bound repeats.

### Target-bound SIL result: invalid pre-gate diagnostic, not qualified

After the forehand source and `+Y` red-face convention were explicitly
reviewed, ten local MuJoCo/RobotIOBackend attempts were recorded against the
immutable target and canonical 80-frame command. Their target and command
hashes were respectively `99fc59e8c9678420a4fa7a661ec42e8511cada66b722ec6fd7971bd6f76be784`
and `159347167c9850e398265a5f295934fa224b4b0b555e971597e00a7a9678e9b9`.
Visual inspection established that the floating-base robot entered those
attempts lying on the floor, then replayed while prone. The measured 1.474 m
position, 2.980 m/s velocity-vector, and 124.79 deg normal errors therefore
describe a fallen initial state, not a standing strike miss. The attempts are
invalid diagnostic evidence and cannot be used for target, repeatability, or
training conclusions.

The former direct body-drive recovery procedure (resetting the simulator to
keyframe 0 and sending a 3-second `PD_STAND`) is retained only as historical
connectivity evidence. It must not be used by the official balance route: a
keyframe reset hides the real zero-torque fall and `PD_STAND` is not the
factory recovery controller. The obsolete direct-body-drive attempts remain
at `artifacts/pilots/T002_001_gao01_7p52_9p52/sil_repeatability_20260718/`
for diagnosis only.

### Native-MOTION balance with right-arm-only strike: SIL stability pass

The replacement balance route uses the official AimSim `MOTION` state machine,
not the direct 31-DOF body-drive runner.  MOTION owns legs, waist, trunk, and
head.  The project-side executor sends the official arm-channel message at
100 Hz; the left-arm values are held at their initially measured posture and
only the seven right-arm values follow the canonical command.  It refuses to
start unless `MOTION` has completed its GET_UP transition and remains still
for two seconds.  During blend, replay, and hold it polls the official pelvis
and torso IMU at 50 Hz; a relative-pelvis tilt above 25 deg or either body
angular speed above 0.5 rad/s stops arm publication and marks the run failed.

On 2026-07-18 the immutable `T002_001_gao01_7p52_9p52` command passed this
native-MOTION stability gate in local SIL.  The 92 preflight samples had at
most 0.0011 deg relative pelvis drift.  Across 129 samples during the 2.56 s
blend/strike/hold sequence, the maximum relative pelvis tilt was 0.485 deg,
with maximum pelvis and torso angular speeds of 0.085 and 0.286 rad/s.  The
official action remained `MOTION` at both start and end.  Evidence is
`artifacts/pilots/T002_001_gao01_7p52_9p52/official_motion_right_arm_20260718_retry01/stability_report.json`;
the executable is `tools/run_a3_official_motion_arm_strike.py`.

This is a local-SIL stability pass only.  The official HTTP service exposes
IMU and joint state but not pelvis height, foot wrench, support polygon, ball
state, or racket pose.  It therefore does not establish target matching,
ball contact, real-robot safety, or permission to use a custom waist path.

### Fixed official startup and recovery contract

This is the default and only supported startup sequence for the official A3
AimSim table-tennis SIL route. It is implemented by
`tools/start_official_aimsim_a3.sh` and
`tools/activate_official_aimsim_sil.py`; it is not a direct 31-DOF posture
controller.

```text
MuJoCo starts in PASSIVE (zero torque)
  -> keep PASSIVE for 1.5 s so the free-base robot naturally falls
  -> DAMPING
  -> official GET_UP policy, until GET_UP_FINISHED
  -> MOTION
  -> wait for the existing two-second stillness/IMU gate
  -> publish only the project-owned right-arm command
```

`MOTION` remains the owner of legs, waist, trunk, head, and the balance
policy. The project must not send leg, waist, torso, or `PD_STAND` commands
on this route. The ping-pong table is visual-only during recovery, so it
cannot obstruct the natural fall or the official get-up motion. The route
never calls `/reset_simulation` as part of a trial.

The launcher starts the recovery helper before creating MuJoCo so that it can
wait up to 45 s for the simulator-backed action service. Once available, the
helper uses the official external-action envelope
`MotionControlAction_USE_EXT_CMD`; it has a 45 s GET_UP deadline. This removes
the previous timing race in which a manual request could arrive before the
simulator had state channels.

Cold-start acceptance requires all of the following:

1. helper log contains `PASSIVE`, `DAMPING`, `GET_UP_ING`, and
   `GET_UP_FINISHED`, in that order;
2. final action is `MotionControlAction_MOTION`;
3. `/baselink_position` reports pelvis height at least `0.75 m` after
   recovery; and
4. the existing native-MOTION stillness/IMU gate passes before any right-arm
   strike command is published.

The helper log is
`third_party/aimsim_official/logs/motion_control/get_up.stdout.log`. On the
2026-07-18 cold-start verification it reached `GET_UP_FINISHED` in about 20 s,
then entered `MOTION`; the post-recovery pelvis height was `1.074 m`. This is
local-SIL evidence only, not real-robot safety validation.

### Right-hand racket overlay: SIL stability pass

The official AimSim vendor package remains unmodified.  The project launcher
now generates and loads a project-side XML overlay that reuses the repository
`a3_pingpong` gripping-hand mesh, red/black racket meshes, and racket-face
collision mesh.  It replaces the stock right-hand visual, so the hand uses
the same established grip and wrist attachment as the earlier pingpong
simulator rather than the former primitive-racket approximation.  A separate
fixed 0.18 kg payload body is retained below `right_wrist_yaw_Link`; it adds
no joint, actuator, or control channel.  Its attachment transform is
`(0.21021, 0.032078, 0.032036)` m in the wrist-yaw frame.  The generated
`right_racket_center` site identifies the blade centre, and the red-face
contract is local racket-frame `+Y` via `right_racket_red_face_y`.  The
overlay defaults on for `tools/start_official_aimsim_a3.sh`; set
`AIMSIM_RACKET_MODEL=0` only when diagnosing the unmodified upstream model.

The launcher keeps its MOTION child alive after the official CLI has detached
the simulator, avoiding a false controller shutdown at startup.  On
2026-07-18, with the reused `a3_pingpong` mesh model loaded, the same
canonical right-arm-only replay passed in local SIL: 92 stationary preflight
samples showed at most 0.0052 deg relative pelvis drift; across 129 samples,
the maximum relative pelvis tilt was 0.500 deg, and pelvis/torso angular
speeds peaked at 0.0540/0.1592 rad/s.  `MOTION` remained active at both ends.
The raw report is
`artifacts/pilots/T002_001_gao01_7p52_9p52/official_motion_right_arm_pingpong_mesh_20260718`;
the model-generation contract is
`assets/official_aimsim_racket/racket_attachment_contract.json`.

This proves that the simulated robot carries the specified rigid paddle and
remains within the current IMU stability gates while replaying the arm
motion.  It still does **not** prove paddle/ball contact, a hit point, target
accuracy, calibrated real-paddle mass/inertia, or real-robot safety.

## Data promotion

`executor_ready_not_for_training` is not a strike-training data state. Promotion to `active_training_candidate` additionally requires a versioned ball-outcome gate (valid contact, centre margin, outgoing ball, net crossing, landing, and no penetration) with genuine incoming-ball state. Without that state, an item can only be a separate motion prior.

TTMD6 source conversion is frozen as `source_right_to_a3_minus_y + velocity_plane_neg`; this only fixes source coordinates/orientation construction, not training eligibility.

## History and cleanup

Historical documents, relabelled targets, old checkpoints, ILC iterations, and prior eval outputs are diagnostic material. Legacy documents were archived with per-file hashes after extraction verification. Phase 0E may physically archive/delete evaluation outputs and other generated assets only from the exact paths listed by `ARCHIVE_MANIFEST.json` after evaluator qualification is reproducible. Until then they remain untrusted and must not be used by training.

The only long-lived `docs/` root entries are this file, `a3_t2d5_parameters.json`, and `README.md`.
