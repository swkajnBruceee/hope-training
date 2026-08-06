# TTMD6 Dataset Audit

Status: audited; 30 pilot clips manually reviewed and approved as source
clips, but none are admitted to the A3 training pipeline yet.

## Source

Local dataset:

```text
/home/bruce/桌面/HOPETableTennis/TTMD6
```

The local copy contains:

```text
TTMD_cut_bat/   12,000 CSV files
TTMD_cut_hum/   12,000 CSV files
```

Every bat file has a matching human file with the same suffix. The local
description says that the data contains six table-tennis skills and 9,000
motions, while the local file inventory is 12,000 paired clips. The published
paper describes 30 athletes and 9,000 strokes, whereas the local filename
tokens include values 1 through 40. This discrepancy is recorded as unresolved;
the local files must not be assigned participant or class semantics by guess.

## Observed file contract

Representative files:

```text
bat_120_4873_2_17_200_178.csv
human_120_4873_2_17_200_178.csv
```

Observed facts from the local files:

| Field | Observed value |
| --- | --- |
| sampling token | `120` for every file |
| stored rows | `200` for every file |
| bat columns | `3` Cartesian values per frame |
| human columns | `42` Cartesian values per frame = 14 points x 3 |
| padding | trailing all-zero rows are present |
| source length | final filename token; may be greater than the stored 200 rows |
| file pairing | bat/human suffixes match exactly |
| numeric values | approximately 0 to 1,600 in sampled files; unit not declared locally |

The final filename token is interpreted as `source_length`: the motion length
before the fixed 200-row storage transform. Files with a source length below
200 contain trailing zero padding; files with a source length above 200 contain
200 nonzero rows and have therefore been downsampled or otherwise resampled.
It must not be treated as a current active length, hit frame, or contact frame.

## Published information that is useful

The associated publication describes:

- six skill labels: forehand attack, forehand drive, forehand push, backhand
  attack, backhand drive, and backhand push;
- infrared motion capture at 120 Hz;
- 38 reflective markers on the participants;
- 14 joint trajectories plus the paddle center of gravity in the released
  dataset;
- four motion phases: backswing, stroke, follow-through, and recovery;
- data collected with a ball projection machine and cross-court instructions.

This supports using TTMD6 as a human motion prior and a source of diverse
paddle trajectories. It does not provide an A3-compatible joint-angle motion
contract by itself.

## Differences from the current A3 pipeline

The current A3 manifest/NPZ pipeline expects robot-specific data such as:

```text
31 A3 joint positions and velocities
body/root poses in the A3/Isaac convention
joint and body name mappings
racket position, velocity, normal, and hit-event metadata
retarget/posture/limit gates
```

TTMD6 provides none of these in the local CSV contract. In particular:

1. The 42 human values are Cartesian marker/joint positions, not A3 joint
   angles.
2. The 14-point order is not declared in the local files.
3. The coordinate axis convention and units are not declared in the local
   files.
4. No racket orientation, racket normal, or racket tangent is provided; only
   paddle center-of-gravity position is present.
5. No explicit impact/contact frame is provided. The filename length token is
   not a contact label.
6. The six skill categories are not yet safely mapped to local numeric token
   values `1..6`.
7. The local copy's file count and participant-like token range disagree with
   the published dataset description.

## Evidence-based semantic checks

The following conclusions are stronger than filename-only guesses, but remain
separate from authoritative source metadata:

### 14-point order

The published skeleton figure lists the points as:

```text
1 hips
2 head
3 left shoulder       4 left arm          5 left forearm
6 right shoulder      7 right arm         8 right forearm
9 left upper leg      10 left leg          11 left foot
12 right upper leg    13 right leg        14 right foot
```

Using this order on the local CSVs produces a connected skeleton with stable
human bone lengths across sampled files. This is sufficient to use the
published order as the current parsing hypothesis, but the adapter must retain
the source-order declaration and fail closed if a future source version
changes it.

### Coordinate axes and scale

The head-to-foot displacement is overwhelmingly along the third coordinate,
so the local third coordinate is very likely vertical (`Z`). The observed human
segment lengths are physically plausible when interpreted as millimetre-scale
Cartesian coordinates. This is strong evidence for `Z-up` and millimetre-like
units, not proof of the mocap export convention. The adapter must preserve a
source-unit/axis field and require a calibration check before conversion to
metres.

### Numeric class tokens

Trajectory statistics provide a high-confidence working hypothesis:

```text
1: forehand attack       2: forehand drive       3: forehand push
4: backhand attack       5: backhand drive       6: backhand push
```

Classes 1--3 and 4--6 occupy distinct lateral trajectory families; classes 2
and 5 have substantially greater speed/range than 1 and 4; classes 3 and 6
have smaller vertical/range profiles. This agrees with the six-class order and
the attack/drive/push distinctions described in the publication. The mapping
remains `high-confidence inferred`, not an authoritative label, until original
metadata or manual skeleton/paddle review confirms it.

### Source length and stored rows

The initial sample disproved the simpler interpretation that the final filename
token is the current active length: some files have source lengths such as 208
or 220 but are stored as 200 fully nonzero rows. The defensible interpretation
is `source_length` before a fixed-length transform. The adapter must preserve
both `source_length` and the actual stored nonzero-row count, reconstruct the
time scaling at 120 Hz, and derive impact timing separately.

### Full-scan source anomaly

The full local scan found 100 paired suffixes that cannot be admitted:

```text
numeric class/group tokens = (3, 13) or (4, 31)
bat CSV              = 400 rows x 3 values
human CSV            = 200 rows x 42 values
```

The affected bat files contain two separated nonzero blocks, while the paired
human files contain one padded 200-row clip. This is not a normal fixed-length
conversion case. The provenance of the extra bat block is unresolved, so the
whole affected subset is excluded from pilot retargeting rather than trimmed
by an inferred rule.

The complete machine-readable audit is:

```text
docs/eval_reports/ttmd6_schema_full.json
```

Its result is `structural_fail` only because of this quarantined 100-file
subset; the separate source index reports 11,900 structurally valid records.

## Admission decision

### Can be used

Yes, as a separate high-quality human motion source after an independent
adapter and audit. Its strongest uses are:

- selecting high-quality forehand/backhand motion families;
- learning paddle trajectory and timing distributions;
- enriching the candidate strike-state archive;
- providing human pose priors for retargeting;
- adding drive/push variants later, if those actions match the project goal.

### Cannot be done yet

Do not:

- convert the CSV columns directly into A3 joint positions;
- assume the numeric class token mapping;
- assume the last filename token is the current active length or `hit_frame`;
- reuse current A3 coordinate transforms or gate thresholds without calibration;
- mix TTMD6 clips into the existing manifest before retarget and replay gates;
- call the local values millimetres merely because their magnitude looks like
  millimetres.

## Required TTMD6-specific adapter stages

```text
TTMD6 CSV pair
    -> parse active length from padding and filename consistency
    -> identify 14-point order and coordinate/unit convention
    -> label six skill classes from authoritative metadata
    -> reconstruct/derive paddle orientation and candidate impact timing
    -> normalize to a human-local frame
    -> retarget to A3 with a new TTMD6-specific IK configuration
    -> apply A3 actuator, wrist, waist, posture, balance and continuity gates
    -> generate a separate TTMD6-derived manifest/NPZ family
    -> only then compare against the current A3 source pool
```

The derived artifacts must use a separate namespace, for example:

```text
sample_motions/ttmd6_a3_retarget_v1/
docs/eval_reports/ttmd6_a3_retarget_v1/
```

They must not overwrite or silently join the existing `p2_stance_*` data.

## Next audit actions

1. Create a local schema report for all 24,000 files.
2. Visualize representative clips from every numeric class and several
   active-length ranges.
3. Confirm the 14-point ordering against the published skeleton figure or
   authoritative source metadata.

## Locked v1 Diagnostic Retarget

The first v1 pilot uses the manually selected production branches:

```text
source_right_to_a3_minus_y
velocity_plane_neg
```

The complete v1 diagnostic report is:

```text
docs/eval_reports/ttmd6_retarget_v1_diagnostic_report.md
```

The formal IK gate rejected all 30 locked pilot candidates. Twenty records
were intentionally sent through optimization as a diagnostic-only cohort
because their position and normal were reachable while their tangent gate
failed. Sixteen produced replay-ready fixed-base trajectories. All 16 NPZ
conversions and complete Isaac reference videos passed file/runtime checks.

These 16 records remain `training_eligible=false`: the cohort is 13 inferred
forehand and 3 inferred backhand records, the paddle orientation is
constructed rather than source-provided, and the tangent gate was bypassed
for diagnosis. The next production action is to expand the same locked source
contract and build a balanced train/held-out pool. The formal tangent gate is
not relaxed merely to increase the sample count.
4. Calibrate units and axes using the human body dimensions and table/court
   geometry, not by magnitude alone.
5. Detect candidate impact timing from paddle trajectory and motion phase; do
   not convert the final active frame into a hit event.
6. Retarget only a small balanced pilot set, then evaluate with the current A3
   visual, actuator, posture, waist, and whole-cycle gates.

Until these actions pass, TTMD6 remains an audited archive/source pool rather
than training data.

## Manual pilot review result

The bounded pilot set contains 30 clips, five per inferred numeric class. All
30 were manually reviewed and marked `manual_pass`. This means that no visual
blocking issue was observed in the source-space viewer for this pilot set. It
does **not** certify the inferred skill labels, units, axes, impact timing,
paddle orientation, A3 reachability, actuator feasibility, or whole-cycle
stability.

The review artifact is:

```text
docs/eval_reports/ttmd6_pilot_manual_review.json
```

The source-space profile generated from these 30 clips is:

```text
docs/eval_reports/ttmd6_pilot_source_profile.json
docs/eval_reports/ttmd6_pilot_source_profile.md
```

The profile confirms that the pilot files can be read as paired 120 Hz source
clips and records active-frame counts and paddle-speed diagnostics under the
current scale hypothesis. It intentionally leaves hit timing and paddle
orientation unassigned.

Every record remains:

```text
training_eligible = false
retarget_status    = pending
a3_replay_status   = pending
```

The next gate is a TTMD6-specific retarget pilot. Only records that pass the
source calibration, impact/orientation construction, A3 IK, actuator, posture,
continuity, and replay checks may later receive `training_eligible=true`.

## Source-normalized pilot result

The 30 manually approved clips have now been copied into a separate source
normalization package:

```text
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v0/
```

The manifest is:

```text
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v0/source_normalized_manifest.json
```

The source-normalized validation report is:

```text
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v0/source_normalized_validation.json
```

All 30 clips passed finite-value, shape, orthonormal-frame, and frame-to-frame
coordinate continuity checks. The worst adjacent-frame axis dot product was
`0.99675`; this validates the diagnostic frame construction, not the physical
correctness of the remaining unit/axis hypotheses.

This package preserves raw coordinates and adds a human-local diagnostic frame
defined from hips, head, and the two shoulders. It is not an A3 frame. The
third-axis and `0.001 m/raw-unit` choices remain hypotheses only. Paddle
orientation and hit frame remain unassigned, so no A3 retarget, NPZ training
motion, or training admission has occurred.

The next implementation step is to construct and validate TTMD6-specific
paddle orientation and impact timing on this bounded pilot before invoking the
existing A3 IK tools.

The current orientation-candidate package is:

```text
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v0/orientation_candidates_manifest.json
```

It contains four separated heuristic orientation branches per clip. All 30
clips have a valid candidate window around the diagnostic speed peak, but the
speed peak remains only a candidate event window; no record has received a
`hit_frame` or `training_eligible=true`.

The next bounded output contains 240 explicitly branched A3-base position and
orientation candidates:

```text
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v0/a3_position_candidates_manifest.json
```

This is 30 clips x 2 lateral-sign mappings x 4 constructed orientation
branches. It is still a command-candidate package, not an A3 joint trajectory
package; A3 IK and replay evaluation are the next gate.

## A3 Probe Status

The first bounded A3 probe has an independent TTMD6-specific summary:

```text
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v0/a3_ik_probe_v0/a3_ik_probe_summary.md
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v0/a3_ik_probe_v0/a3_ik_probe_summary.json
```

It contains 48 branched targets. Initial IK produced 36 passes and 12
rejects. All 36 IK-pass candidates now have an optimization report. Eighteen
optimized candidates pass the current fixed-base replay precheck, but remain
diagnostic-only.

The summary deliberately does not reuse the legacy optimizer's
`bad_source_data` category: TTMD6 has no old A3 source-quality flags. It keeps
source provenance, IK reachability, trajectory geometry, wrist/waist comfort,
joint dynamics, replay precheck, and training admission separate.

No TTMD6 clip has been promoted to a training manifest. Constructed paddle
orientation and candidate hit frame are still hypotheses, not source truth.

## Diagnostic Replay Video Package

The 18 optimized candidates have a separate replay-video package for manual
visual inspection:

```text
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v0/replay_video_v0/video_manifest.json
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v0/replay_video_v0/videos/
```

The videos are 60 fps, 960x720, and approximately 3.9--4.3 seconds long. Each
includes a short hold before and after the source trajectory so that a complete
swing can be reviewed. This package is diagnostic-only: it is neither an RL
policy result nor official MOTION/AimSim validation, and no TTMD6 candidate is
admitted to training by generating these videos.

The complete manual visual review is recorded at:

```text
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v0/replay_video_v0/visual_review_v0.json
```

Review result: 8/18 videos accepted visually and 10/18 rejected. Every
`_pos.mp4` branch was rejected, as was
`class3_sample101__source_right_to_a3_plus_y__upright_plane_neg.mp4`. The
rejected videos are retained as diagnostic/archive artifacts and remain
excluded from all training manifests.

For subsequent TTMD6 datasets, the lateral coordinate sign is now locked to
`source_right_to_a3_minus_y`. The versioned contract is:

```text
data/analysis/mocap_cleaning/ttmd6_a3_coordinate_contract_v1.yaml
```

The `plus_y` branch is retained only for historical diagnosis and must not be
silently mixed into future candidate packages.

The selected production paddle-orientation construction is
`velocity_plane_neg`; the other three orientation branches remain historical
diagnostic hypotheses.

## 2026-07-17 Expansion Intake Decision

The next TTMD6 intake is tracked separately at:

```text
data/analysis/mocap_cleaning_outputs/TTMD6_expansion_intake_v1/
```

It contains a balanced 24-record source batch, four records per numeric class,
with the 30-clip pilot excluded. The batch is still `training_eligible: false`.

One formal A3 probe per class was run with the current wrist/waist comfort
configuration. Five of six passed the task-space IK check and all five were
replay-ready after optimization. A parallel run requiring the constructed
paddle tangent to match rejected all six, even though five had sub-2 mm
position error and sub-4 degree paddle-normal error. Because TTMD6 stores paddle
center position but no measured paddle orientation, the constructed tangent is
not ground truth. It is now a diagnostic/ranking metric, not a hard admission
gate. This does not waive the position, normal, joint-limit, continuity,
dynamic, native-PD, and visual gates.

The exact probe reports are:

```text
data/analysis/mocap_cleaning_outputs/TTMD6_expansion_intake_v1/a3_ik_formal_probe_v1/
data/analysis/mocap_cleaning_outputs/TTMD6_expansion_intake_v1/a3_ik_task_probe_v1/
```

Do not promote the five probe outputs yet. They still require NPZ/FK checks,
native zero-residual calibration, and replay review. The K12 admission plan is
documented in:

```text
hope_training/whole_body_tracking/docs/K12_DATA_ADMISSION_PLAN.md
```
