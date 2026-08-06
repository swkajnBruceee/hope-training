# TTMD6 Retarget Contract v0

Status: pilot source package prepared; A3 retarget not admitted.

## Allowed Source Fields

The adapter may consume:

- 14 Cartesian human points per frame;
- paddle center-of-gravity position per frame;
- 120 Hz sampling token;
- filename class/group/sample provenance;
- declared source length as provenance only.

## Explicitly Missing Fields

The local TTMD6 files do not provide:

- paddle orientation, face normal, or tangent;
- paddle handle direction;
- ball position or ball velocity;
- impact/contact frame;
- A3 joint angles, root pose, or A3 actuator state.

## Current Hypotheses

These are diagnostic hypotheses and must remain labeled as such:

```text
source point order: published 14-point order
source vertical axis: third coordinate
source scale: 0.001 m per raw unit
local frame: hips origin, shoulder lateral, head-up
```

The source-normalized pilot preserves raw values and does not overwrite them.

## Adapter Rules

1. Never use `source_length` as `hit_frame`.
2. Never use paddle speed peak as a final `hit_event` without an independent
   timing rule and validation.
3. Never invent a single paddle normal from the paddle center trajectory and
   call it ground truth.
4. Any constructed orientation must be marked `constructed`, carry its
   construction rule, and be evaluated as a task command rather than a human
   orientation label.
5. A3 retarget output must use a separate namespace:

   ```text
   data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v0/a3/
   ```

6. A record can receive `training_eligible=true` only after A3 IK, joint
   limits, wrist/forearm naturalness, actuator dynamics, continuity, posture,
   balance proxy, and visual replay gates all pass.

## Next Pilot Experiment

The first bounded candidate package is now generated at:

```text
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v0/orientation_candidates_manifest.json
```

It contains four separated orientation hypotheses per clip:

```text
velocity_plane_pos / velocity_plane_neg
upright_plane_pos / upright_plane_neg
```

All four are explicitly `constructed_heuristic`, not source ground truth. The
peak-speed frame remains a diagnostic candidate only. The next experiment is
to map each candidate separately into A3 task coordinates, run A3 IK and
replay gates, and compare task success, joint margins, wrist/forearm geometry,
and continuity before selecting any orientation branch.

The current position-candidate package contains 240 files: 30 clips x 2
lateral-sign hypotheses x 4 orientation hypotheses. The lateral mapping is
also intentionally branched because the source-local frame is not the A3
frame. These files are A3-base command candidates only; they are not A3 joint
trajectories and remain ineligible for training.

## A3 Probe Diagnostic Result

The bounded A3 probe now has an independent TTMD6 summary:

```text
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v0/a3_ik_probe_v0/a3_ik_probe_summary.md
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v0/a3_ik_probe_v0/a3_ik_probe_summary.json
```

Current counts are:

```text
48 candidate targets
36 initial IK passes
12 initial IK rejects
36 optimized reports available
0 IK passes not optimized yet
18 optimized replay-ready diagnostic candidates
```

These are not training counts. The 18 replay-ready values mean only that the
fixed-base CSV passed the current finite/schema/base/lower-body/dynamics and
task-geometry prechecks. They do not certify TTMD6 units/axes, constructed
paddle orientation, candidate impact timing, visual replay, balance, or real
A3 execution.

The old optimizer's `bad_source_data` label is not used as a TTMD6 rejection
reason because TTMD6 intentionally lacks the older A3 source-quality flags.
Geometry, wrist, waist-yaw, dynamics, and replay fields are reported
separately. Every record remains `training_eligible=false`.

## Diagnostic Replay Videos

All 18 optimized diagnostic candidates have been rendered as standalone Isaac
reference replays:

```text
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v0/replay_video_v0/video_manifest.json
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v0/replay_video_v0/videos/
```

Each video is encoded at 60 fps and 960x720, with 30 hold frames before and
after the source motion. The resulting duration is approximately 3.9--4.3 s,
so the complete swing is visible rather than only the impact window.

These files are visual review artifacts for the optimized fixed-base A3
reference trajectories. They are not RL policy rollouts, do not validate the
official A3 controller or real hardware, and do not change
`training_eligible=false` for any TTMD6 candidate.

The first complete manual visual review is recorded in:

```text
data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v0/replay_video_v0/visual_review_v0.json
```

The review accepted 8/18 videos and rejected 10/18. All nine filenames ending
in `_pos.mp4` were rejected, along with
`class3_sample101__source_right_to_a3_plus_y__upright_plane_neg.mp4`. The
remaining eight videos were marked visually acceptable. Rejected files remain
diagnostic evidence and are not deleted or admitted to training.

## Locked TTMD6 Lateral Mapping

For future TTMD6 processing, the lateral mapping is fixed to
`source_right_to_a3_minus_y`. The explicit coordinate contract is:

```text
data/analysis/mocap_cleaning/ttmd6_a3_coordinate_contract_v1.yaml
```

The candidate exporter now defaults to this mapping. The historical
`source_right_to_a3_plus_y` branch remains available only through an explicit
diagnostic option and is not part of the production candidate path.

The manually selected production paddle-orientation branch is now
`velocity_plane_neg`. The other three orientation constructions remain
diagnostic-only and are excluded from future default exports.
