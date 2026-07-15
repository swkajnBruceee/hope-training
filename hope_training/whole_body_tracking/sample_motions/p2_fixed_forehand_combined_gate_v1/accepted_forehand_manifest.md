# Accepted Forehand Manifest - combined_gate_v1

- source: `/home/bruce/桌面/HOPETableTennis/hope_training/whole_body_tracking/sample_motions/p2_fixed_forehand_combined_gate_v1/native_zero_residual_manifest.json`
- accepted motions: `4`
- gate: hit task + robot posture / arm margin + wrist naturalness
- status: visual accepted on 2026-07-14; current forehand accepted training candidate source

## Accepted

- `T002_015_gao01_15p25_17p25_dyp10cm`
- `T002_015_gao01_15p25_17p25_dyp15cm`
- `T002_015_gao01_15p25_17p25_dyp20cm`
- `T03_012_gao01_12p10_14p10_dyp20cm`

## Gate Result

```text
hit_composite_pass_rate     = 4 / 4
robot_posture_pass_rate     = 4 / 4
wrist_naturalness_pass_rate = 4 / 4
whole_cycle_pass_rate       = 4 / 4
visual_review               = 4 / 4 accepted
```

Per-motion exact-hit wrist / forearm metrics:

```text
T03_012...dyp20: wrist roll 15.52 deg, pitch 11.93 deg, yaw 13.01 deg, bend 17.65 deg, forearm-racket 5.11 deg
T002...dyp10:    wrist roll 33.66 deg, pitch 11.09 deg, yaw 12.19 deg, bend 16.48 deg, forearm-racket 5.12 deg
T002...dyp15:    wrist roll 32.89 deg, pitch 11.31 deg, yaw 12.41 deg, bend 16.79 deg, forearm-racket 5.25 deg
T002...dyp20:    wrist roll 32.16 deg, pitch 11.77 deg, yaw 12.42 deg, bend 17.11 deg, forearm-racket 5.14 deg
```

Replay command used for visual spot-check:

```bash
cd /home/bruce/桌面/HOPETableTennis/hope_training/whole_body_tracking
source setup_train_env.sh
TMPDIR=/home/bruce/tmp_isaac hope_isaac_py scripts/replay_npz.py \
  --robot agibot_a3 \
  --motion_file /home/bruce/桌面/HOPETableTennis/data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/retarget_p2_fixed_a3_forehand_combined_gate_v1/optimized_motion_npz/T002_015_gao01_15p25_17p25_dyp10cm.npz \
  --steps 1200
```

Visual review result:

```text
2026-07-14: all 4 accepted forehands replayed. User reported these are
particularly good and accepted them visually.
```
