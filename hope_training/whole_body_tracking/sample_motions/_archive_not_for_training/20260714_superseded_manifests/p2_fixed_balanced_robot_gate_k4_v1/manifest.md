# Balanced Robot-Gate K4 v1

Status: native-calibrated zero-residual manifest passed the robot gate.

Use this file for training:

```text
hope_training/whole_body_tracking/sample_motions/p2_fixed_balanced_robot_gate_k4_v1/native_zero_residual_manifest.json
```

Zero-residual evaluation after native target calibration:

```text
hit_composite_pass_rate = 4/4
legacy posture_pass_rate = 2/4
robot_posture_pass_rate = 4/4
whole_cycle_pass_rate = 4/4
```

The two forehands intentionally use robot-gate semantics rather than the old
human-reference torso gate. They are accepted because their absolute torso
tilt/roll/pitch and non-waist arm margin pass the current robot posture gate.

## Motions

- `forehand` `T002_015_gao01_15p25_17p25_dyp10cm` -> `hope_training/whole_body_tracking/sample_motions/p2_fixed_balanced_robot_gate_k4_v1/forehand/T002_015_gao01_15p25_17p25_dyp10cm.npz`
- `forehand` `T03_012_gao01_12p10_14p10_dyp20cm` -> `hope_training/whole_body_tracking/sample_motions/p2_fixed_balanced_robot_gate_k4_v1/forehand/T03_012_gao01_12p10_14p10_dyp20cm.npz`
- `backhand` `T002_023_gao01_26p64_28p64` -> `hope_training/whole_body_tracking/sample_motions/p2_fixed_balanced_robot_gate_k4_v1/backhand/T002_023_gao01_26p64_28p64.npz`
- `backhand` `T03_030_gao01_0p99_2p99` -> `hope_training/whole_body_tracking/sample_motions/p2_fixed_balanced_robot_gate_k4_v1/backhand/T03_030_gao01_0p99_2p99.npz`
