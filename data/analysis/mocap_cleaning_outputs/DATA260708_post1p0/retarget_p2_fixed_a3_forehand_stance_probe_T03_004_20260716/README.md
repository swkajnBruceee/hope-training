# Fixed-Base Stance Offset Probe: T03_004 (2026-07-16)

This is a diagnostic probe only. It is not a training or held-out promotion
manifest.

## Experiments

- Input: `T03_004_gao01_3p08_5p08`.
- Baseline: fixed base `[3.15, -0.35, 0.3084]`.
- The first probe changed only the fixed base to `[2.9466, -0.6887, 0.3084]`,
  using the median relative hit point of accepted forehands.
- The second probe enabled a constant `stance_offset_xy` optimization variable
  and used Isaac's `soft_joint_pos_limit_factor=0.9` during IK/trajectory
  optimization. The world hit target remained unchanged.

## Result

Naive root-shift probe:

```text
native hit:          1/1
wrist naturalness:   1/1
robot posture:       0/1
whole cycle:         0/1
minimum arm margin:  -0.0429
```

Joint stance optimization with soft limits:

```text
stance offset XY:    [-0.5100, -0.3848] m
native hit:          1/1
wrist naturalness:   1/1
robot posture:       1/1
whole cycle:         1/1
minimum arm margin:  +0.1000
```

The base shift alone did not release the right shoulder limit. The successful
variant optimized a constant base XY variable jointly with the upper-body
trajectory and used the same 0.9 soft-limit factor as the Isaac A3 asset. It
preserved the world hit event and passed the native-calibrated gate. This is
still a single-motion diagnostic; it is not yet a walking controller or a
training-manifest promotion.
