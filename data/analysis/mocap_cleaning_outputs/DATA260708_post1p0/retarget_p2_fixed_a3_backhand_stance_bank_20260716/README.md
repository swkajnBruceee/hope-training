# Backhand Stance Probe v1

This is a diagnostic branch, not a training pool.

## Result

Four previously fixed-base-qualified backhands were reprocessed with the
forehand stance contract: A3 soft joint limits and one constant XY stance
offset optimized jointly with the 10DOF waist/right-arm trajectory.

```text
selected: 4
IK pass: 3
trajectory optimization pass: 1
trajectory optimization reject: 2
```

The rejects were `waist_yaw_fail` and `fixed_base_dynamic_fail`. They are kept
as diagnostics and are not promoted by relaxing the yaw or jerk gates.

## Interpretation

The existing fixed-base backhand pool remains the active backhand source. This
probe does not show that backhand requires walking or prepositioning. The
prepositioned contract remains available for a future target that truly lies
outside the fixed-base comfort region.
