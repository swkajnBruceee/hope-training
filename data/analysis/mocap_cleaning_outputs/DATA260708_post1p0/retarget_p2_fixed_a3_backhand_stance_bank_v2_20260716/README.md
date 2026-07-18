# Backhand Stance Probe v2

This is a diagnostic branch, not a training pool.

## Result

Four additional fixed-base-qualified backhands were tested with the same
constant-XY stance optimization and A3 soft-limit contract.

```text
selected: 4
IK pass: 3
trajectory optimization pass: 0
trajectory optimization reject: 3
```

Two candidates reached the `waist_yaw` boundary and one failed the dynamic
jerk gate. No sample is promoted and no threshold is relaxed to manufacture a
backhand stance bank.

The active backhand source remains the separately evaluated fixed-base pool.
