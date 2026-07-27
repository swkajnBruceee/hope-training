# Contract Notes

## Historical Contract

The archived `model_3396` chain used a floating base, root position near
`(0, 0, 1.04)`, and asset-default root quaternion `(1, 0, 0, 0)`. Its direct
lineage changed motion manifests from forehand to K8 and then K17. The old
checkpoint and its observation normalizer therefore encode that contract.

## Current Contract

The retraining package uses the corrected strike work point `(3.15, -0.35,
1.04)`, root quaternion `(0, 0, 0, 1)`, and the validated flexed ready pose:

```text
hip_pitch: -0.160 rad
knee:       0.320 rad
ankle_pitch: -0.155 rad
left_hip_roll: 0.080 rad
right_hip_roll: -0.080 rad
```

The root yaw, target-relative observations, manifest semantic label, reset
state, and zero-velocity tail are one joint contract. Changing only the root
quaternion is not an equivalent experiment.

## Ownership

```text
upper model_900: waist + right arm strike action, frozen in F1
Stage-A/F1 actor: 14-D public Base contract, only 12 leg channels effective
root: floating and passively stabilized, not target-driven
feet: may adjust support in place, must not step toward the target
```

## Non-negotiable Audit Fields

Every stage report must include:

```text
manifest hash and motion IDs
checkpoint parent and checkpoint hash
env.yaml and agent.yaml
root position/orientation at reset
ready-pose joint values
observation dimension and normalizer state
actual joint target and actual joint position
root drift, foot displacement and foot slip
exact-strike racket position/velocity/normal errors
motion wrap/resample count
```
