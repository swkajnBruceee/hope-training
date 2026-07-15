# Sample Motions

Generated `.npz` files in this directory are ignored by git.

Create the smoke clip with:

```bash
hope_isaac_py scripts/create_smoke_motion.py \
  --headless \
  --frames 120 \
  --output sample_motions/agibot_a3_smoke_stand.npz
```

The smoke clip is only for pipeline verification. Replace it with a retargeted
ping-pong motion before running meaningful training.

Current retargeted ping-pong training data is documented in:

```text
README_DATASETS.md
```

Do not use an old manifest by directory name alone. The native-strike task now
requires an explicit `motion_manifest=...` argument because several historical
sets pass older numeric gates but fail the current visual / wrist / robot-quality
gate.

Current status:

```text
Balanced K8 training candidate:
  p2_fixed_balanced_k8_current_v1/manifest.json

Status:
  4 forehands visually accepted
  4 backhands numeric-gate accepted and visually accepted
  K8 zero-action gate passed 8 / 8 whole-cycle

Use this as the current small native-strike residual-PPO training candidate.
```

Historical manifests have been moved out of the top-level training search area:

```text
_archive_not_for_training/20260714_superseded_manifests/
```
