# V22 Wide/Deep Stability Contract

## Objective

V22 prioritizes completing the full 6.5 s episode without falling. Strike
accuracy is a regression guard during this phase and may temporarily exceed
10 cm.

## Fixed Priors

- Frozen legacy Stage-A and fixed-base upper policies remain unchanged.
- The V21 `model_0.pt` legacy actor is copied, but a new 2-D support adapter
  starts at exact zero.
- Critic, optimizer, critic normalizer, iteration count, and exploration
  standard deviation are reset.
- Only the new 15-D leg/waist support adapter is trainable; the 7-D arm
  correction remains frozen.

## Stance

- Left foot forward: 4 cm.
- Right foot backward: 4 cm.
- Each foot moves laterally outward by 4 cm.
- Nominal knee flexion: 0.42 rad.
- Hip/ankle pitch and roll are solved together to preserve foot orientation
  and ground height.

This stance won the deterministic scan by the worst-case criterion:
the earliest failure was control step 144, versus 132-137 for the deeper or
wider alternatives. With the V21 `model_500` support adapter, all six motions
reached control steps 193-205 before `recovery_tilt`, but full traces showed
that all six failures were backward falls around -31 degrees. Therefore
`model_500` is retained only as a visual/diagnostic comparison and is not a
training warm start.

## Two-Dimensional Support Strategy

The old V21 branch explicitly represented only sagittal capture margins. V22
uses a 23-D support contract containing:

- both feet XY positions in the pelvis-yaw frame;
- COM relative to the support center;
- sagittal and lateral capture points;
- front, rear, positive-lateral, and negative-lateral capture margins;
- sagittal and lateral stance spans;
- per-foot normalized load and contact;
- root planar velocity and roll/pitch rate.

Both sagittal and lateral capture margins receive dense penalties. This keeps
the wide stance from being treated as a cosmetic reset offset and gives the
adapter direct state for front/back and left/right load transfer.

## Training Priorities

1. Natural timeout and no non-foot contact.
2. Keep both feet loaded and preserve fore-aft/lateral support spans.
3. Reduce root tilt, horizontal velocity, and angular velocity through
   recovery.
4. Preserve strike behavior where compatible with safety.

## Checkpoints

Run deterministic six-motion, full-cycle audits at iterations:

`0, 25, 50, 100, 200, 300, 500`, then every 100 iterations.

The primary checkpoint ordering is:

1. Safety pass count.
2. Earliest failure step.
3. Median and mean failure step.
4. Root displacement and tilt.
5. Strike position, velocity, and normal errors.

Do not select a checkpoint only because its mean termination step improves.
A motion falling backward or a single outlier surviving longer is not a
stability pass.

## Promotion Gate

V22 may return to precision-focused joint training only after:

- `6/6` motions end by natural timeout;
- no base-height, recovery-tilt, or non-foot-contact termination occurs;
- no motion trades a forward fall for a backward fall;
- both feet remain meaningfully loaded through recovery.

Run:

```bash
bash scripts/run_v22_wide_deep_stability_256x1500.sh
```
