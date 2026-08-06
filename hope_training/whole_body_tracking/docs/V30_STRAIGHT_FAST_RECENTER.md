# V30 straight-READY Fast Recenter

V30 uses V26 as the experiment baseline because it has the smallest variable
set with deterministic two-shot evidence. It does not delete or modify the
V27/V28 bent-READY work.

## Fixed contract

- Frozen parents: Stage-A `model_3396`, upper `model_900`, and V22 coordinator
  `model_1499`.
- V25 sagittal support gain and five-step exit remain unchanged.
- V26 fail-closed re-arm remains unchanged: two feet, capture/velocity/tilt
  limits, ready reference, and stable hold must all pass.
- The straight V26 READY manifold, 50-step prelude, 20-step follow-through
  hold, and 100-step reference return remain unchanged.
- The six-motion 50 Hz strike-only manifest is pinned explicitly; it has a
  frame-30 hit, so the 128-step rollout reaches post-hit collection.

## Learned scope

The V10 recovery adapter is zero initialized and receives the frozen 227-D
V22 state plus capture-point rate and a post-hit gate. The gate is zero before
hit (including the prelude), so PPO cannot improve recovery by weakening the
strike. Natural-prefix collection runs from fresh reset and stores only
post-hit recovery transitions; no PhysX contact-active snapshot is restored.

## Acceptance order

1. Five-shot continuous safety and no accumulated state drift.
2. Single-shot and two-shot hit/safety regression preservation.
3. `hit -> READY` reduction from the V26 baseline of about 4.9 seconds:
   first to 3.5--4.0 seconds, then to 2.5--3.0 seconds.
