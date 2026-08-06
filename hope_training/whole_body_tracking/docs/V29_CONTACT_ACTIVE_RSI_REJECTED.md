# V29 contact-active RSI: rejected

V29 direct snapshot restore is retained only as regression evidence. It is not
training input and must not be used to populate an RSI bank.

Evidence from motion 0 / SETTLED on the GPU:

- Direct restore: Gate A passed; Gate B/C rejected.
- Gate B top-20 was dominated by PhysX-derived racket kinematics; direct task
  state fields had no substantive mismatch.
- Reconstruction prefixes of 10, 20, and 40 control steps all failed
  Gate A-R/B-R/C-R. The errors were large and non-monotonic.
- No prefix is bank-eligible.

The final fresh-reset causal control runs both branches to the original target
step 328:

  A: fresh reset -> normal model_100 actor
  B: fresh reset -> A's recorded actuator/controller state

After aligning the observation-noise RNG stream and frozen-parent observation
sampling order, A and B match in explicit state, observation, and the complete
step trace. This confirms that golden target replay is not the defect; the
defect is the non-serializable contact-active PhysX state at a restored anchor.

The replacement training contract is natural_prefix_recovery: true: fresh
reset trajectories run normally, the frozen parents remain active, the learned
V29 adapter is zero before the selected recovery window, and only masked
recovery transitions enter PPO storage.

## V29-P0 implementation and first replay

The natural-prefix path was runtime-verified on CUDA with 256 environments and
a 128-step rollout.  Masked pre-recovery transitions were excluded from PPO
storage, while the adapter remained zero until the recovery mask latched.

A 100-iteration single-shot P0 run completed 3,276,800 environment steps and
produced `model_99.pt`.  This is a training-path smoke result, not a safety
qualification.  Replay of that checkpoint gave:

- single shot: physical termination at control step 165;
- two shots `[0, 0]`: `recovery_tilt` at control step 659 during shot 2;
- five shots `[0, 0, 0, 0, 0]`: the same `recovery_tilt` at step 659, before
  shot 3.

Therefore the natural-prefix implementation is operational, but the first
100-iteration checkpoint is not yet recovery-safe or multi-shot eligible.
The next training decision is to continue P0/diagnostics from fresh-reset
trajectories; no snapshot or reconstruction-prefix sample may enter the bank.

A second fresh-reset run was stopped after its persisted `model_100.pt` was
available for an apples-to-apples check.  It still terminated at step 165 with
`recovery_tilt`; two-shot replay terminated at step 665 during shot 2.  The
failure state retained both-feet contact but had a negative front capture
margin, so this result does not isolate the arm posture as the cause.
