# V1.3B Workspace Expansion

Workspace Expansion is a pure-V1.3B continuation stage:

```text
best completed pure-V1.3B actor
        + fresh critic/optimizer
        -> audited strike-anchor target distribution
        -> wider spatial target-conditioned control
```

It is not another CompletePriors teacher transfer.

## Immutable public contract

- Actor observation remains exactly 98-D.
- Critic remains the existing privileged contract (99-D where configured).
- Action remains exactly 26-D, including the existing microstep channels.
- Goal order remains `[position_xyz, velocity_xyz, normal_xyz, signed_time_to_hit]`.
- All goal quantities use the existing racket reference-point and pelvis/base-yaw local frame.
- READY is the right-front, wide/deep stance; reset perturbation is disabled.
- One 10-second episode contains one goal and one authoritative strike event.
- Existing action scales, joint order, microstep semantics, rewards and post-hit recovery are unchanged.

## Prior and reference kill contract

The WorkspaceExpansion environment is registered separately from
`CompletePriors` and uses the pure reference-free environment class. It does
not instantiate a motion command, reference action, P5U migration, model_900,
or model_3396. Startup is fail-closed unless all of these are false and both
prior alphas are zero:

```text
p5u_migration_loaded=false
model18900_loaded=false
model900_loaded=false
model3396_loaded=false
upper_prior_alpha=0
lower_prior_alpha=0
reference_action_enabled=false
```

The source checkpoint is an explicit `pure_v13b_actor` checkpoint. Actor and
actor observation normalization are loaded; critic, critic normalization,
optimizer moments and learning-rate schedule are fresh.

## Anchor-only metadata path

`training/utils/workspace_anchor_bank.py` loads only each manifest row's
`canonical_goal_10d` fields:

```text
position_m
linear_velocity_mps
normal_w
time_to_hit_s
```

It never loads an NPZ, joint trajectory, q/qdot, phase, body preview or
teacher action. `source_motion_id` is retained only as private sampler/debug
metadata and is never appended to the actor observation. The runtime sampler
always constructs:

```text
selected audited anchor + local position perturbation
```

The first expansion stage uses 100% anchor targets and 0% global-box targets.
The workspace box is only a fail-closed boundary; it is not claimed to be a
uniformly feasible workspace. A future global mixture requires a separately
audited feasible target bank.

## Curriculum

Velocity, normal and timing are already at their final difficulty from the
first update:

```text
velocity: ±20%
normal:   ±12 degrees
timing:   ±100 ms
```

Only spatial coverage changes. Position perturbation ramps from ±1 cm to ±8
cm by 70% of the independent workspace-update clock and remains at ±8 cm for
the final 30% plateau. Anchor coverage uses a smooth active-set schedule:

```text
progress 0%    support_distance <= 0.00 m (W0)
progress 10%   support_distance <= 0.05 m
progress 25%   support_distance <= 0.10 m
progress 40%   support_distance <= 0.20 m
progress 60%   support_distance <= 0.35 m
progress 100%  all qualified anchors
```

Eligibility uses distance outside the old learned support, not nearest-row
count. With nominal `p0=[0.42,-0.18,0.18]` and half-range
`h=[0.08,0.08,0.08]`,
`d_support=||max(abs(p-p0)-h,0)||_2`. W0 is exactly `d_support <= 1e-4 m`;
the audit must report its count/fraction before admission. No late fallback to
`[0.42, -0.18, 0.18]` is permitted. An
out-of-bound sample is rejected/resampled; `workspace_nominal_fallback_count`
must remain zero.

The manifest is also a hard admission input: its top-level
`physics_qualified` and `training_admission` flags must both be true before
the train entry point accepts a WorkspaceExpansion run. A metadata-only audit
of a pending candidate manifest is not a substitute for this gate.

## Required admission sequence

Before any formal WorkspaceExpansion long run:

1. Parse YAML, compile Python and run `git diff --check`.
2. Print anchor-bank xyz/distance statistics for all 23,118 manifest rows and
   inspect density imbalance.
3. Run the anchor sampler at progress 0, 0.10, 0.25, 0.40, 0.60 and 1.0;
   verify eligible counts and sampled xyz ranges expand monotonically.
4. Run the prior kill-test with model_18900/model_900/model_3396/reference
   action files unavailable.
5. Run 128 environments for 20 and 100 PPO updates. Both runs must report:
   `source_checkpoint_kind=pure_v13b_actor`, all prior flags false/zero,
   `workspace_global_fraction=0`, and
   `workspace_nominal_fallback_count=0`.

No 16,384/30,000 formal run is authorized by this document. The first stage
must pass both preflights before a long-run command is considered.
