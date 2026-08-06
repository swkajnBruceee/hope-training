# A3 Base Recovery-A Envelope Manual Review V1

## Decision

**Recommendation: `revise`.** Envelope B may enter manual-approval review as `candidate_not_approved`; this report does not approve it.

The corrected state machine supports B_core_only as a candidate for subsequent human approval review, while preserving all auxiliary velocity channels as quality evidence. Automatic approval remains prohibited; upper has 9 safety terminations. Missing contact/load/torque evidence blocks only promotion of auxiliary metrics into hard recovery criteria.

No gate, training environment, reward, disturbance setting, or source trajectory was changed. No PPO or stochastic audit was run.

## Corrected state-machine semantics

The analyzer uses `RECOVERING -> RECOVERED -> OUTSIDE`. An exit cycle starts only on `RECOVERED -> OUTSIDE`. Threshold crossings while already OUTSIDE do not create more cycles. Another cycle is possible only after a complete enter-envelope dwell confirms recovery.

**The old 1563/894/541 values are deprecated and invalid as exit-cycle counts.** They counted threshold runs while an episode was already outside.

`recovery_time_s` is dwell completion, not dwell start. `longest_exit_duration_s` is the OUTSIDE-state duration until confirmed recovery or episode end; threshold-violation duration is separate.

## A/B/C corrected outcomes

- **A_current_full**: clean-tail inside=0.9841, clean threshold runs=183.
  - candidate: transient=99.22%, durable=91.02%, final1s=71.09%, exit cycles=425; core_body_reinstability=187, ankle_velocity_only=6, multi_metric_exit=232.
  - medium: transient=99.61%, durable=78.52%, final1s=63.67%, exit cycles=387; core_body_reinstability=157, ankle_velocity_only=3, multi_metric_exit=227.
  - upper: transient=90.62%, durable=54.69%, final1s=33.20%, exit cycles=314; core_body_reinstability=104, waist_velocity_only=4, ankle_velocity_only=6, multi_metric_exit=199, numerical_or_contact_spike_contact_unverified=1.
- **B_core_only**: clean-tail inside=0.9965, clean threshold runs=68.
  - candidate: transient=100.00%, durable=94.92%, final1s=75.78%, exit cycles=448; core_body_reinstability=218, multi_metric_exit=230.
  - medium: transient=100.00%, durable=80.08%, final1s=66.02%, exit cycles=419; core_body_reinstability=190, multi_metric_exit=229.
  - upper: transient=94.92%, durable=61.33%, final1s=36.72%, exit cycles=367; core_body_reinstability=157, multi_metric_exit=210.
- **C_core_plus_aux_200ms_rms**: clean-tail inside=0.9822, clean threshold runs=15.
  - candidate: transient=100.00%, durable=94.92%, final1s=71.88%, exit cycles=418; core_body_reinstability=192, ankle_velocity_only=6, multi_metric_exit=218, numerical_or_contact_spike_contact_unverified=2.
  - medium: transient=100.00%, durable=80.08%, final1s=65.23%, exit cycles=385; core_body_reinstability=159, ankle_velocity_only=4, multi_metric_exit=222.
  - upper: transient=93.36%, durable=57.42%, final1s=34.77%, exit cycles=335; core_body_reinstability=113, waist_velocity_only=10, ankle_velocity_only=3, multi_metric_exit=209.

Exit-cycle counts are not expected to decrease monotonically from A to B to C: a less restrictive envelope can confirm recovery earlier and more often, creating more legitimate opportunities for a later RECOVERED-to-OUTSIDE transition. Durable recovery and final-1s stability must therefore be reviewed alongside the cycle count.

## Recommended candidate structure

**Envelope B (`B_core_only`) is the recommended hard-recovery candidate, status `candidate_not_approved`.** Its seven core metrics are the only hard recovery criteria. All raw waist/ankle velocities and all 200 ms RMS waist/ankle velocities are quality metrics; they do not veto hard recovery.

Envelope C remains a `candidate_not_approved` comparison. Dwell=0.30 s and hysteresis=1.25 are research-candidate settings only.

B 0.30 s/1.25 corrected research result: candidate transient=100.00%, durable=91.02%, final1s=75.78%, exit cycles=263; medium transient=100.00%, durable=74.22%, final1s=66.02%, exit cycles=230; upper transient=93.75%, durable=56.25%, final1s=36.72%, exit cycles=221.

B materially-sensitive=False under the documented review heuristic; difficulty ordering by baseline p90 recovery time is candidate < medium < upper.

## Per-exit classification

Classification is per exit event. No-exit episodes are `no_post_recovery_exit`; episodes with different event classes are `multiple_exit_categories`. Only auxiliary-only events with at most two actual exit-threshold violation steps may be `numerical_or_contact_spike_contact_unverified`.

## Right-ankle decision

**Option E:** temporarily remove raw and RMS ankle velocity from hard recovery decisions while retaining both as quality evidence. Missing contact/torque evidence blocks promotion of auxiliary metrics into a hard gate; it does not block later human approval of the core-only candidate.

## Evidence and integrity

Strict integrity validation passed=True: four finite 500×256 trajectories, unique profile-matched trace indices, verified hashes and termination sums, and reproduced source envelope.

**Optional evidence unavailable:** torso; ankle target/actual/torque; foot load/contact. No unavailable evidence is fabricated.

The JSON retains the requested episode manifest. Upper per-environment termination reasons are inferred/unknown. Replay tooling remains unavailable.

## Approval state

All approval fields are false and `gate_mutated=false`. No candidate is automatically approved.
