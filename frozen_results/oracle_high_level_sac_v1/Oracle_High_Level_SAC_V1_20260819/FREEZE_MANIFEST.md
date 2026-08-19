# Oracle High-Level SAC V1 — Frozen Baseline

Freeze date: 2026-08-19

## Status

- METHOD_FEASIBILITY_VALIDATED
- INDEPENDENT_TEST_PASSED
- DEPLOYMENT_WORK_PAUSED
- FROZEN_AS_BASELINE
- SUPERSEDED_FOR_DEPLOYMENT_BY: Deployment-Aligned High-Level SAC V2

## Architecture

Privileged physical oracle
→ High-Level SAC V1
→ 4D [dy, vx, vy, vz]
→ frozen HOPE model_21800
→ 31D whole-body action

## Canonical high-level checkpoint

checkpoint_update_000512.pt

SHA256:
4a12d4b68bfcd35715ec0bfc1f4c92302b10b23960cdc24df4e717d7f93a4b5d

SAC updates:
512

## Locked independent Candidate-M final test

Pretrain baseline:
417 / 512 legal
81.4453125%

Oracle High-Level SAC V1 u512:
491 / 512 legal
95.8984375%

Absolute improvement:
+14.453125 percentage points

Simulated instability:
0%

## Scientific boundary

This V1 is frozen as an oracle/feasibility baseline.

It is NOT the final deployment policy.

Its observation contract contains privileged physical-oracle information,
including launch-to-hit-plane physical flight time.

Its original 4D physical action contract was developed for the Stage5
Isaac integration and must NOT be silently remapped to the final native
deployment command envelope.

All future deployment-oriented training belongs to:

Deployment-Aligned High-Level SAC V2
