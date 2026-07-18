# Retained-Target Actuator Pilot

- episode: `T002_001_gao01_7p52_9p52`
- source manifest: `/home/bruce/桌面/HOPETableTennis/hope_training/whole_body_tracking/sample_motions/p2_stance_train_k8_v1_20260716/manifest.json`
- active target: original ball-contact/retarget target recovered from provenance
- status: not for PPO or data-pool promotion

Run zero-residual evaluation under an explicit actuator profile first. If it fails, compensate the command trajectory while retaining this target; never relabel the target to the failed replay state.

## 2026-07-18 Status

- `official_pd_stand_approx`, zero residual: rejected.
  - position error: `0.1230 m`
  - velocity error: `0.6703 m/s`
  - normal error: `12.89 deg`
  - robot posture and wrist naturalness: pass
- `ilc_iter01`: rejected. The bounded independent joint-error correction
  worsened the strike state. It is retained as a negative control only.

This pilot is not training data. The next variant must use task-space,
forward-actuator optimization while retaining the original target.
