# p2_fixed_balanced_k8_torso_control_v1

Current 4FH+4BH candidate after fixing backhand torso pitch/roll references.

Training candidates:
- 4 forehands inherited from the current accepted forehand set.
- 3 primary backhand torso-control samples.
- 1 supplement backhand: `T002_023_gao01_26p64_28p64`.

Held-out/candidate backhands are stored in `heldout_backhand_candidates.json`.

Status: candidate. It still needs visual replay review and a clean physical zero-residual check before PPO training promotion.
