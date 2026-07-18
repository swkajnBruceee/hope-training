# TTMD6 A3 Probe Summary

> Diagnostic only. No record in this report is training eligible.

## Scope

- Candidate targets: `48`
- Initial IK reports: `48`
- Initial IK pass: `36`
- Initial IK reject: `12`
- Optimized reports available: `36`
- IK-pass candidates without optimization: `0`
- IK-rejected candidates not optimized: `12`
- Optimized replay-ready diagnostic candidates: `18`

The legacy optimizer's `bad_source_data` label is not used here. It is a
contract mismatch because TTMD6 intentionally lacks the old A3 source
quality flags. Geometry, wrist, waist, dynamics, and replay fields are
reported independently.

## Status Counts

- `ik_reject`: `12`
- `optimized_reject_hit_geometry`: `14`
- `optimized_reject_waist_yaw`: `4`
- `optimized_replay_ready_diagnostic`: `18`

## By Inferred Skill Class

| class | label | total | IK pass | optimized | replay-ready diagnostic |
|---:|---|---:|---:|---:|---:|
| 1 | forehand_attack | 8 | 5 | 5 | 3 |
| 2 | forehand_drive | 8 | 6 | 6 | 4 |
| 3 | forehand_push | 8 | 7 | 7 | 3 |
| 4 | backhand_attack | 8 | 8 | 8 | 5 |
| 5 | backhand_drive | 8 | 5 | 5 | 2 |
| 6 | backhand_push | 8 | 5 | 5 | 1 |

## Records

| episode | class | IK | optimized | diagnostic status | near hard limit |
|---|---|---|---|---|---|
| `class1_sample1__source_right_to_a3_minus_y__velocity_plane_pos` | `forehand_attack` | `reject` | `no` | `ik_reject` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_elbow_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class1_sample1__source_right_to_a3_minus_y__velocity_plane_neg` | `forehand_attack` | `pass` | `yes` | `optimized_replay_ready_diagnostic` | `waist_roll_joint, waist_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class1_sample1__source_right_to_a3_minus_y__upright_plane_pos` | `forehand_attack` | `pass` | `yes` | `optimized_replay_ready_diagnostic` | `waist_roll_joint, waist_pitch_joint, right_shoulder_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class1_sample1__source_right_to_a3_minus_y__upright_plane_neg` | `forehand_attack` | `pass` | `yes` | `optimized_replay_ready_diagnostic` | `waist_roll_joint, waist_pitch_joint, right_elbow_joint, right_wrist_pitch_joint` |
| `class1_sample1__source_right_to_a3_plus_y__velocity_plane_pos` | `forehand_attack` | `reject` | `no` | `ik_reject` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class1_sample1__source_right_to_a3_plus_y__velocity_plane_neg` | `forehand_attack` | `reject` | `no` | `ik_reject` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class1_sample1__source_right_to_a3_plus_y__upright_plane_pos` | `forehand_attack` | `pass` | `yes` | `optimized_reject_waist_yaw` | `waist_roll_joint, waist_pitch_joint, right_shoulder_roll_joint, right_wrist_yaw_joint` |
| `class1_sample1__source_right_to_a3_plus_y__upright_plane_neg` | `forehand_attack` | `pass` | `yes` | `optimized_reject_waist_yaw` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class2_sample51__source_right_to_a3_minus_y__velocity_plane_pos` | `forehand_drive` | `pass` | `yes` | `optimized_replay_ready_diagnostic` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class2_sample51__source_right_to_a3_minus_y__velocity_plane_neg` | `forehand_drive` | `pass` | `yes` | `optimized_replay_ready_diagnostic` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class2_sample51__source_right_to_a3_minus_y__upright_plane_pos` | `forehand_drive` | `pass` | `yes` | `optimized_replay_ready_diagnostic` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class2_sample51__source_right_to_a3_minus_y__upright_plane_neg` | `forehand_drive` | `pass` | `yes` | `optimized_replay_ready_diagnostic` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_yaw_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class2_sample51__source_right_to_a3_plus_y__velocity_plane_pos` | `forehand_drive` | `reject` | `no` | `ik_reject` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class2_sample51__source_right_to_a3_plus_y__velocity_plane_neg` | `forehand_drive` | `pass` | `yes` | `optimized_reject_waist_yaw` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class2_sample51__source_right_to_a3_plus_y__upright_plane_pos` | `forehand_drive` | `reject` | `no` | `ik_reject` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_yaw_joint` |
| `class2_sample51__source_right_to_a3_plus_y__upright_plane_neg` | `forehand_drive` | `pass` | `yes` | `optimized_reject_waist_yaw` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class3_sample101__source_right_to_a3_minus_y__velocity_plane_pos` | `forehand_push` | `pass` | `yes` | `optimized_reject_hit_geometry` | `waist_roll_joint, waist_pitch_joint, right_shoulder_roll_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class3_sample101__source_right_to_a3_minus_y__velocity_plane_neg` | `forehand_push` | `reject` | `no` | `ik_reject` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class3_sample101__source_right_to_a3_minus_y__upright_plane_pos` | `forehand_push` | `pass` | `yes` | `optimized_replay_ready_diagnostic` | `waist_roll_joint, waist_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class3_sample101__source_right_to_a3_minus_y__upright_plane_neg` | `forehand_push` | `pass` | `yes` | `optimized_reject_hit_geometry` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_elbow_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class3_sample101__source_right_to_a3_plus_y__velocity_plane_pos` | `forehand_push` | `pass` | `yes` | `optimized_reject_hit_geometry` | `waist_roll_joint, waist_pitch_joint, right_shoulder_roll_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_yaw_joint` |
| `class3_sample101__source_right_to_a3_plus_y__velocity_plane_neg` | `forehand_push` | `pass` | `yes` | `optimized_reject_hit_geometry` | `waist_roll_joint, waist_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class3_sample101__source_right_to_a3_plus_y__upright_plane_pos` | `forehand_push` | `pass` | `yes` | `optimized_replay_ready_diagnostic` | `waist_roll_joint, waist_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_yaw_joint` |
| `class3_sample101__source_right_to_a3_plus_y__upright_plane_neg` | `forehand_push` | `pass` | `yes` | `optimized_replay_ready_diagnostic` | `waist_roll_joint, waist_pitch_joint, right_shoulder_roll_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint` |
| `class4_sample151__source_right_to_a3_minus_y__velocity_plane_pos` | `backhand_attack` | `pass` | `yes` | `optimized_replay_ready_diagnostic` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class4_sample151__source_right_to_a3_minus_y__velocity_plane_neg` | `backhand_attack` | `pass` | `yes` | `optimized_replay_ready_diagnostic` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class4_sample151__source_right_to_a3_minus_y__upright_plane_pos` | `backhand_attack` | `pass` | `yes` | `optimized_replay_ready_diagnostic` | `waist_roll_joint, waist_pitch_joint, right_shoulder_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class4_sample151__source_right_to_a3_minus_y__upright_plane_neg` | `backhand_attack` | `pass` | `yes` | `optimized_reject_hit_geometry` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_yaw_joint, right_wrist_yaw_joint` |
| `class4_sample151__source_right_to_a3_plus_y__velocity_plane_pos` | `backhand_attack` | `pass` | `yes` | `optimized_replay_ready_diagnostic` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_elbow_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class4_sample151__source_right_to_a3_plus_y__velocity_plane_neg` | `backhand_attack` | `pass` | `yes` | `optimized_replay_ready_diagnostic` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class4_sample151__source_right_to_a3_plus_y__upright_plane_pos` | `backhand_attack` | `pass` | `yes` | `optimized_reject_hit_geometry` | `waist_roll_joint, waist_pitch_joint, right_shoulder_roll_joint, right_wrist_yaw_joint` |
| `class4_sample151__source_right_to_a3_plus_y__upright_plane_neg` | `backhand_attack` | `pass` | `yes` | `optimized_reject_hit_geometry` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_wrist_yaw_joint` |
| `class5_sample201__source_right_to_a3_minus_y__velocity_plane_pos` | `backhand_drive` | `pass` | `yes` | `optimized_replay_ready_diagnostic` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class5_sample201__source_right_to_a3_minus_y__velocity_plane_neg` | `backhand_drive` | `reject` | `no` | `ik_reject` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class5_sample201__source_right_to_a3_minus_y__upright_plane_pos` | `backhand_drive` | `pass` | `yes` | `optimized_reject_hit_geometry` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class5_sample201__source_right_to_a3_minus_y__upright_plane_neg` | `backhand_drive` | `pass` | `yes` | `optimized_replay_ready_diagnostic` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class5_sample201__source_right_to_a3_plus_y__velocity_plane_pos` | `backhand_drive` | `reject` | `no` | `ik_reject` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class5_sample201__source_right_to_a3_plus_y__velocity_plane_neg` | `backhand_drive` | `reject` | `no` | `ik_reject` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class5_sample201__source_right_to_a3_plus_y__upright_plane_pos` | `backhand_drive` | `pass` | `yes` | `optimized_reject_hit_geometry` | `waist_roll_joint, waist_pitch_joint, right_shoulder_roll_joint, right_wrist_yaw_joint` |
| `class5_sample201__source_right_to_a3_plus_y__upright_plane_neg` | `backhand_drive` | `pass` | `yes` | `optimized_reject_hit_geometry` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_yaw_joint` |
| `class6_sample251__source_right_to_a3_minus_y__velocity_plane_pos` | `backhand_push` | `pass` | `yes` | `optimized_reject_hit_geometry` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class6_sample251__source_right_to_a3_minus_y__velocity_plane_neg` | `backhand_push` | `reject` | `no` | `ik_reject` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class6_sample251__source_right_to_a3_minus_y__upright_plane_pos` | `backhand_push` | `pass` | `yes` | `optimized_reject_hit_geometry` | `waist_roll_joint, waist_pitch_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class6_sample251__source_right_to_a3_minus_y__upright_plane_neg` | `backhand_push` | `pass` | `yes` | `optimized_replay_ready_diagnostic` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_yaw_joint, right_wrist_yaw_joint` |
| `class6_sample251__source_right_to_a3_plus_y__velocity_plane_pos` | `backhand_push` | `reject` | `no` | `ik_reject` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class6_sample251__source_right_to_a3_plus_y__velocity_plane_neg` | `backhand_push` | `reject` | `no` | `ik_reject` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint` |
| `class6_sample251__source_right_to_a3_plus_y__upright_plane_pos` | `backhand_push` | `pass` | `yes` | `optimized_reject_hit_geometry` | `waist_roll_joint, waist_pitch_joint, right_shoulder_roll_joint, right_wrist_yaw_joint` |
| `class6_sample251__source_right_to_a3_plus_y__upright_plane_neg` | `backhand_push` | `pass` | `yes` | `optimized_reject_hit_geometry` | `waist_roll_joint, waist_pitch_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_wrist_yaw_joint` |

## Admission Decision

- `training_eligible=false` for every record.
- `optimized_replay_ready_diagnostic` means only that the current A3 fixed-base replay precheck passed.
- It does not certify TTMD6 units, axes, skill labels, impact timing, constructed paddle orientation, or real A3 execution.
- Before admission, the remaining optimizer candidates must be processed and the diagnostic candidates must pass TTMD6-specific visual, actuator, posture, balance, and impact validation.
