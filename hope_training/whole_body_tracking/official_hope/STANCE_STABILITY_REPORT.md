# STANCE_STABILITY_REPORT

## Existing System Audit

### Ten explicit audit answers

1. **DoF:** the MJCF has 31 actuated joints; head yaw/pitch (indices 3/4) are passive at deployment, hence the active policy view is 29 DoF.
2. **Mapping:** the canonical SDK/action order is the 31-name order in `joint_order_agibot_a3.yaml`; leg joints occupy indices 19–30 and are addressed by name in MuJoCo.
3. **Nominal pose:** the measured MuJoCo keyframe is the baseline used here; root height is 1.068390 m and joint q values are listed below. Isaac `InitialStateCfg` values were separately audited.
4. **PD:** deploy gains come from `models/model_21800/policy/params/deploy.yaml`; PD-only static/disturbance runs use the existing reference runner `official_stand`/PD_STAND profile, while policy-idle uses the deploy gains.
5. **Timing:** MuJoCo physics is 0.001 s, control is 0.020 s / 50 Hz with 20 substeps; Isaac is 0.005 s with decimation 4 / 50 Hz.
6. **Frames:** root is `pelvis_link`; feet are `left_ankle_roll_Link` and `right_ankle_roll_Link`; +x is forward and +y is robot-left.
7. **Contact:** MuJoCo foot-ground contacts and actuator/joint/root sensors are read directly; Isaac uses the existing `contact_forces` sensor configuration.
8. **Policy contract:** observation is 110-D and action is 31-D; decoded command is `default_q + scale * raw_action` with clipping, and this test does not alter that contract.
9. **RL terms:** the existing Isaac task contains motion-tracking, action-rate, joint-limit and undesired-contact terms/terminations; no reward or termination code was changed.
10. **Evaluation:** existing entry points are `scripts/play.py`, `scripts/evaluate.py`, `scripts/mujoco_eval_onnx.py` and the reference ONNX runner; the Phase-D test uses the existing `model_21800` ONNX policy path.

- Simulator: MuJoCo MJCF `/home/bistu/桌面/HOPETableTennis/agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml`; MJCF timestep `0.001` s; control dt `0.02` s; 20 physics substeps/control tick.
- Root body: `pelvis_link`; feet: `left_ankle_roll_Link`, `right_ankle_roll_Link`; +x is forward and +y is left, confirmed from the project command/reference convention.
- Actual MJCF baseline pelvis root height: `1.068390` m; measured ankle-body center width: `0.264713` m.
- MJCF baseline foot centers: left `[-0.04542166351091837, 0.13238814353944744, 0.06745825413483647]`, right `[-0.046405585363877, -0.13232535006031806, 0.06745617144462573]`; x offset left-right is `0.000984` m and y separation is `0.264713` m.
- The model has 31 actuated joints. Head yaw/pitch are present in the 31-D contract but passive in deployment, leaving the 29-DoF active policy view.
- Action=0 in the deploy contract decodes to the published `default_joint_pos`; the PD-only tests below bypass policy inference and command `q_des` directly.
- Isaac path: `robots/agibot_a3.py` → `AGIBOT_A3_CFG` → `TrackingEnvCfg.actions.joint_pos` → `ClampedJointPositionAction`; MuJoCo path: `MujocoDirectBridge`/this test runner → named actuator PD.
- Isaac physics is configured at 0.005 s with decimation 4 (50 Hz control); this MuJoCo model is 0.001 s with 0.02 s control (20 substeps).
- Isaac contact sensor is `contact_forces` over robot bodies; MuJoCo exposes foot-ground contacts, actuator-force sensors, joint position/velocity sensors, pelvis frame/IMU sensors and qvel/root state directly.
- Existing evaluation entrances are `scripts/play.py`, `scripts/evaluate.py`, `scripts/mujoco_eval_onnx.py`, and the reference runner `mujoco_reference/reference/a3_deploy_onnx_ref_pingpong/__main__.py`.
- The MuJoCo path was chosen first because its actual MJCF is loadable in-process and exposes all required disturbance/telemetry primitives; Isaac remains the cross-simulator check path.

### MJCF baseline leg q (radians)

| index | joint | q0 (rad) |
|---:|---|---:|
| 0 | `waist_yaw_joint` | -0.000000152 |
| 1 | `waist_roll_joint` | 0.000163394 |
| 2 | `waist_pitch_joint` | 0.013678700 |
| 3 | `head_yaw_joint` | -0.169416000 |
| 4 | `head_pitch_joint` | 0.000000000 |
| 5 | `left_shoulder_pitch_joint` | 0.295233000 |
| 6 | `left_shoulder_roll_joint` | 0.112098000 |
| 7 | `left_shoulder_yaw_joint` | -0.002811210 |
| 8 | `left_elbow_joint` | 0.807255000 |
| 9 | `left_wrist_roll_joint` | 0.000113008 |
| 10 | `left_wrist_pitch_joint` | 0.007775730 |
| 11 | `left_wrist_yaw_joint` | -0.000782062 |
| 12 | `right_shoulder_pitch_joint` | 0.295264000 |
| 13 | `right_shoulder_roll_joint` | -0.112059000 |
| 14 | `right_shoulder_yaw_joint` | 0.002836970 |
| 15 | `right_elbow_joint` | 0.807309000 |
| 16 | `right_wrist_roll_joint` | -0.000126059 |
| 17 | `right_wrist_pitch_joint` | 0.007985940 |
| 18 | `right_wrist_yaw_joint` | 0.000788832 |
| 19 | `left_hip_pitch_joint` | -0.131545000 |
| 20 | `left_hip_roll_joint` | 0.008016050 |
| 21 | `left_hip_yaw_joint` | -0.035203000 |
| 22 | `left_knee_joint` | 0.251534000 |
| 23 | `left_ankle_pitch_joint` | -0.129267000 |
| 24 | `left_ankle_roll_joint` | -0.009612620 |
| 25 | `right_hip_pitch_joint` | -0.131429000 |
| 26 | `right_hip_roll_joint` | -0.007556530 |
| 27 | `right_hip_yaw_joint` | 0.035127300 |
| 28 | `right_knee_joint` | 0.251558000 |
| 29 | `right_ankle_pitch_joint` | -0.129385000 |
| 30 | `right_ankle_roll_joint` | 0.010159100 |

Isaac `InitialStateCfg` uses the same nominal leg pattern but records `hip_pitch=-0.1311`, `knee=0.2468`, `ankle_pitch=-0.1204`, left/right hip roll `+0.0056/-0.0056`, hip yaw `-0.0348/+0.0348`, and ankle roll `-0.0078/+0.0078`; the MJCF keyframe values above are the measured MuJoCo baseline used for MuJoCo experiments.
- Stance generation uses model-backed numerical leg IK and records residuals/invalid configurations. Fore-aft is relative: lead + offset/2, trail - offset/2.

## Experimental Setup

- Requested test: `combined`; trials: `static=5, push=1, swing=1`; seed base: `0`.
- Controller: plain clipped PD, `tau = Kp(q_des-q)-Kd qdot`, using existing `official_stand` gains; no reward, actor, critic, observation, or planner changes.
- Failure is automatic: root height/orientation or non-foot ground contact. Foot slip is measured only while a foot-ground contact is present.

## Results

- Raw rows: `outputs/stance_stability/combined/stance_results.csv`; aggregate rows: `outputs/stance_stability/combined/stance_summary.csv`.
- This report is generated from the rows present in this run. A result is not promoted to a recommended stance until static, push, and swing measurements are available.

## Generated Figures

- `hip_knee_heatmap.png` (when a static grid is run).
- `push_direction_recovery.png` (when push trials are run).

## Policy Integration Recommendation

No policy nominal-pose, reward, observation, action, network, or checkpoint change is made by this tool.
Phase D zero-strike policy testing is implemented separately in `scripts/stance_policy_idle_test.py`; it preserves the original 110-D observation, 31-D action, and deploy default-q contract. The recorded baseline and left-lead idle runs both fell during the tested window, so the current policy is not promoted as a stable nominal-pose controller.
Phase E nominal-pose comparison remains a report-only recommendation: change only the configuration/nominal pose after static, push, swing, and policy-idle evidence are reviewed.
IsaacLab/IsaacSim cross-check was not runnable in this environment (`ModuleNotFoundError` for both packages); MuJoCo results are therefore the executable evidence and the Isaac path is explicitly marked pending.
