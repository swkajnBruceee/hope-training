# P4B 确定性 motion prior 修复

## 当前结论

motion 0 已产生第一个低维退限位候选，并通过 MuJoCo 几何/限位审计和重建软件包一致性审计。它仍是 `training_approved=false` 的评测候选，PPO 没有开始。

修复表示为 10 次 Bernstein 基函数，39 帧轨迹使用 11 组时间系数。确定性步骤为：

1. 将 `waist_roll_joint` 和 `waist_pitch_joint` 拉回带正裕量的软限位区间；
2. 用 `waist_yaw + right arm` 的阻尼 IK 保持 wrist-offset policy TCP；
3. 将逐帧 IK 形变投影到低维平滑基；
4. 重建 joint velocity、body FK、body velocity 和 upper momentum；
5. 经过版本化 scene placement 生成正式 P1 评测包。

## 离线结果

| 指标 | 修复前 | 修复后 |
| --- | ---: | ---: |
| 最小软限位裕量 | -0.04538 rad | +0.02165 rad |
| 命中 TCP 位置漂移 | — | 1.052 mm |
| 命中法向漂移 | — | 0.0722 deg |
| 命中速度漂移 | — | 0.0487 m/s |
| 最小显式碰撞净空 | 110.13 mm | 109.63 mm |
| basis 后二次 clip | — | 0 rad |

离线修复门槛全部通过。详细证据在：

* `eval_outputs/strike_goal_p4/p4b_repair_candidates/motion_00/repair_audit.json`
* `eval_outputs/strike_goal_p4/p4b_package_consistency_audit.json`

重建后软件包的最大 body FK 位置误差为 `5.66e-8 m`，最大姿态误差为 `4.68e-6 deg`，关节轨迹与退限位候选逐值一致。

## 新发现的速度合同

旧 NPZ 的 `joint_vel/body_*_vel` 不能通过对 50 Hz `joint_pos/body_pos` 直接差分来覆盖。修复器因此使用：

```text
repaired velocity = source velocity + deformation-induced velocity delta
```

这保证零形变时严格恢复旧速度场，不会在重建过程中静默把挥拍速度缩放。旧 velocity 与 50 Hz 位置采样的真实时间语义仍需要在后续速度合同审计中独立闭环。

## P4C 四层状态

候选 manifest 已显式保留：

1. canonical Planner ball-center 目标分量；
2. 修复前 canonical motion label；
3. legacy calibrated control-anchor offset；
4. adapted reference hit state；
5. actual execution hit state（动态回放时填写）。

`scripts/play.py` 的 external-target 和 single-shot 报告现在会把这些层连同 runtime target/actual state 一起记录。

离线 manifest 不会把 `hit_frame / fps = 0.6 s` 伪装成 Planner 的实时 `time_to_strike`。该字段保持为 `null`，并明确要求在命令接收时由控制时钟填充；旧 READY prelude 不属于 motion frame 时间。

## Isaac runtime 核实

先前“当前宿主没有可用 Isaac runtime”的结论是错误的。原因是审计命令误用了 base Python，没有使用项目已配置的 HOPE 环境。本机有效环境为：

```text
Python:    /workspace/anaconda3/envs/hope/bin/python (3.10.20)
Isaac Sim: 4.5.0.0
Isaac Lab: 2.1.0
PyTorch:   2.5.1 + CUDA 12.4
GPU:       NVIDIA GeForce RTX 4060 Laptop GPU
```

`setup_train_env.local.sh` 已将 `HOPE_ISAAC_PYTHON` 指向该解释器。正式运行方式为：

```bash
source setup_train_env.sh
hope_isaac_py scripts/play.py ...
```

使用该 runtime 已成功完成正式 P1 场景静态 FK 审计和 PhysX 全轨迹回放，runtime 不再是阻塞项。

## 正式 P1 审计结果

静态 FK 审计中，修复后 motion 0 的命中帧位置误差为 `2.40e-7 m`，法向误差为 `0 deg`，速度误差为 `0.0232 m/s`；全轨迹最小软/硬限位裕量分别为 `+0.02325 rad` 和 `+0.05816 rad`。证据：

* `eval_outputs/strike_goal_p4/p4b_candidate_p1_static_fk_audit.json`

动态回放使用现有 checkpoint 直接跟踪修复后先验，共运行 323 个控制步，无物理终止，并能在尾段恢复 ready/stable。但旧策略不能直接高精度执行新先验：

| 动态指标 | 旧 motion 0 | 修复后 motion 0 + 旧策略 |
| --- | ---: | ---: |
| 命中位置误差 | 0.04844 m | 0.09580 m |
| 命中法向误差 | 3.33 deg | 9.25 deg |
| 命中速度误差 | 1.000 m/s | 1.199 m/s |
| reference 最小软限位裕量 | -0.04538 rad | +0.02165 rad |
| actual 最小软限位裕量 | -0.04858 rad | -0.05698 rad |
| actual waist-roll 最小软限位裕量 | -0.03495 rad | -0.00137 rad |
| actual 最小硬限位裕量 | -0.000071 rad | +0.00490 rad |

动态证据：

* `eval_outputs/strike_goal_p4/p4b_motion0_p1_policy_nominal.json`

这说明确定性修复已把 **reference** 拉回安全区，但现有策略仍在重放旧的动力学执行模式，且出现了右肩软限位越界。因此 motion 0 仍为 `training_approved=false` 的 P4B candidate。P4C 进一步证明 reference-only 下仍然存在实际动态越限和 14.95 cm 命中误差，所以下一步应先闭环直接参考的相位、速度与动态制动，然后再做监督式轨迹适配/跟踪能力继承。

仍未闭环的门槛为：

* 对正式资产 self-collision 进行独立的全轨迹距离审计；
* 在 actual execution 上恢复正软限位裕量和命中精度；
* 随机初始状态和动力学扰动下的鲁棒性重测。

这些门槛通过之前，不会将该 candidate 用于 PPO。
