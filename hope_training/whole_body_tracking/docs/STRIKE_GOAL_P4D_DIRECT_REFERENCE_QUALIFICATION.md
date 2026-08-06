# P4D 直接参考动态资格修复

## 当前结论

P4D 已将五条允许 prior（motion 0/2/3/4/5）的**受控腰部与右臂**实际软限位裕量全部转正，且正式 P1 场景中均无物理终止。安全整形不依赖旧 upper actor、coordinator 或 target adapter；评测使用 `reference_only`。

但 P4D 尚未批准 adapter 教师数据生成，原因是 reference→actual 击球误差仍大且随 motion 明显变化，motion 4/5 的速度误差尤其高；此外，被动左肩在 prelude 中仍轻微越过软限位。PPO 与 adapter 训练继续关闭。

## 可复现性纠正

所有有效消融均锁定以下正式 P1 合同：

```text
manifest_frame_z_offset = 0.0
scene_root_position_w_m = [-0.5, -0.7625, 1.04]
table_z_offset = 0.76
full_table = true
control_dt = 0.02 s
seed = 20260725
```

用该命令复跑的 D0 与 P4C 历史报告逐值一致：位置 `0.1495021 m`、法向 `9.4399 deg`、速度 `1.3198224 m/s`、无终止。早期误用 manifest 默认 `0.76 m` frame offset 和 `1.0684 m` root z 的预实验均作废，没有纳入结论。

## 动态辨识

`tools/analyze_p4d_joint_dynamics.py` 从 P4C trace 得到：

* `waist_pitch_joint` 击球前 tracking RMS `0.07237 rad`；
* `waist_roll_joint` 击球前 tracking RMS `0.03229 rad`；
* `right_shoulder_roll_joint` 回程 tracking RMS `0.04702 rad`，最佳速度响应滞后约 `10` 控制步，即 `0.20 s`。

证据：`eval_outputs/strike_goal_p4/p4d_joint_dynamics_identification.json`。

## 有效安全整形

候选配置：`cfg/task/HOPEA3FloatingTargetConditionedP4DDirectReferenceAudit.yaml`。

它采用：

```text
waist inner margin by motion:
  [0.12, 0.12, 0.12, 0.12, 0.08, 0.12] rad

right_shoulder_roll recovery endpoint offset by motion:
  [-0.08, -0.08, -0.08, -0.08, -0.02, -0.08] rad
```

腰部内限保护覆盖 prelude 和 swing；右肩保持原 50 步 minimum-jerk 回程，只改变最终恢复终点。motion 4 必须使用较小整形：把 motion 0 的 `0.12/-0.08` 全局套用会导致击球后物理终止。

## 五条 prior 结果

| motion | 物理终止 | 位置误差 | 法向误差 | 速度误差 | 受控上肢最小实际软裕量 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0.12721 m | 6.90 deg | 1.183 m/s | +0.02617 rad |
| 2 | 0 | 0.08727 m | 4.99 deg | 1.072 m/s | +0.05950 rad |
| 3 | 0 | 0.16901 m | 10.24 deg | 1.038 m/s | +0.04362 rad |
| 4 | 0 | 0.08629 m | 5.59 deg | 2.330 m/s | +0.01137 rad |
| 5 | 0 | 0.14254 m | 8.88 deg | 1.993 m/s | +0.02772 rad |

因此受控上肢 bank 的安全资格通过，但任务空间动态跟踪资格未通过。最大位置误差 `0.16901 m`，最大速度误差 `2.33027 m/s`。

机器可读汇总：`eval_outputs/strike_goal_p4/p4d_direct_reference_qualification_summary.json`。

## 被拒绝的方案

* 全臂回程从 50 步延长到 100 步：虽然右肩越限消失，但改变配重时序，正式有效命令下不作为候选。
* 第一版预测制动：位置/速度修正过猛；加入修正幅值上限后仍不优于轨迹整形。通用 measured-state guard 保留且默认关闭，本候选的动态 guard 实际输出峰值为零。
* 全局 `0.12/-0.08` 参数：motion 4 在 control step 210 发生 `non_foot_ground_contact`。
* 被动左肩内限 clamp：能将左肩实际裕量转正，但破坏 motion 4 的隐式配重并导致恢复终止，故未纳入候选。

## 仍未闭环

1. reference→actual 的位置、法向和速度误差仍不能作为安全教师标签；
2. motion 4/5 速度跟踪误差仍接近或超过 `2 m/s`；
3. `left_shoulder_roll_joint` 的 READY/prelude 合同与冻结下肢配重耦合，需在全身稳定器适配时联合修复；
4. 尚未做初始状态、控制延迟和动力学扰动资格测试；
5. 还未完成五条 prior 的确定性前馈/相位补偿拟合。

## 下一步

进入 P4D 的动态跟踪补偿子阶段，而不是 PPO：对每条 prior 拟合有限的关节级 phase lead、速度前馈和命中窗口补偿；补偿必须经过当前正裕量参考整形。只有五条 motion 的 reference→actual 误差显著下降、全身软限位与恢复同时通过后，才开始 P4F 低维 adapter 的监督能力继承。
