# P4C 动作链责任审计

## 结论

motion 0 的修复参考已经在运动学层保持正软限位裕量，动作处理后的上肢 command 也全程保持正软限位裕量；但 PhysX 中的实际关节仍然越限。责任分类为：

```text
C_DYNAMIC_OVERSHOOT_OR_TRACKING
```

因此不能把越限归因于修复参考或 command clip，也不能直接开始 adapter 监督继承。必须先使直接参考驱动在动力学上通过。

## 审计链

`record_trace=true` 现在逐控制步记录全部上肢 10 关节的：

```text
safe reference
+ frozen upper actor contribution
+ coordinator upper contribution
+ legacy target-adapter contribution
+ safety override
→ processed position/velocity command
→ actual position/velocity
→ soft/hard limit margin
```

`p4c_upper_execution_mode=reference_only` 是只用于评测的可逆开关。它保留冻结的下肢稳定器，但将 frozen upper actor、coordinator upper correction 和 legacy target adapter 的实际上肢贡献归零。默认 `policy` 路径不变。

## 三组配对结果

所有组使用同一 checkpoint、seed、正式 P1 场景根位姿、50 Hz 控制频率和 323 步审计窗口，均无物理终止。

| 模式 | 位置误差 | 法向误差 | 速度误差 | reference 最小软裕量 | command 最小软裕量 | actual 最小软裕量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧参考 + 旧策略 | 0.04844 m | 3.33 deg | 1.000 m/s | -0.04538 rad | 0 rad | -0.04651 rad |
| 修复参考 + 旧策略 | 0.09580 m | 9.25 deg | 1.199 m/s | +0.02165 rad | +0.02885 rad | -0.05698 rad |
| 修复参考 + reference-only | 0.14950 m | 9.44 deg | 1.320 m/s | +0.02165 rad | +0.02885 rad | -0.04779 rad |

`reference_only` 并未改善击球，反而将位置误差从 9.58 cm 增大到 14.95 cm。这证明旧的 actor/adapter 修正对旧动力学跟踪有实际帮助，9.58 cm 不是由它们单独造成的。但这些 legacy 修正不能作为新 adapter 的教师。

## 任务空间误差拆分

修复参考的 runtime target 到 adapted reference 位置误差只有 `6.01e-8 m`，法向和速度标签误差为零。因此：

```text
target → repaired reference: 通过
repaired reference → actual: 9.58 cm (policy) / 14.95 cm (reference-only)
```

击球误差是动态执行误差，不是参考 FK 生成误差。

## 越限细分

reference-only 下的实际越限关节为：

| 关节 | command 最小软裕量 | actual 最小软裕量 | 越限采样数/646 | 主要阶段 |
| --- | ---: | ---: | ---: | --- |
| `waist_roll_joint` | +0.02885 rad | -0.00132 rad | 6 | 击球前动态超调 |
| `waist_pitch_joint` | +0.03025 rad | -0.04047 rad | 100 | READY→swing 连接/击球前滞后 |
| `right_shoulder_roll_joint` | +0.07200 rad | -0.04779 rad | 24 | 击球后 minimum-jerk 回程振荡 |

右肩最差点发生在 control step 141、tail step 55：

```text
safe reference:     -0.120044 rad
processed command:  -0.120044 rad
command velocity:   +0.006495 rad/s
actual position:    -0.000206 rad
actual velocity:    -0.003567 rad/s
actual soft margin: -0.047790 rad
```

该时刻 actor/coordinator/target-adapter 上肢贡献全部为零，因此可排除旧 actor 作为右肩越限的必要原因。

## 门槛状态

P4C 的动作链责任审计已完成，但直接参考驱动的动力学资格门槛未通过。`PPO` 和 adapter 监督训练仍未启动。

下一步必须优先处理：

1. waist-pitch 的 READY→swing 相位/速度合同；
2. waist-roll 的预测制动裕量；
3. right-shoulder-roll 的回程时长、阻尼和动态软限位制动；
4. 在 reference-only 下恢复正 actual 软限位裕量，再开始低维 adapter 教师数据生成。

机器可读总结：

* `eval_outputs/strike_goal_p4/p4c_action_chain_responsibility_summary.json`

三份原始动态报告：

* `eval_outputs/strike_goal_p4/p4c_pair_old_reference_policy.json`
* `eval_outputs/strike_goal_p4/p4c_pair_repaired_reference_policy.json`
* `eval_outputs/strike_goal_p4/p4c_pair_repaired_reference_only.json`
