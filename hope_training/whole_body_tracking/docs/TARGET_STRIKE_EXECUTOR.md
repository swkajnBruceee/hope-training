# 外部目标单拍执行器（P10）

当前仿真入口接收一个绝对球拍目标点，完成一次：

```text
RECEIVE_TARGET -> SELECT_ANCHOR -> COMMIT -> SWING -> EVALUATE -> RESET
```

输入合同为：

```text
external_target_position_b = [x, y, z]  # metres
```

`b` 是命令接收瞬间锁定的机器人 base yaw-heading 坐标系。动捕与球轨迹预测模块应先把预测击球点转换到该坐标系；本阶段不要求它是球心或拍面接触点语义。

## 当前可用边界

P10 自动选择只在下列已标定、完整击球后尾段无物理终止的动作中选择。每个中心周围的已验证范围均为三轴 `±1 cm`，击球时刻固定为原生 `1.56 s`。

| motion | 实测控制中心 `[x, y, z]` m |
| --- | --- |
| 0 | `[0.492765, 0.418899, -0.151599]` |
| 2 | `[0.358620, 0.244731, -0.152456]` |
| 3 | `[0.483471, 0.066297, -0.082999]` |
| 4 | `[0.323881, 0.136188, -0.010517]` |
| 5 | `[0.399065, 0.146525, -0.066793]` |

motion 1 未纳入自动选择：它的原生全尾段恢复仍会终止。外部请求若落在未标定动作、超出 `±1 cm`，或要求非原生击球时刻，会被拒绝而不是隐式执行。

自动选择依据的是稳定全身系统的**实测控制中心**，而不是旧 manifest 的名义锚点；随后冻结原始挥拍速度、拍面法向与恢复制动，只以局部 target adapter 修正位置。

## 运行

在 `whole_body_tracking` 目录执行：

```bash
source setup_train_env.sh
hope_isaac_py scripts/play.py \
  task=HOPEA3FloatingTargetConditionedP9Motion5 \
  algo=ppo_joint_coordinator \
  checkpoint=logs/rsl_rl/agibot_a3_floating_target_conditioned_p4_recovery/2026-07-31_17-50-54_p4_motion3_brace_residual_80it/model_79.pt \
  headless=true \
  auto_select_motion=true \
  external_target_position_b='[0.489471,0.059297,-0.073999]' \
  target_audit_post_hit_steps=250 \
  target_audit_report=eval_outputs/target_response/request.json
```

报告会写出选择的 `motion_id`、候选控制中心、局部偏移、命中位置、位置误差和安全终止信息。上例会选择 motion 3，目标相对其中心为 `( +0.6, -0.7, +0.9 ) cm`。

## 已验证结果

所有下列组合点审计均使用同步 sibling、禁用启动阶段物理域随机化的确定性仿真条件；随机性来自目标点采样，不等同于真实硬件或扰动鲁棒性。

| motion | 随机三轴点数 | 平均误差 | P95 | 最大误差 | 物理终止 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 70 | 1.17 mm | 2.34 mm | 2.66 mm | 0 |
| 2 | 70 | 0.99 mm | 1.35 mm | 1.50 mm | 0 |
| 3 | 140 | 2.66 mm | 4.44 mm | 6.43 mm | 0 |
| 4 | 70 | 2.16 mm | 5.06 mm | 5.58 mm | 0 |
| 5 | 70 | 2.47 mm | 5.06 mm | 7.11 mm | 0 |

端到端五中心测试只提供绝对点、不指定动作，选择结果为 `[0, 2, 3, 4, 5]`，且无物理终止。单目标自动选择 motion 3 的中心误差为 `0.93 mm`；三轴组合 `(+0.6, -0.7, +0.9) cm` 误差为 `2.78 mm`。

## 尚未具备的能力

这仍不是连续对打或真实球闭环接口。当前尚未验证：真实运动球接触、目标速度/拍面法向控制、任意工作空间追点、非 `1.56 s` 击球时刻、motion 1、快速恢复后的下一拍，以及硬件安全性。
