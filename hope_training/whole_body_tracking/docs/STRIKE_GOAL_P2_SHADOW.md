# P2：StrikeGoal shadow 接线报告

生成日期：2026-08-03。状态：**只读 shadow 已运行通过；没有接入 actor，正式场景迁移前禁止训练。**

## 已实现的数据路径

```text
ROS RacketCommand replay
  -> strike_goal_10d/ball_center_impact_v1
  -> LatchedStrikeGoal (Isaac control time)
  -> explicit HOPE-world-to-tracking-sim transform
  -> frozen base-heading receipt frame
  -> ball centre / effective face / link origin separation
  -> actual face velocity = link velocity + omega x r
  -> immutable JSON trace
```

`training/utils/strike_goal_shadow.py` 不依赖 Isaac 或 policy，且没有 observation、command、
action、reward 或 termination 写接口。`scripts/play.py` 的集成点在动作计算和 `env.step()` 之后
采集实际球拍状态；报告中的每条 sample 均固定包含 `action_effect=false`。

shadow 输入必须显式提供 `HOPE world -> sim world` 的旋转和平移。缺任一项就拒绝启动，不能
把两个都叫 `world` 的 frame 静默视为相同物理场景。

## 接触点与速度合同

当前唯一通过自然碰撞验证的配置是诊断 proxy：

```text
ball radius                              = 20 mm
link origin -> effective face (+normal)  = 3 mm
ball centre -> link origin (+normal)      = 17 mm

p_face = p_ball + 0.020 n
p_link = p_face - 0.003 n
```

Planner velocity 保持为目标拍面碰撞点速度。shadow 不虚构目标 link 速度；实际拍面速度按
`v_face = v_link + omega_link x r_link_to_face` 计算。

## Isaac 运行结果

使用当前 P9/P10 checkpoint、motion 0、真实 ROS solver 采样命令
`p=(0,-0.7625,0.3), time_to_strike=0.5 s` 运行 31 个 shadow sample：

* 倒计时从 `0.5 s` 平滑递减，在 control step 25（`0.5 s`）变为 0；
* policy frame 冻结为 `base_heading_receipt/v1`；
* 无 termination，旧策略仍按 motion 0 执行；
* normalization 保持关闭，因为可行工作空间尺度尚未冻结。

### 三种场景变换假设

| 假设 | 转换后 link 目标（base receipt） | 到 motion 0 中心 | 结论 |
| --- | --- | ---: | --- |
| frame identity + 0.76 m 高度 | `[3.134, 0.413, -0.003]` | 2.645 m | 正式坐标轴一致，但 legacy robot authored 在另一桌端 |
| 完整桌中心 180 度镜像 | `[0.426, 0.413, -0.003]` | 0.163 m | 位置变近，但错误翻转策略局部 normal，拒绝 |
| 平移一个桌长、不旋转向量 | `[0.394, 0.413, -0.003]` | 0.178 m | 当前最佳 authored-scene 假设，但没有部署资格 |

结构化比较在 `eval_outputs/strike_goal_p2/frame_hypothesis_summary.json`。

即便使用最接近的场景平移，示例 Planner 目标仍离 motion 0 中心约 `17.8 cm`，而命令剩余
时间 `0.5 s` 明显短于旧执行器约 `1.56 s` 的原生击球时间。因此当前系统不能直接执行该
Planner 目标；这不是 TCP 的毫米级偏移能解决的问题。

## shadow 不影响 action 的运行证明

用 identity 和 2.74 m scene-translation 两个相差很大的 shadow 目标分别运行相同种子、
相同 checkpoint、相同 motion 的 5 个控制步。最终下发 action trace 的 SHA-256 完全相同：

```text
ba09b1b2da137c57c65f0d4db7f5cc673f60af739b7c26b9eac26a01c0264af5
```

因此 shadow 目标没有进入动作路径。

## 架构结论

后续 10D 训练应以正式 HOPE match scene 的世界坐标作为任务合同。legacy tracking motion 的
world placement 不能继续定义策略工作空间；motion 应先转换成命中帧的 base-local 安全先验，
再由目标条件适配器放置到正式场景。这样既能保留已有动作，又不会把 P2 侧录制位置伪装成
Planner frame transform。

进入 actor 前仍需：

1. 正式 match scene 已把 P1 base 定义为 `(-0.5,-0.7625)`、yaw 0；下一步需验证
   legacy motion 转成 base-local prior 后在该放置下的击球点与可达性；
2. 用相同 shadow 命令确认目标落入合理的 base-local 工作空间；
3. 冻结 synthetic/planner 共用的 normalization 范围；
4. 真机部署前完成 hardware TCP 与 source/control clock 标定。
