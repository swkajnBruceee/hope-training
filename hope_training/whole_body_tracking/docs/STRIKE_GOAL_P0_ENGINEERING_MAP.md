# P0 工程映射：目标条件挥拍泛化改造

生成日期：2026-08-03。范围：当前可复现的 P9 单拍外部目标执行路径和 P11
恢复训练路径。本文只记录已从 Hydra 解析配置、导入/调用链和源码确认的事实；没有
以文件名或注释推断未运行的功能。

## 结论

当前 P9 不是一个以 Planner 10 维目标为条件的全身挥拍策略。它是一个六 motion
库上的单拍执行器：选择最近的已准入锚点，保留该锚点的拍速、法向和击球时间，并将
外部**位置**的厘米级偏移通过按 motion 标定的前馈矩阵和 coordinator 小残差施加。

因而 P9 是应保留的安全/精度基线，但不能作为目标条件泛化策略的验收对象。P11 只
训练击球后恢复的下肢残差，也不能解决挥拍目标泛化。

## 已确认的实际执行链

| 功能 | 当前实际实现 | 已确认事实 |
| --- | --- | --- |
| 训练入口 | `scripts/train.py` | Hydra 组合 `cfg/train.yaml`、`cfg/task/*`、`cfg/algo/*`，再调用 `parse_env_cfg()` 和 `_apply_task_overrides()`。 |
| 评测/外部执行入口 | `scripts/play.py` | 加载同一 gym 配置和 checkpoint；处理 `external_target_position_b`、motion 选择、审计和完整物理仿真。 |
| P9 gym 环境 | `HOPE-FloatingTargetConditionedRecoveryMotion5Calibrated-AgibotA3-v0` | 注册到 `A3FloatingTargetConditionedRecoveryMotion5CalibratedEnvCfg`。 |
| P9 环境配置 | `training/tasks/tracking/config/agibot_a3/native_strike_env_cfg.py` | P9 继承 P3/P4 的 floating target-conditioned coordinator 和 recovery 支路。 |
| motion reference | `eval_outputs/v27_bent_ready/motion_package_b/manifest.json` | P9 使用 6 个 motion，且执行器仅准入 0、2、3、4、5。 |
| 外部目标接收 | `scripts/play.py` | 接收 `external_target_position_b`，在命令接收时冻结的 base-yaw-heading 坐标系解释。 |
| 当前路由 | `MotionCommand.select_nearest_strike_motion_ids()` + `play.py` | 最近锚点选择、最大距离和每 motion 局部范围均 fail-closed；不是连续 goal policy。 |
| 球拍命令 | `training/tasks/tracking/mdp/hope_commands.py:RacketTargetCommand` | 维护目标和实际球拍状态，并以 reference motion strike frame 产生 manifest target。 |
| 球拍参考点/FK | `training/tasks/table_tennis/mdp/racket.py`、`hope_commands.py` | 优先使用 body `pingpang_red_Link`；不存在时使用 `right_wrist_yaw_Link` 加固定 mount offset。当前运动库 reference 和实时状态共用该 FK 逻辑。 |
| 上肢先验 | frozen `model_900` | 10 action、56 actor observation 的固定上肢/腰部击球先验。 |
| 下肢先验 | frozen `model_3396` | 14 action、126 actor observation 的 Stage-A 历史下肢先验。 |
| 训练 coordinator | P3/P4 `TargetConditionedRecoveryActorCritic` | 22 action：12 腿 + 3 腰 + 7 右臂 correction；P4/P9 的新增可训练分支是恢复残差。 |
| 目标位置局部修正 | `A3FloatingTargetConditionedActionsCfg` 和 action manager | 使用按 motion 标定的 target feedforward，raw clip 0.15；不是由轨迹适配器生成的整段目标轨迹。 |
| 终止 | native env + P9 配置 | P9 启用 `recovery_tilt_max_deg: 30`，另有高度、碰撞等环境终止。 |
| 真机部署 | `model_deployment/` 与 Agibot 示例 | 当前交付是两个子策略和部署 wrapper 合同；没有 10D Planner 输入、没有真机资格认证。 |

## 当前目标合同：真实情况与目标合同的差异

现有 Planner 的真实 ROS 输出是 `hope_ws/src/msgs/msg/RacketCommand.msg`。其可传给
policy 的数值槽位正好是 10 维：`position(3) + normal(3) + velocity(3) +
time_to_strike(1)`；实际 `solver_node.cpp` 以 `header.frame_id = "world"` 发布它们。
`world` 是 HOPE canonical world（球台近侧左角桌面为原点；X 向对手、Y 为 P1 视角左、Z
向上）。消息还保留 `strike_time`、来/出球速度、有效性、过网状态和 bounce 数，不能丢弃。

这里的 **position 物理语义已由源码和 ROS 实测确认为预测球心，不能称它为球拍 TCP**：当前
`HitPlanSolver::solve()` 直接执行 `plan.p_hit = strike.p_ball`，而 trajectory 的
`StrikeTarget.p_ball` 明确定义为“预测击球时的球位置”。随后 `solver_node.cpp` 又直接发布
`out.position = plan.p_hit`。即当前 Planner 输出链中没有球半径、拍面厚度或
link→TCP 刚体变换。头文件中“desired racket center”的注释与执行代码相矛盾。
同时，`velocity` 是理想碰撞模型算出的球拍冲击速度，没有绑定到任何已命名的
机器人刚体点，也没有提供不同点速度换算需要的角速度。因此 P1 的真实原始合同已版本化为
`strike_goal_10d/ball_center_impact_v1`：球心位置、理想拍面法向、理想冲击速度与剩余时间。
它不能未经 contact-point 标定就被重命名为 `racket_reference_point`。

当前 P10 的公开入口仅为：

```text
external_target_position_b = [px, py, pz]
```

虽然底层 `RacketTargetCommand` 中存在目标位置、速度、法向和 `time_to_strike`，但 P9
执行的外部请求仅改变位置。`TARGET_STRIKE_EXECUTOR.md` 明确规定：速度、法向和击球
时间来自选中的 manifest anchor，且 native strike time 固定为 1.56 s。

因此，计划中的 `StrikeGoal10D` 必须从真实 `RacketCommand` 适配而来：其初始 frame
应为 `world`，再通过唯一共享 transform 转成 policy frame。此前候选接口将默认 frame
写成 base-heading 是未对齐的假设，已纠正；绝不能再让 P10 的 position-only latch
代替 Planner 的完整消息。

## Observation 与 action 合同

| 网络/支路 | actor observation | action | 当前目标可见性 | 训练状态 |
| --- | ---: | ---: | --- | --- |
| model_900 upper prior | 56 | 10 | P9 下保留 anchor target，外部位置不能改变其原始合同 | frozen |
| model_3396 Stage-A prior | 126 | 14 | 只接收其历史 support 合同 | frozen |
| P3 coordinator | 204 | 22 | 包含 `joint_coordinator_target_conditioned_observation`；另有 `coordinator_upper` 私有组可看到外部位置 | P3 可训练，但 P9 checkpoint 是已训练局部系统 |
| P4/P9 recovery adapter | 213 (= 204 + 9) | 仅 14 个恢复可用通道 | 后缀是捕获点/稳定/恢复信息；不会重训主挥拍 actor | 可训练，当前 P11 仅用作恢复试验 |
| fixed-base P0 adapter（历史支线） | 25 | 7 右臂 residual | 仅局部 delta、误差、速度、时间、phase、motion one-hot、前一 residual | 局部适配器 |

P3/P9 的专用目标路径必须特别注意：

1. `racket_target_pos_b()` 在 target-conditioned anchor mode 下返回 anchor 位置，以保护
   frozen model_900 输入；
2. `coordinator_racket_target_pos_b()` 才会向 coordinator 私有上肢组提供外部位置；
3. P9 的 position、velocity、normal、time 不是统一的 10D runtime contract；
4. P11 的 `TargetConditionedRecoveryActorCritic` 冻结原 204D actor，只允许门控恢复
   residual 更新。因此它不能学习新的全身击球轨迹。

## 当前奖励：为什么不产生全局泛化

P3 训练配置明确将绝对位置主奖励置零：`racket_position_weight: 0.0`、
`racket_position_y_weight: 0.0`、`racket_position_fine_weight: 0.0`、
`racket_hit_coupled_weight: 0.0`。主要信号是 paired local incremental Huber/gain/cross-axis
损失，目标偏移范围是 `[0.01, 0.01, 0.01]` m。

这证明当前训练目标是局部系统辨识和厘米级补偿，而不是绝对 10D 命中。P4 增强的是
post-hit 支撑/恢复奖励；它没有解除上肢 anchor 约束。

## 已确认的坐标与参考点事实

* Planner `RacketCommand`：HOPE canonical `world` frame。
* P10 外部位置：命令接收时冻结的 base yaw-heading frame。
* 常规 `racket_*_b` 观测：通过当前 base yaw transform 表达；这与“接收时冻结”的
  external delta 不是同一时间语义，P1 必须禁止二者混用。
* 实时与 motion reference strike state 均调用同一球拍 body/FK 路径，这是可复用基础。
* 球拍 normal 当前由姿态矩阵的 axis=1、sign=+1 得到。10D 合同需要显式冻结这个
  normal-axis 约定与 racktet reference-point version。

## 未确认项（P1 前必须通过运行时探针确认）

1. P9/P11 完整 observation 的字段切片、normalizer mean/std 和每个 group 的精确
   runtime shape；静态 204/213 合同已确认，但需要一次 CUDA rollout 记录张量。
2. `pingpang_red_Link` 的几何原点是否就是物理球拍接触参考点；静态 URDF 只能确认它是从
   `right_hand_pingpang_Link` 固连的 mesh root，mesh 是约 160.4 mm 的薄圆拍面，不能证明
   原点就是面中心或哪一侧接触面。
3. Planner raw position（当前代码实际为球心预测）到最终 policy TCP 的显式转换；不能用
   网络吸收球半径/拍面厚度偏差。
4. `strike_time` 属于动捕球消息 header 时钟，`time_to_strike` 在 trajectory 节点按
   `strike_time - last_t_` 计算；它与 Isaac/控制时钟的映射、接收延迟和逐 tick 倒计时仍未
   验证。
5. 部署 wrapper 的 on-robot FK、base heading 与 Isaac frame 的数值一致性；现有部署包
   没有 10D 输入和真机资格认证。

## P1 的不可变基线

在任何实现前，冻结下列资产及其审计：

* P9 `model_79.pt`、Hydra resolved config、五个安全 motion 中心和 ±1 cm 单拍审计；
* P11 `model_79.pt` 及其 70 case full-tail 审计（作为失败/回归证据，不作为候选）；
* model_900、model_3396 的 hash、observation/action 元数据；
* 各基线的球拍位置/速度/法向、base/foot、joint margin、termination trace。

新路径必须提供 `strike_goal.enabled=false` 配置，使其外部行为回退到 P9 的 anchor
执行合同；此回归是 P1/P2 的发布门槛。

## 推荐的首个实现切片（P1，不训练）

只实现并测试以下不可学习的接口，不接入 PPO：

```text
StrikeGoal10D (raw, frame, version, source, receipt timestamp)
  -> StrikeGoalValidator
  -> StrikeGoalFrameTransform
  -> StrikeGoalNormalizer
  -> immutable trace observation
```

该切片必须做到：10D 中位置/法向/速度使用同一 frame（但不偷换为同一物理点）；
normal 归一化；每周期
`time_to_hit` 更新；synthetic 与 Planner 输入可生成逐字节等价的环境命令；关闭开关时
不改变现有 P9 action。完成这些测试后，才可以让 10D 输入真正进入新的 trajectory
adapter。
