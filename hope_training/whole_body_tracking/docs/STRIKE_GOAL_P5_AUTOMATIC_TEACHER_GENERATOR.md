# P5 自动反手教师轨迹生成器

## 决策

P5 不再把 motion-specific control anchor 当作任务目标，也不继续扩大
motion-specific adapter。它保留已有 motion、平衡先验、安全过滤器和
PhysX 回放作为**求解初值与资格基础设施**，并以 P1 canonical 10D strike
goal 作为唯一教师验收目标。

本文件区分三个不能混淆的量：

```text
canonical_goal_10d        Planner / 教师 / 部署任务真值
control_anchor            旧局部 adapter 的输入坐标原点，仅用于兼容诊断
actual_execution_state    PhysX 实测结果
```

任何报告必须同时给出 `actual -> canonical_goal` 与（若适用）
`actual -> control_anchor`。后者不得替代前者，也不得写回
`strike_target_b0`。

## 最小 P5A/P5B 生成合同

第一版仅覆盖一种规范反手风格，并只扩展位置；法向、速度和命中时间先从
被选 seed 的 canonical 标签继承。一个请求的输入为：

```text
initial_robot_state
canonical_goal_10d
style = backhand_canonical_v1
seed_motion_ids
```

初始位置采样必须在由下列门槛定义的可行域内，而不是独立均匀地随机 10D：

1. 多初值命中 IK 可行；
2. 全程正软限位裕量；
3. 无自碰撞和球台碰撞；
4. 命中速度/时间与 seed 的可执行范围相容；
5. 不要求超出已验证根基座稳定包络的姿态。

每个 target 使用多个现有 backhand motion 作为离线优化初值。motion ID
不能出现在最终策略输入或教师标签中。

## 候选轨迹表示

教师候选是一段完整轨迹，而不是命中姿态：

```text
READY -> pre-swing -> hit(p, n, v, t) -> safe follow-through -> recoverable state
```

位置形变使用 Bernstein/B-spline 的低维系数；速度、加速度与 jerk 由同一
时间参数化导出。仅在下列离线门槛均通过时，轨迹才可进入 PhysX 队列：

```text
target -> reference position / normal / velocity / time
reference soft-limit margin
reference collision clearance
velocity / acceleration / jerk limits
style-distance regularizer
```

MuJoCo 只做候选生成与快速筛选，不能授予教师资格。

## Reference 准入标准（固化合同）

Reference 分为两种用途，准入门槛不能混用：

* **P5D tracker reference**：不要求 reference-only PhysX 已经精准命中，
  但必须任务语义正确、几何正确、全程安全、时间连续、动态大致可执行，
  并处于冻结 `model_900 + model_3396 + support-state machine` 的有界残差范围内。
* **最终教师 reference**：必须通过 `reference -> safety -> tracker/command ->
  PhysX actual -> canonical_goal_10d` 全链路、扰动和恢复资格；只有此类样本
  才能作为端到端教师。

每条 reference 必须逐项通过以下七类检查，并把证据写入 manifest；缺证据不能
默认为通过：

1. **目标语义**：绑定完整 `canonical_goal_10d = position + normal + velocity +
   time_to_hit`，统一 TCP、world/base-heading frame 和时间合同；不得写回
   actual 落点、control anchor 或旧 motion-specific offset。
2. **完整轨迹**：包含 `READY -> pre-swing -> acceleration -> hit(p,n,v,t) ->
   follow-through -> recoverable state`；不能用 `READY` 到 `q_hit` 的简单线性
   插值冒充挥拍。
3. **全程运动学安全**：逐帧正软限位裕量、硬限位、全身自碰撞、球台碰撞、
   TCP 路径和基座/左侧冻结关节均须审计。
4. **动态平滑**：同一时间参数化下审计全程 `dq`、`ddq` 和 jerk；命中速度
   修正不得造成命中前尖峰、方向反转或随挥振荡。
5. **先验相容性**：记录 `reference - model_900 prior` 的全周期和命中窗口
   max/RMS，并确认所需 10-D residual 不长期饱和、不超 action scale。
6. **目标分布可行**：位置、法向、速度、时间遵守物理耦合；第一版优先扩展
   position，其他 10-D 分量继承或小范围变化，不独立均匀随机完整 10-D。
7. **邻域连续性**：相邻目标的 hit 姿态、轨迹系数和 TCP 路径变化必须平滑；
   优先使用 continuation，禁止仅凭随机冷启动生成大量孤立动作岛。

首版离线动态阈值固定在
`cfg/p5_reference_dynamics_v1.json`（50 Hz、`|dq|<=6`、`|ddq|<=100`、
`|jerk|<=2500`），恢复段使用 50 个控制帧；任何调整都必须更新合同版本。
命中后 zero-velocity tail 或只有 strike-only 截断的 NPZ 不得进入
`TRACKER_TRAINING_ELIGIBLE`。

以下生成方式一律禁止：旧 motion 随机关节噪声、两个 motion 轨迹直接线性
插值、只由 IK 姿态生成整段动作、为了通过评估重标 canonical target、直接把
PhysX actual 当规范 reference、只检查 hit frame、只审计右臂而不审计全身。

### 机器可读准入状态

每条样本必须标记且只能处于以下状态之一：

```text
OFFLINE_REJECTED
TRACKER_TRAINING_ELIGIBLE
QUALIFIED_TEACHER
```

`OFFLINE_REJECTED` 表示运动学、限位、碰撞、动态平滑或合同检查失败，或必需的
先验/安全层证据缺失（缺证据 fail-closed）；`TRACKER_TRAINING_ELIGIBLE` 表示
七类离线检查以及冻结 `model_900 + model_3396 + state machine` 的有界残差、
reference-only safety replay 均有证据，但尚未证明 PhysX 精准命中；
`QUALIFIED_TEACHER` 还必须通过正式 PhysX、tracker、扰动和恢复。
旧的 `PENDING_PHYSX` 只能作为生成器内部过渡值；只有补齐上述 tracker 准入证据
后才能映射为 `TRACKER_TRAINING_ELIGIBLE`，不能直接发布或与最终教师混淆。

P5D-2 的首次完整重生成固定排除 `motion 4`。当前审计还记录了 `motion 1` 的
完整恢复段软限位失败（最小裕量 `-0.010818 rad`），因此也不会用未验证样本
凑足锚点数量；缺失锚点必须显式报告，禁止静默替换。

## PhysX 资格门槛

正式 P1 场景必须用部署相同的控制频率、执行器、安全过滤器、浮动根基座与
下肢稳定器回放，并记录：

```text
safe reference -> processed command
processed command -> actual state
actual TCP -> canonical_goal_10d
```

候选只在以下条件下进入教师集：

* safety projection 不主导命中误差；
* actual 的限位与碰撞裕量为正；
* canonical 位置、法向、速度、时间满足配置化阈值；
* 无物理终止，且随挥/恢复通过；
* 在小初始状态与动力学扰动下重复通过。

## P5D-2 完整 reference 重生成记录

生成器脚本为 `tools/build_p5d2_complete_references.py` 和
`tools/build_p5d2_complete_runtime_reference_bank.py`。候选先由
`READY -> hit -> 50 帧 quintic follow-through/recovery` 补齐为 81 帧，再逐帧
重算全身 body/FK、速度、加速度、jerk、软限位和碰撞；runtime 包使用冻结的
P1 root anchor `[-0.5, -0.7625, 1.04]` 进行 b0→world 物化。首次重生成结果为
265 个候选，其中 208 个通过完整离线门槛、57 个拒绝；随后选出 71 条 bank：
42 train、9 validation、10 bridge holdout、10 boundary OOD。训练集由 4 条通过
完整恢复审计的 anchor（0/2/3/5）、28 条 local continuation、5 条
endpoint-flat programmatic phase-warp、5 条 multi-seed dynamics variant 组成；
bank 不包含 `motion 4`，并明确记录 `motion 1` 因完整轨迹软限位失败未入锚点集。

在冻结先验和正式 safety filter 的 reference-only replay、邻域连续性审计完成前，
这些 71 条只能保留为“离线静态通过、runtime 证据待补”的候选，不能启动训练。
当前审计报告见
`eval_outputs/strike_goal_p5/p5d2_complete_runtime_reference_audit/`
（71 条静态通过、0 条最终 `TRACKER_TRAINING_ELIGIBLE`、0 条教师）。

首次正式 reference-only PhysX 回放已完成（P5 residual=0，冻结
`model_900 + model_3396 + support-state machine`）。71/71 条到达 hit frame，
`canonical target -> reference` 平均误差 `1.10 mm`；但
`reference -> actual` 平均误差仍为 `31.23 cm`。全周期 safety projection
最大 `0.055444 rad`、平均 `0.024213 rad`，只有 7/71 条低于暂定的
`0.01 rad` 透明阈值。这证明新 reference 的几何合同正确，但旧冻结先验尚不能
无损执行这一批完整轨迹；不是 reference 生成器的几何误差。注意，
`reference -> actual` 误差只作为 P5D tracker 的训练损失诊断，不作为 reference
拒绝条件。按 safety projection 门槛重新分组后，7 条进入“透明候选、待恢复/连续性
审计”，64 条进入“必须将运行时 safety filter 纳入优化的候选”；分组报告见
`p5d2_safety_reference_split_report.json`。回放原始日志和汇总见
`p5d2_physx_reference_only_replay_v2.log` 与
`p5d2_physx_reference_only_replay_summary.json`。

不通过的样本保留失败原因，并由求解器重新优化或丢弃；不得通过重标目标、
写入 actual hit state，或复用旧 control anchor 来伪造成功。

## 教师标签、唯一性与质量等级

监督学习的主标签是通过资格验证的 `teacher_safe_reference` 的低维轨迹
系数，绝不直接模仿 `actual_trajectory`。每个样本至少保存：

```text
teacher_safe_reference      监督主标签
processed_command           安全/确定性执行层输出
actual_trajectory           动态资格与误差模型证据
canonical_goal_10d          任务真值
```

`actual` 中的滞后、超调、偶然根基座漂移和安全投影只能用于执行误差建模，
不能被学生学成计划动作。

每个目标由多个 seed 产生候选，但第一版只保留一个确定性的
`backhand_canonical_v1` 教师，按如下统一排序选择：任务误差、软限位与碰撞
裕量、基座稳定、速度/加速度/jerk、恢复能力和动作风格距离。motion ID 与
seed 不得进入最终策略输入。

参考与执行资格等级为：

* **R（reference）**：canonical `p+n+v+t`、平滑、限位和碰撞均通过离线
  合同；它是目标条件轨迹生成器的监督标签，以及动态跟踪器的参考输入；
* **D（dynamic diagnostic）**：R 级 reference 在正式 PhysX 中 safety
  基本透明但 `processed_command -> actual` 偏差显著；它不是失败/负样本，
  而是 reference-tracking residual controller 的训练与诊断数据；
* **E（executable）**：R 级 reference 由通用 tracker 在正式 PhysX 中实际
  命中 canonical goal，并通过扰动和恢复资格；只有 E 级可作为端到端执行教师。

在 E 级内部使用以下质量等级：

* **A**：严格 PhysX 命中、安全层基本透明、扰动重复通过且可恢复；高权重监督；
* **B**：安全、结构合理但存在中等任务误差；仅低权重形状/覆盖预训练；
* **C**：投影主导、越限/碰撞/恢复失败或动态不可达；不作正向模仿，仅作
  可行性与安全负样本。

离线候选、名义 PhysX A 级和轻扰动 PhysX A 级的数值阈值必须分别版本化并
随数据集发布；不得只在单次实验命令中临时改变。

## 覆盖、留出与失败合同

在生成前（而不是数据生成后）划分 `training`、`validation`、`bridge holdout`、
连续 `workspace holdout` 与 `boundary holdout`。任何 holdout 均不得参与教师
调参、curriculum、early stopping 或 PPO 奖励调节。

生成报告按位置体素记录采样数、IK 通过数、离线通过数、PhysX 通过数和最终
教师数；每个目标还记录尝试 seed、成功 seed、最终 seed 和候选间轨迹距离。
这同时审计工作空间覆盖和 seed dependence，防止自动流程退化成密集模板库。

DAgger 重新求解必须显式返回以下之一：

```text
SOLVED
SOLVED_WITH_RELAXATION
UNREACHABLE
UNSAFE_INITIAL_STATE
TIME_INSUFFICIENT
OPTIMIZER_FAILED
```

失败状态保留为负样本或安全拒绝数据。不得静默切换最近 motion、改变目标或
延长命中时间。

## P5D 通用动态跟踪器与冻结执行先验

P5D 不能从一个没有站立/支撑能力的新 22-D actor 开始。第一版执行路径必须是：

```text
P5 safe reference（canonical goal 不变）
       + frozen model_900（熟悉的腰部/右臂挥拍先验）
       + frozen model_3396（12 个下肢支撑通道）
       + 已审核 support-state machine
       + 新 P5D PPO 的 10-D [waist(3), right arm(7)] 小残差
       -> 同一个 joint safety filter -> PhysX actual
```

`model_900`、`model_3396` 与状态机均为 inference-only；P5D PPO 在第一阶段**绝不
输出腿部动作**，也不允许重用 P4 target adapter、control anchor、motion ID 或
motion-specific feed-forward。纯安全 reference 加 22-D 新 actor 的实验只保留为
“没有先验”的失败归因消融，不得用它判断 R 级轨迹是否稳定或是否可执行。

P5D 训练的是 `safe reference + actual state -> bounded upper-body residual`，不选择
motion、不产生 canonical goal，也不改写 reference。其输入至少包含实际关节
状态、根/足接触状态、当前 `q/dq/ddq` reference、未来 reference preview、
phase/time-to-hit 与 TCP reference/actual 误差；输出为经过既有 safety layer 的
小位置（可选速度前馈）残差。奖励以 `processed command -> actual`、命中窗口的
TCP tracking、residual 平滑/幅度、限位、碰撞、根稳定和恢复为主。

启动课程固定如下，且每轮保留旧样本回放以防遗忘：

1. **P5D-0 执行归因**：在同一 P1 初始条件下报告旧 reference / P5 reference
   各自的 `zero residual` 与 `frozen 900+3396` 结果；每条都分解
   `reference -> command`、`command -> actual`、`actual -> canonical`。
2. **P5D-1 旧分布回归**：冻结 900/3396/状态机，只训练 10-D 小残差复现既有
   安全反手 reference；旧策略可作为行为先验/回归基线，不能作为最终教师标签。
3. **P5D-2 连续扩展**：按目标位置 bin 加入时间缩放、小形变、程序化安全轨迹和
   P5 continuation reference；训练集始终混入旧分布，预先划定的空间 holdout 不得
   进入课程。
4. **P5D-3 扰动资格**：只在名义跟踪已进入捕获范围后加入初始状态、执行器和
   动力学轻扰动；失败按既定 DAgger 状态码记录，不得暗中回退到某个旧 motion。

已有旧执行链的 P1 名义回放在保留模型 900、3396 和状态机时完成 323 个控制步且
无物理终止，但对 canonical 目标仍有约 19.0 cm 的位置误差；这证明的是平衡先验
确实被调用、但其动态击球能力仍是窄域的，并不授予 E 级资格。反之，纯 22-D
新 actor 的 zero-residual 回放约第 80 步因基座高度终止，不能被解释为 P5 reference
本身让机器人跌倒。

随后完成的 P5D prior-guided 零残差复核给出了更直接的归因证据：动作管理器公开
动作维度为 10，`model_900` 原始输出全周期最大绝对值为 0.5、实际腰臂先验贡献
最大 0.16 rad，`model_3396` 的 14 维 Stage-A 输入中 12 个腿支撑通道非零且被
应用；P5 残差严格为 0，安全投影严格为 0。323 个控制步无物理终止，最低根高
1.040 m；motion 3 命中 canonical 目标的位置误差 8.06 cm、法向 6.64°、速度
1.57 m/s。这次结果明确说明“没有复用旧先验所以跌倒”的假设不成立：复用后没有
跌倒，但仅靠旧先验仍不是 E 级教师。原始报告和分层恒等式见
`eval_outputs/strike_goal_p5/p5d_prior_10d_zero_residual_motion3_v2.json` 及其
`p5d_tracker_action_chain/v2` 摘要。

高层 10D-to-trajectory PPO 在 P5D 期间保持关闭。P5D 只有在多条 R 级参考上将
actual 进入配置化捕获范围后，才可参与 E 级资格或 P5 的联合 PhysX 优化。
第一版捕获门槛暂定为约 3--5 cm TCP 位置、5--8 度法向和 0.5--0.8 m/s 速度
误差，同时保持正安全裕量、无物理终止并在小初始状态扰动下维持改善；这不是 E
级批准阈值，而是证明通用跟踪器值得进入联合迭代的门槛。

## 数据与训练阶段

合格样本保存初始状态、canonical 10D goal、完整安全 reference、导数、
actual trajectory、命中状态、所有安全裕量与资格版本。

监督模型先预测低维 Bernstein/B-spline 系数，而非直接记忆 motion ID 或
逐帧动作。随后通过 PhysX DAgger 重新从学生实际到达状态求解教师。只有在
监督模型已经稳定进入捕获范围后，PPO 才能作为小残差控制器启用。

## 当前边界

P4D motion 3 仅保留为 seed 与执行诊断。P5 不批准它成为教师，也不继续为它
增加 control-anchor 偏移、局部 adapter 能力或未经资格验证的 phase/velocity
补偿。

## P5D-2 完整 reference bank 审计（2026-08-04）

当前冻结的完整 bank 为 71 条，motion 4 按用户要求排除；motion 1 因完整轨迹软
裕量未通过而排除。分类为 anchor 4、local 28、programmatic phase-warp 5、
multi-seed dynamics variant 5、validation 9、bridge 10、boundary 10。

71 条均通过离线 canonical `p+n+v+t` 几何检查（平均 target→reference
1.10 mm，均不超过 5 mm）。正式 P1 reference-only 回放捕获 71/71 个命中帧；
该回放中 reference→actual 平均 31.23 cm，只作为 tracker 的学习误差，**不作为
reference 拒绝条件**。安全层最大投影 0.055444 rad、平均 0.024213 rad，按
0.01 rad 透明阈值分为 7 条透明候选和 64 条需要安全感知重优化的候选。

对 7 条透明候选执行 130 控制步的 PhysX 恢复审计：7/7 命中、0 次物理终止、
0 次 timeout；最低根高 0.9038 m，最低根 upright 指标 0.7495。该结果只证明
名义回放没有物理终止，不等于已完成小扰动资格，也不等于最终教师批准。

连续性审计已生成邻近 canonical 目标、命中姿态和完整轨迹的逐样本指标；报告
当前 fail-closed，不自动提升资格。重复 canonical goal 会单独标记，以避免同一
目标的多 seed 解在单模态训练中被误当成独立连续目标。
本 bank 在全 10-D 合同 1e-5 容差下有 26 个 unique canonical goals；71 条记录中
有 67 条的最近邻是同一目标的 seed/变体，因此训练采样时必须按目标去重或显式
处理多解，不能把 71 条直接当成 71 个独立目标。

其余 64 条已冻结为安全感知重优化队列。重优化必须调用部署时同一 safety filter
或精确等价逻辑，保持 canonical goal、TCP/frame/time 和完整恢复合同不变；不得
用 actual 落点重标目标。以上步骤均未启动 PPO 训练。

## P5D-2 tracker formal run result (2026-08-04)

已按冻结合同完成 P5D-2 tracker 训练与成对评估。训练 manifest 使用去重后的 24 条 canonical goal 代表，其中 train 16、validation 4、holdout 4；validation/holdout 未进入 PPO 采样。4096 环境采样 smoke、64 环境多 reference smoke 均通过；4096 环境正式训练完成，使用 A/B 短训中表现更好的 fresh 初始化 A，继续训练 2000 轮（resume 总计 2197/2199），总采样 196,509,696 timesteps。

model_900、model_3396 和 support-state machine 全程冻结；P5D actor 仍为 10-D 腰部+右臂 residual，actor observation 不含 motion/reference ID；canonical goal 未重标。正式 checkpoint：`logs/rsl_rl/agibot_a3_p5d_prior_guided_reference_tracker_p5d2/2026-08-04_01-26-45_p5d2_formal_4096x2000/model_2198.pt`。

最终 checkpoint 的固定 PhysX paired replay（每组同一 manifest、130 控制步）如下；`learned` 是 PPO residual，`zero` 是同 checkpoint 下强制零 P5D residual。两者均未达到教师批准门槛，当前结论为 **TRACKER_TRAINING_RESULT，不是 QUALIFIED_TEACHER**：

| split | learned TCP pos (m) | zero TCP pos (m) | learned TCP vel (m/s) | zero TCP vel (m/s) | learned normal (deg) | zero normal (deg) |
|---|---:|---:|---:|---:|---:|---:|
| train (16) | 0.2071 | 0.3237 | 1.1004 | 0.9318 | 12.08 | 24.36 |
| validation (4) | 0.1235 | 0.2391 | 1.4173 | 1.0356 | 9.26 | 19.98 |
| holdout (4) | 0.1241 | 0.2108 | 1.3557 | 1.4111 | 11.10 | 17.63 |

residual 对位置和法向有明确改善，且 holdout 仍改善位置/法向；速度没有一致改善。训练和评估均未产生物理终止，但 composite teacher pass rate 仍为 0。该 checkpoint 不得写入教师库，也不得 relabel canonical target。

完整日志：`eval_outputs/p5d2_formal_4096x2000.log`、`eval_outputs/p5d2_formal_train_eval.log`、`eval_outputs/p5d2_formal_validation_eval.log`、`eval_outputs/p5d2_formal_holdout_eval.log` 及对应 `*_zero_eval.log`。

## P5D-2 完成审计与当前结论（2026-08-04，v2）

本轮全部证据汇总在 `eval_outputs/p5d2_completion_audit_v2.json`。该文件是当前
状态的唯一汇总，不得把旧的 `runtime_gate_summary_v1` 中关于 standalone
diagnostic wrapper 超时的说明误读为正式任务失败：64-env smoke、4096-env
sampling smoke、4096-env formal PPO 和 paired PhysX replay 均实际运行并完成。

本轮没有重新启动训练。已完成的 formal run 使用 4096 个环境，从 A/B 短训中
表现更好的零初始化 A checkpoint 开始，最终 `model_2198.pt` 共采样
196,509,696 timesteps。真实 P5D-1 `model_999` warm-start 的严格 B 对照反而
更差，因此不能以“加载旧 checkpoint”作为默认假设。

固定 checkpoint 的 train/validation/holdout paired replay 已按同一 manifest 做
learned-vs-zero 对照；同时完成了 300、500、700、900、1100、1300、1500、1700、
1900、2198 的固定 checkpoint 重放。这里必须如实注明：这些 interval replay 是
训练结束后的 checkpoint 审计，不是训练循环内实时 callback；所有 checkpoint 的
composite teacher pass rate 都是 0。

结论不是“训练无效”，而是：P5D residual 已在位置和法向上对 train、validation
以及未参与 PPO 的 holdout 产生一致改善，但速度改善不一致，最差 reference 仍有
较大误差，且没有任何样本达到最终教师资格。因此当前 checkpoint 的等级是
`TRACKER_TRAINING_RESULT_NOT_QUALIFIED_TEACHER`，不得写入教师库、不得 relabel
canonical goal、也不得在用户未要求时自动开启下一轮训练。

冻结先验的实际文件为 `checkpoints/frozen_priors/model_900.pt` 和
`checkpoints/frozen_priors/model_3396.pt`；审计中保存了两者 SHA-256。actor 的
输入不含 motion/reference ID，动作维度保持 10，P5D 只输出腰部+右臂 residual。
`multireference_correctness_4096_v1.json` 还记录了 uniform 4096-env 的 16 个
reference 和四个 workspace region 的真实采样计数；uniform、balanced-by-region、
difficulty-weighted、curriculum 四种采样模式均有 64-env smoke 日志。

## P5D-3A 困难 reference 审计（仅审计，未启动新训练）

基于最终 checkpoint 的逐 reference paired replay，24 条 canonical-goal 代表已先
按现象分类，报告为 `eval_outputs/p5d3a_difficulty_audit_v2.json`（v1 是仅基于
已有标量报告的初步分类）。当前有 10 条
reference 的 learned position 与 reference-tracking 误差都仍在 30 cm 以上；另有
9 条已经进入相对可学习的位置/法向范围，另有 5 条是位置已经接近但速度方向或
幅值明显不匹配的候选。所有样本 residual clip fraction 均为 0，物理终止为 0，
安全投影最大值不超过 0.0122 rad，因此目前没有证据表明“统一放大 residual
权限”是正确修复。

新增的 target/reference/actual 速度向量和最佳位置时刻给出了更具体的判断：10 条
dynamic-hard reference 的最佳位置通常出现在标记 hit step 之前约 48--51 个控制
步，随后实际 TCP 已离开目标；这首先指向 phase/time-to-hit 或轨迹时序合同问题，
而不是单纯的速度奖励不足。相反，anchor 与部分 holdout reference 在标记 hit
附近位置已经很近，但 actual 速度幅值很低或方向相反；它们才是速度前馈/速度观测
强化的优先对象。

同一 command 坐标下的 reference-prior 差距仍需作为后续单独审计；本轮已补充
root/base 平移责任分解。v5 paired PhysX 回放见
`eval_outputs/p5d3a_root_responsibility_v1.json`。24 条样本 learned 的 TCP
`reference→actual` 均值为 17.99 cm，zero 为 29.12 cm；root 平移误差均值分别为
8.06 cm 和 8.46 cm，且 learned/zero 几乎没有同步下降。因此 root 平移是重要的
全局状态误差，但不能解释 learned 相对 zero 的主要改善，也不能把剩余 TCP 误差
直接归因于基座；root orientation、上肢关节和相位仍需分离回放。该报告明确只做
平移分解，不宣称完整 base 贡献。

v5 还补出了 action-chain 的先验/跟踪器分量：24 条样本命中时刻的
`model_900` primary contribution 最大值均值约 `0.155 rad`，P5D tracker
residual 最大值均值约 `0.051 rad`，且 tracker residual 没有 clip。`0.155 rad`
是既有 model_900 先验的实际命令贡献，不是新的 tracker 输出；因此不能把总的
`reference→actual` 偏差直接归咎于 P5D residual，后续仍需逐关节审计
reference/prior 相容性。

逐关节 v6 审计见 `eval_outputs/p5d3a_joint_responsibility_v1.json`。在 10 条
`dynamic_hard` 中，P5D residual 的绝对均值主要集中在 `right_wrist_roll`
(`0.053 rad`)、`right_wrist_yaw` (`0.050 rad`) 和 `right_elbow` (`0.045 rad`)；
这说明困难样本不是单纯腰部权限不足。5 条 `speed_phase_candidate` 则以
`waist_roll` (`0.050 rad`)、`right_elbow` (`0.039 rad`) 为主，和相位/速度错位
现象一致。冻结 model_900 的大贡献主要位于肩部（dynamic-hard 的
`right_shoulder_pitch/roll/yaw` 接近其配置幅值），所以不能直接把肩部先验贡献
当作 P5D 学习结果。下一步应针对这些责任关节做低维时间/轨迹优化和相位验证，
而不是全 10 维统一放大或启动新训练。

在这些证据完成并重新判定 `dynamic_hard`、`speed_phase` 与 `moderate_learnable`
后，才决定重优化 reference 或强化 phase/velocity 观测；本阶段不扩大数据集，
也不自动启动 P5D-3A 训练。

### P5D-3A phase candidate screening（2026-08-04）

> 历史记录：本节原始 v1/v2 screen 使用了错误的 manifest payload 路径，结论已被
> 文末“基础合同勘误”取代。有效结果只看 `*_fixed` 报告。

已对 10 条 `dynamic_hard` reference 各生成 6 个 endpoint-flat pre-hit phase
warp（共 60 条），保持 hit frame、canonical goal、TCP 合同和命中端点状态不变。
60/60 通过离线软限位、碰撞、速度、加速度和 jerk 门槛，并在正式 PhysX 中完成
60 条 reference-only tracker replay；0 次物理终止。

相对原始 learned replay，10 个 source 各至少有一个候选位置误差下降，典型下降
约 0.2--2.4 cm；但最好候选仍约 33--42 cm，composite teacher pass rate 仍为
0，且整批 replay 的 safety projection 最大达到约 0.060 rad。因此这些候选只能
标记为 `PENDING_PHYSX_PHASE_SCREEN`，不能替换正式 24 条 dataset，也不能直接
进入训练。逐候选排序和安全阈值筛选见
`eval_outputs/p5d3a_phase_reoptimization_screen_v1.json`；候选 manifest 见
`eval_outputs/strike_goal_p5/p5d3a_phase_reoptimization_v1/physx_manifest.json`。

随后使用更大相位幅度 `[-6,-5,-4,4,5,6]` 完成第二轮 60 条候选筛选；报告为
`eval_outputs/p5d3a_phase_reoptimization_screen_v2.json`，结果相对第一轮已饱和，
没有产生足以改变分类的额外收益，正式 24 条 dataset 仍未替换。

当前结论是：phase warp 方向有小幅收益，但不足以解决动态困难 reference，下一步
应在保留最佳候选的前提下做更低维的 PhysX-in-the-loop 时间/轨迹优化，或先补充
reference-prior 与 root/base 责任分解；仍不应盲目扩大 residual 权限或启动新 PPO。

### P5D-3A low-dimensional PhysX search（2026-08-04）

> 历史记录：本节未带 `_fixed` 的 screen 数值是旧日志结果，不能作为当前候选有效性
> 结论；有效 paired replay 以 fixed-path 报告为准。

按责任关节选择了 2 条 `dynamic_hard`（wrist/elbow）和 2 条
`speed_phase_candidate`（waist-roll/elbow），固定 canonical goal、hit frame、
model_900、model_3396 和 model_2198，生成 endpoint-flat 的单关节 phase/amplitude
候选。共 64 条候选，64/64 通过离线限位、碰撞、速度、加速度和 jerk 门槛；首次
批量 screening 仅显示最多约 3.47 cm 的位置改善。

随后对每个 source 的 baseline/best 做同一 manifest 的 learned/zero paired replay。
这是本轮的决定性筛选：dynamic-hard 的两个“best”相对同 source baseline 分别
恶化约 0.71 cm 和 0.40 cm；speed-phase 也没有形成稳定的联合位置/速度收益。
因此 0/7 shortlist candidate 被批准，正式 reference bank 未替换，报告见
`eval_outputs/p5d3a_lowdim_screen_v1.json`、
`eval_outputs/p5d3a_lowdim_paired_v1.json`，候选 manifest 见
`eval_outputs/strike_goal_p5/p5d3a_lowdim_reoptimization_v1/`。

该结果说明当前一因子局部 phase/amplitude 参数不足以形成可重复的共享动态修正；
下一步仍应保持不训练，改为更严格的多参数时间/轨迹优化或先处理 reference-prior
时序相容性，不能把第一次 batch ranking 当成优化成功。

随后又进行了责任关节协同 phase 搜索：dynamic-hard 使用 elbow/wrist 三关节组合，
speed-phase 使用 waist-roll/elbow 组合，共 72 条候选，72/72 离线通过。批量排序中
出现的最多约 1.41 cm 位置改善，在 baseline/best 同一 manifest paired replay 中
没有复现；四个 source 均未形成稳定的位置、速度联合收益。结果见
`eval_outputs/p5d3a_lowdim_screen_v2.json` 和
`eval_outputs/p5d3a_lowdim_paired_v2.json`。这排除了“少量关节 phase 独立或简单
协同补丁”作为当前可靠修复，下一步应回到 reference/prior 的时序合同和执行层建模。

## P5D-3A 基础合同勘误（2026-08-04，必须优先于旧结论）

本轮复核发现一个会直接改变实验有效性的 loader 合同错误。`MotionLibraryLoader`
原先按 `library_motion_npz → motion_npz` 顺序取路径；旧 phase candidate generator
只替换了 `motion_npz`，却保留了源 reference 的 `library_motion_npz`。因此旧的 v2
manifest 和旧 screen log 在运行时实际加载的是原始源 NPZ，不是候选 NPZ。旧的
`p5d3a_phase_reoptimization_screen_v1.json`、`screen_v2.json` 以及对应的旧 lowdim
screen 都标记为 `SUPERSEDED_INVALID_PAYLOAD_PATH`，不得继续作为“phase 无效/饱和”
的证据。v1 manifest 已重新物化为一致 payload，但其旧 log/report 仍然作废。

已完成以下修复：

1. `training/tasks/tracking/mdp/commands.py::_entry_motion_path` 对 p5d3a/canonical
   candidate 在两个显式路径都存在且不一致时 fail-closed，直接报错；旧的 packaged
   manifest（没有 `canonical_motion_npz`）保留其已经约定的 provenance/library 优先级。
2. phase/low-dimensional candidate generator 同时写入
   `motion_npz == library_motion_npz == canonical_motion_npz`，并补齐运行时要求的
   world-frame arrays、placement contract 与 `joint_names_utf8`。
3. fixed-path manifest audit 记录在
   `eval_outputs/p5d3a_motion_manifest_payload_contract_audit_v1.json`；当前 v2
   历史 manifest 为 60/60 冲突，重新物化后的 v1/fixed v1/v2/v3 phase 与 lowdim
   v1/v2 全部通过。

修复后的 phase 回放才是有效证据。它显示正向 pre-hit phase lead 对 10 条
`dynamic_hard` reference 的位置误差均有改善，较大的 lead 可将位置误差从约
`0.38–0.43 m` 降至约 `0.23–0.30 m`；但同时会改变 TCP 速度方向，不能直接批准
为教师。换言之，重复出现的 elbow/wrist 问题首先暴露了一个共享的执行时序滞后，
而不是已经证明了三个关节各自的 sign/order 错误。fixed v2/v3 报告分别为：

* `eval_outputs/p5d3a_phase_reoptimization_screen_v2_fixed.json`
* `eval_outputs/p5d3a_phase_reoptimization_screen_v3_fixed.json`
* `eval_outputs/p5d3a_phase_paired_v2_fixed.json`
* `eval_outputs/p5d3a_phase_paired_v3_fixed.json`

### model_900 原始训练合同的核查纠正

进一步读取原始训练运行的 Hydra 快照
`outputs/2026-07-24/22-00-27/.hydra/config.yaml` 后，确认此前把 V7
coordinator 的合同误写成 model_900 的训练合同。`model_900` 实际来自：

```text
task                 HOPEA3NativeStrikeManifest
scene                fixed-base A3NativeStrikeEnvCfg
manifest             p2_data260708_backhand_strike_only_v1，6 条 backhand reference
num_envs             256
iterations           3000
seed                 7
action                10-D native waist/right-arm residual
native_residual_scale 0.25
raw_clip             0.50
shoulder lookahead   pitch/yaw 各 12 步
velocity feedforward  未配置（默认 none）
```

其网络实际是 actor `56 → 512 → 256 → 128 → 10`，critic `110 → 512 → 256 →
128 → 1`；PPO 为 24 steps/env、5 epochs、4 minibatches、初始学习率 `0.001`、
adaptive KL、`gamma=0.99`、`lambda=0.95`。因此 model_900 不是在 floating-base、
model_3396 共用或 task-phase velocity feedforward 条件下从头训练的。

原始 six-reference 评估的平均 lag-window TCP 位置误差约 `4.72 cm`（范围
`2.98–6.66 cm`），这说明它在固定基座、六条旧 reference 的局部域内有效，但不能
直接视为 canonical 多 reference floating-base tracker。

此前“model_900 训练合同包含 task-phase feedforward”的表述作废。真正的问题是：
当前 P5D 运行在 frozen model_900 上额外启用了 `task_phase, beta=0.75`，这改变了
model_900 原始执行合同；因此当前 12-step + feedforward 训练应视为**合同迁移实验**，
而非对原 model_900 合同的严格复现。

当前责任判断应改为：

* **确定的基础错误**：candidate manifest payload path 冲突；以及先前把 coordinator
  运行合同误认成 model_900 原始训练合同。
* **强证据的系统问题**：fixed-path phase screen 对正向 pre-hit lead 敏感，存在
  共享的执行时序差异，但是否需要 feedforward 仍未定。
* **尚未证明的关节级问题**：wrist roll/yaw、elbow、waist roll 的 sign/order 或
  单独 actuator 缺陷，必须在统一合同下重新判断。

已完成一次只读的 contract-fixed replay（24 环境、130 控制步，未训练）：

```text
                         mean TCP position error
旧 P5D-2 合同 / learned             0.1793 m
旧 P5D-2 合同 / zero residual       0.2912 m
恢复 model_900 合同 / learned       0.3464 m
恢复 model_900 合同 / zero residual  0.3623 m
```

这不是“恢复合同后模型变差”的最终结论，而是证明 `model_2198` 已经把旧的错误
prior timing 一并学进 residual；把 model_900 合同修正后，旧 checkpoint 不再具有
可比性。好的一点是 corrected-contract 下 residual 仍有约 1.6 cm 的平均改善，说明
闭环仍在工作；但必须先做 contract-consistent 的新 baseline/paired replay，再判断
elbow/wrist 是否还有独立问题。日志为
`eval_outputs/p5d3a_contract_fixed_model2198_replay.log` 和
`eval_outputs/p5d3a_contract_fixed_model2198_zero_replay.log`。

在完成 contract-consistent 的零 residual / model_2198 paired replay、并用速度方向
和位置共同评分之前，不启动新的 PPO、不替换 reference bank、不批准任何 teacher。

### 肩部 12 步 lookahead 的只读消融

为回答“固定提前相位是否只是针对某条 motion 的特定补丁”，在同一 manifest、同一
`model_2198`、同一 24 环境/130 步 PhysX 回放中，保持 `task_phase` 速度前馈
(`beta=0.75`，肩 pitch/yaw) 不变，只切换肩部 `lookahead=12` 与 `lookahead=0`。
本实验未训练、未改 canonical goal、未改 safety filter。

结果如下：

```text
                         lead=12                 lead=0
learned position       0.3464 m                 0.1861 m
learned velocity err   1.9308                   1.0773
learned normal err     16.28 deg                11.54 deg
zero position          0.3623 m                 0.2844 m
zero velocity err      1.5443                   0.9080
zero normal err        17.69 deg                21.01 deg
```

因此在这个 checkpoint 上，`lead=0` 的 learned 位置、速度和法向平均值都优于
`lead=12`；但这不能直接宣布“0 步提前是最终正确配置”，因为 `model_2198` 是在
旧的 P5D 运行合同（无 lookahead、无 velocity feedforward）下训练的，`lead=12`
对它属于合同外分布。该结果首先证明：**固定 lookahead 是共享的、关节特定的执行先验，
不是 per-motion 目标改写；但它会改变策略输入/执行合同，因此必须在合同一致地重新训练后，
用 train/validation/holdout 分别验证迁移性。** 不允许为每条 reference 单独调 lead。

日志：

* `eval_outputs/p5d3a_lead12_ff_explicit_model2198_replay.log`
* `eval_outputs/p5d3a_contract_fixed_model2198_zero_replay.log`
* `eval_outputs/p5d3a_lead0_ff_model2198_replay.log`
* `eval_outputs/p5d3a_lead0_ff_model2198_zero_replay.log`

本消融只用于定位合同兼容性，不批准教师，也不启动训练。
