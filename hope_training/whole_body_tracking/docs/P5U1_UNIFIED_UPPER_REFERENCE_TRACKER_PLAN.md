# P5U-1 Unified Upper Reference Tracker

## Physical READY stance contract (mandatory)

Every P5U training and replay must use the previously qualified V22 wide/deep
staggered READY stance: `staggered_stance_half_span_m=0.04 m` (left foot
forward/right foot backward), `stance_lateral_widen_per_foot_m=0.04 m`, and
`stance_knee_flexion_rad=0.42`. The runtime computes the corresponding
hip/ankle pitch-roll compensation and pelvis-height correction. A default
parallel, shallow stance is not a valid P5U contract. Forced visual replays
must preserve this physical READY pose and may only change the future motion
reference; they must not teleport the robot to motion frame zero.

状态：正式执行计划（2026-08-04）  
来源：[P5U-1 Unified Upper Reference Tracker Plan.docx](</home/bistu/下载/P5U-1%20Unified%20Upper%20Reference%20Tracker%20Plan.docx>)  
来源 SHA-256：`7bd2e87a70bd7f790d741adc784cdc53515da55c56a7454c32cc3acf01476fd2`

## 0. 执行原则

本计划不是继续修补旧 `model_900 + P5D residual`，而是重新训练一个与最终部署合同一致的 floating-base、canonical multi-reference、reference-conditioned unified upper-body tracker。

当前正在运行的 4096×1500、12-step lookahead + task-phase feedforward 实验必须完成并归档，但只能标记为：

```text
DIAGNOSTIC_CONTRACT_MIGRATION_NOT_FINAL
```

禁止把它作为正式模型、教师模型或新训练 warm start。禁止因 reward 上升而宣称 canonical 击球成功。

## 1. 已确认的历史事实

原始 `model_900` 的真实训练合同来自 `outputs/2026-07-24/22-00-27/.hydra/config.yaml`：

```text
task                 HOPEA3NativeStrikeManifest
base                 fixed-base
reference            6 条旧 backhand reference
num_envs             256
iterations           3000
seed                 7
actor                56 → 512 → 256 → 128 → 10
critic               110 → 512 → 256 → 128 → 1
shoulder lookahead   pitch/yaw 各 12 步
velocity feedforward none
native residual      0.25
raw clip             0.50
```

PPO 为 learning rate `0.001`、adaptive KL、5 epochs、4 minibatches、`gamma=0.99`、`lambda=0.95`。原始六条 reference 的 lag-window 平均位置误差约 `4.72 cm`，范围约 `2.98–6.66 cm`。这只能证明 fixed-base 六 reference 局部能力，不能证明 floating-base、canonical multi-reference、未见 reference 泛化或 10D canonical 联合命中。

此前把 task-phase velocity feedforward 误写成 `model_900` 原始合同的结论作废。

## 2. 当前 B3 实验的定位

当前实验为：

```text
旧 model_900 权重
+ shoulder lookahead = 12
+ task_phase velocity feedforward, beta = 0.75
+ P5D residual
+ 4096 environments × 1500 iterations
```

它是合同迁移诊断，不是原始 `model_900` 复现。完成后 paired PhysX 归档，但不得 warm start P5U。
当前 B3 使用的是 `p5d2_all_manifest.json`，包含全部 24 条 reference；因此它不是 split-clean
实验，只能作诊断。P5U 正式训练必须使用 `p5d2_train_manifest.json`，并将 validation 与
holdout manifest 封存到合同选择完成之后。

## 3. 基础链路防护

### 3.1 Reference payload

候选 manifest 必须 fail-fast 校验：manifest path、resolved runtime path、实际加载 NPZ、trajectory hash、joint-order hash、world arrays、hit frame、control dt。任一不一致立即报错，不得静默回退到 source motion 或优先使用冲突的 `library_motion_npz`。

必须测试：candidate/source 同时存在时加载 candidate；制造路径冲突时失败；不同 candidate 的 runtime hash 不同；4096 环境独立采样且 phase、hit frame、joint order 不串环境。

### 3.2 Checkpoint 合同

checkpoint 必须绑定并校验：task、base mode、manifest version、observation schema hash、action schema hash、lookahead、velocity feedforward mode/beta、control frequency、episode timing、TCP、frame、joint order、scale、clip、actuator 和 safety 配置。禁止只加载权重而静默改变合同。

## 4. P5U 正式架构

```text
canonical_goal_10d
    ↓
verified multi-reference trajectory
    ↓ q_ref / dq_ref / future reference
new unified upper tracker
    ↓ q_ref_upper + upper residual
safety filter
    ↓
PhysX actual
```

下肢保留冻结 `model_3396` 支持、状态机、足底支持和 safety filter。正式新架构默认移除 `model_900` runtime upper prior，不再要求新模型先抵消旧 model_900 的固定时序。

固定合同：`canonical_goal_10d` 是唯一真值；不得修改目标或把 actual 写回 target；control anchor 仅诊断；actor 不接收 motion/reference/seed ID；motion 只作离线初值和审计字段；train、validation、holdout 严格隔离；所有 reference 先过 payload contract。

## 5. 新 upper tracker action

第一版使用腰部+右臂 10D 位置 residual：

```text
q_cmd_upper = q_ref_upper + action_scale * actor_action
```

输出层严格零初始化，使初始策略为 `q_cmd=q_ref`。raw clip 和逐关节 scale 配置化，记录 max/RMS/clip rate，不统一放大所有关节。不加入 motion-specific phase action 或 reference 专属补丁。若纯位置 residual 后续确实无法修正速度，再单独评估 velocity residual，不在第一版同时改变多个控制变量。

## 6. Actor observation

至少包含：`q_actual`、`dq_actual`、`q_ref`、`dq_ref`、可选 `ddq_ref`、当前误差、future reference preview、phase、time-to-hit、marked hit step、actual/reference TCP position/normal/velocity、TCP 三类误差、base orientation/angular velocity/必要的 linear velocity、foot/contact 状态和上一动作。

future preview 至少为 `+1/+3/+6/+12` steps。这里的 `+12` 是 observation preview，不是硬编码 shoulder command lead。

## 7. 合同短消融

正式长训练前使用同一 reference、seed、初始状态完成：

### Contract A（默认推荐）

```text
command lookahead = 0
velocity feedforward = none
多尺度 future reference observation
```

### Contract B

所有上身关键关节统一、显式、配置化 command lookahead；velocity feedforward 为 none。禁止只给肩部静默设置 lead。

### Contract C

command lookahead 为 0，测试 `velocity feedforward = beta*dq_ref`，`beta ∈ {0.50, 0.75, 1.00}`。

不得在第一轮同时改变 lookahead 与 feedforward。用 validation canonical position、velocity vector、normal、timing、termination、safety projection、residual saturation 选择合同，不用总 reward 选择。

## 8. Reference 数据

第一阶段继续使用严格 payload 修复、去重后的 24 条 reference，motion 4 排除：

```text
16 train
4 validation
4 holdout
```

覆盖多个 workspace region；不得重复同一 canonical goal 的多 seed 版本。validation 仅用于合同/checkpoint 选择，holdout 在最终模型确定前封存。

## 9. Multi-reference correctness

必须证明 4096 环境同一 rollout 中存在多个不同 reference，并输出 active reference count、unique runtime trajectory hash、每条 reference 和 region 的 sample count、最大/最小比例。actor 不接收任何 ID。

条件化测试必须验证：相同 actual state 下不同 reference 产生不同 reference observation；future preview 交换会改变输出；preview 置零的输出变化被记录。

## 10. 训练前基线矩阵

### B0：Reference-only

`model_3396 frozen`、`model_900 disabled`、new upper residual=0、`q_cmd_upper=q_ref`。

### B1：原始 model_900 回归

仅在原始 fixed-base、6 条旧 reference 上复现：lookahead 12、velocity feedforward none、residual 0.25、raw clip 0.50。目的只是确认历史能力可复现。

### B2：model_900 floating-base migration

当前 24 条 canonical reference、`model_3396 frozen`、model_900 原始 lookahead 合同、P5D residual=0。仅作迁移对照，不是正式新架构。

### B3：当前 1500 轮实验

完成后 paired replay，标记 `DIAGNOSTIC_ONLY`，禁止 warm start。

## 11. Reward

全周期包括 joint position/velocity、TCP reference position/normal/velocity、base/foot stability、limit/collision、residual magnitude/action rate。命中窗口必须直接奖励 actual TCP 到 canonical position、normal、velocity vector（方向、幅值和沿目标速度投影）以及 marked hit timing。

必须惩罚到点停车、速度方向相反和错误时刻通过目标。Mean reward 上升但 canonical 指标不改善时触发失败门槛，不继续宣称有效。

分别记录 joint tracking、TCP tracking、canonical position/normal/velocity、timing、stability reward。

## 12. 分阶段执行

### Stage 0：静态/执行链

payload、合同 hash、model_900 未调用、model_3396 调用且冻结、10D zero action、safety、NaN、reference 隔离全部通过。

### Stage 1：Smoke

64–128 env、10–20 iterations、至少 8 条 train reference，检查 loss、稳定性、canonical reward、多 reference 和冻结权重。

### Stage 2：合同短消融

每个合同 1024 或 4096 env、300–500 iterations、相同 seed/data；不使用 holdout 调参。

### Stage 3：正式训练

短消融通过后，从严格零初始化启动 4096 env、2000 iterations。不得加载 model_900、model_2198 或 B3 checkpoint。保存并评估 `100,200,300,400,500,600,800,1000,1200,1400,1600,1800,2000`，按 validation paired PhysX 选最佳 checkpoint，而不是默认最后一轮。

## 13. 失败保护

若 300–500 轮 joint tracking 改善但 canonical position/normal/velocity 均无改善，停止该合同并审计 canonical reward 是否进入 actor loss、marked hit 索引、reference observation、action→processed command、safety 裁剪和 reward 权重。

只有在 residual 接近 clip、processed command 随 action 变化、safety 未完全裁剪且 actual 对该关节 action 单调改善时，才允许提高单关节 scale。禁止全局放大。

## 14. 评估与教师资格

每条 reference 输出 target→reference、reference→command、command→actual、actual→canonical、position/normal/velocity vector/magnitude/direction、timing、base、root、termination、safety projection、residual max/RMS/clip；汇总 mean/median/P75/P90/P95/worst、改善率、退化率和 train/validation/holdout 分区。

只有 verified reference + model_3396 + new unified upper tracker + safety + PhysX actual 同时通过 canonical position、normal、velocity、timing、safety、recovery 门槛，才可标记 `QUALIFIED_TEACHER`。

## 15. 禁止事项

禁止把 B3 当正式模型；从错误合同 checkpoint warm start；继续把 model_900 当不可替代 upper prior；只增加轮数；只看 Mean reward；用 lag-window 最佳位置替代 marked hit；修改 canonical target；actual 写回 reference；输入任何 ID；静默加载 source motion；训练/部署合同不一致；同时引入未经消融的多个控制变量；validation/holdout 泄漏。

## 16. 严格执行顺序

1. 完成并归档当前 B3 1500 轮诊断实验。
2. 复现原始 model_900 fixed-base 合同（B1）。
3. 固化 payload fail-fast 测试和 checkpoint 合同 hash。
4. 实现无 model_900 的 unified upper tracker。
5. 完成 B0/B1/B2/B3 paired baseline 和 multi-reference correctness test。
6. 运行 smoke test。
7. 运行 Contract A/B/C 短消融。
8. 按 validation canonical 指标选择合同。
9. 从零启动 4096×2000 正式训练。
10. 周期性 paired PhysX，选择 validation 最佳 checkpoint。
11. 最后只执行一次 holdout。
12. 对完整通过门槛的轨迹单独授予教师资格。

完成标准不是 reward 上升，而是在正确 candidate payload、统一合同、floating-base、多 canonical reference 条件下，从零训练出不依赖 model_900 的 upper tracker，并在 validation 和 holdout 同时改善 canonical position、normal、velocity 和 timing。

并行预检记录：[p5u1_parallel_preflight_v1.json](../eval_outputs/p5u1_parallel_preflight_v1.json)。
B1 原始 model_900 回归预检：[p5u1_b1_original_model900_preflight_v1.json](../eval_outputs/p5u1_b1_original_model900_preflight_v1.json)。B1 仅完成配置和权重一致性预检，必须等 B3 归档后才可运行 PhysX。

## 17. 严格跌倒审计合同（2026-08-04）

`terminated_count=0` 不能单独证明机器人没有跌倒。P5U 回放和训练必须同时记录并审计：

* root height；
* root projected-gravity upright/tilt；
* 足底接触与滑移；
* 非足身体接触；
* 终止原因及首次终止控制步。

当前严格终止门槛为：root tilt 超过 `45°`、root height 低于 `0.82 m`，或 `torso_Link` 自身 tilt 超过 `45°` / 高度低于 `0.70 m`，任一条件持续 `2` 个控制步即判定 `strict_fall`。这是必要的：浮动 root 仍直立时，腰部/上身已经可能塌倒。原有 `1.55 rad` recovery tilt 与 `0.65 m` base-height 只保留作兼容诊断，不能替代严格判定。

非足身体接触阈值同步从 `10 N` 收紧到 `1 N`；足底、球拍和手部允许接触过滤保持不变。

训练同时加入 `strict_fall_risk` dense penalty，在 root upright 小于 `0.80`、root height 低于 `0.90 m`，或 torso upright 小于 `0.85` / torso height 低于 `0.80 m` 时提前惩罚；终止后继续使用 `fall=-150` 结果惩罚。任何视觉上明显倒下但未触发现有旧阈值的回放，都必须按严格审计重新判定，不能报告为稳定。

训练入口还执行 fail-fast 合同检查：凡任务名包含 `UnifiedUpperReferenceTracker`，若组合后的 working-tree 配置缺少 `terminations.strict_fall`、`rewards.strict_fall_risk`，或其阈值不是 `45° / 0.82 m / 2 steps`，训练在创建环境前直接拒绝启动。这样不会因为 Hydra 组合、旧安装包或 YAML 漏配而静默训练一个没有跌倒惩罚的策略。
