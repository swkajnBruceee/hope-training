# HOPE 模型优化方案

> 状态：设计基线
>
> 版本：v1.4
>
> 更新时间：2026-08-13
>
> 适用项目：`official_hope`

本文档是后续模型代码修改、训练实验和结果分析的参照基线。除非明确更新本文档，否则不改变本文档规定的模型边界、接口约束和实验顺序。

## 1. 优化目标与范围

本项目只优化策略模型，不修改任务定义和仿真环境。核心目标是在 HOPE 官方预训练策略的基础上，学习小幅、受约束的任务相关策略修正，同时保留官方模型已有的全身运动能力。

### 1.1 允许修改的部分

- Actor 网络结构；
- Actor 参数初始化和冻结/解冻策略；
- PPO 中 Actor 的策略均值、分布和训练方式；
- Critic 的初始化和正常微调方式；
- 模型训练配置；
- 模型导出和 PyTorch/ONNX 一致性测试；
- 模型侧的消融实验与诊断指标。

### 1.2 明确不修改的部分

- 仿真环境、机器人模型、球台和来球逻辑；
- reward、termination、curriculum 和 motion 文件；
- 训练任务定义；
- Actor observation/action 的维度和关节顺序；
- 原有动作解码、安全限幅和部署接口。

## 2. 当前官方模型契约

当前模型基线为：

```text
task:              HOPE-HitterPingPong-AgibotA3-v0
checkpoint:        checkpoints/model_21800.pt
actor observation: hitter_pure, 110D
actor action:      31D raw joint-position residual
actor MLP:         [512, 256, 128]
control frequency: 50 Hz
```

官方动作不是轨迹点，而是原始关节位置残差，动作解码关系为：

```text
q_des = default_q + raw_action * action_scale
```

head 的被动关节和已有动作适配逻辑必须继续遵循当前实现。

观测契约的唯一参考为：

[actor_observation_contract.py](../source/whole_body_tracking/whole_body_tracking/tasks/tracking/actor_observation_contract.py)

当前 `hitter_pure` 的 110 维结构为：

| 观测项 | 维度 | 模型侧归类 |
|---|---:|---|
| `base_ang_vel` | 3 | proprioception |
| `joint_pos` | 31 | proprioception |
| `joint_vel` | 31 | proprioception |
| `actions` | 31 | proprioception |
| `projected_gravity` | 3 | proprioception |
| `base_forward_xy` | 2 | proprioception |
| `base_target_delta_xy` | 2 | spatial goal |
| `racket_target_rel_base` | 3 | spatial goal |
| `racket_target_vel_w` | 3 | spatial goal |
| `time_to_strike` | 1 | temporal condition |

结构化编码网络只能根据该契约拆分输入，不能新增观测字段或改变字段顺序。

### 2.1 Observation preprocessing 契约

Residual 和 Structured Residual 必须接收与官方 HOPE Actor 完全相同的 observation preprocessing 输出：

```text
raw hitter_pure observation (110D)
        ↓
official observation preprocessing / normalization
        ├──────── Official HOPE Actor
        └──────── Residual / Structured Residual branch
```

当前官方 `model_21800.pt` 对应的实际契约为：

```text
observation_normalization: none
normalizer state:           absent
Residual input:             same raw 110D tensor as official Actor
```

当前 `cfg/algo/ppo.yaml` 中的 `empirical_normalization` 不等于 Actor running normalization；项目现有配置说明、checkpoint 内容和 ONNX manifest 均表明这一 lineage 使用 identity preprocessing。Residual MVP 不得自行创建或在线更新独立的 RunningMeanStd。

如果未来官方 Actor 真的启用 observation normalization，Residual 必须复用官方 normalizer 及其状态；独立 branch normalization 只能作为单独消融实验，并写入独立模型 metadata。

### 2.2 `actions` 观测语义

`hitter_pure` 中的 31D `actions` 不是 Residual 输出，也不是任意的 `last_action` 缓存。当前任务使用：

```text
policy action_t
      ↓
same ActionManager / joint_pos action path
      ↓
passive-joint override and applied-raw bookkeeping
      ↓
actions observation_(t+1) = applied_raw_actions
```

`applied_raw_actions` 是原始 action 域中实际应用的动作；被动 head 关节列会被置零。它不是 `q_des`，也不是仅有 `Δμ` 的 Residual。Residual policy 必须只替换送入官方 ActionManager 的完整 31D action，不能改变该观测字段的语义。

实现时必须验证：

```text
HOPE:     action_t → ActionManager → actions_obs_(t+1)
Residual: combined_action_t → same ActionManager → actions_obs_(t+1)
```

依据为：

[hope_env_cfg.py](../source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_env_cfg.py:1731)

[hope_observations.py](../source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_observations.py:72)

[hope_actions.py](../source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_actions.py:146)

### 2.3 Checkpoint 与 optimizer 恢复契约

官方 `checkpoints/model_21800.pt` 当前包含：

```text
model_state_dict:      actor / critic / std
optimizer_state_dict:  official optimizer moments and parameter groups
iter:                  checkpoint iteration metadata
infos:                 optional training metadata
normalizer state:      absent
```

架构消融（Residual、Larger Residual、Structured Residual、Structured+FiLM）统一采用：

```text
actor / critic / std:  load from official checkpoint
normalizer:            load official state; current state is identity/none
optimizer:             reinitialize
Residual parameters:   initialize new
experiment iteration:  start from zero
```

这样不会把官方 Adam/AdamW 的 `exp_avg`、`exp_avg_sq` 动量带入新增参数或冻结 Actor 的新 parameter groups。

`Direct HOPE Fine-tune` 主对照定义为 genuine continuation：恢复 actor、critic、std 和 optimizer state，并保留 checkpoint iteration metadata；但由于官方权重不包含完整 `hope_exact_resume_state`，不得将其描述为物理环境边界的 exact resume。另设 `Direct FT — Frozen Std` 扩展对照时，恢复 actor/critic、冻结 std，并重新初始化 optimizer。

### 2.4 Active-action mask

模型仍然保持 31D policy contract，但 Residual 只允许修改有实际控制权限的 action columns。定义：

\[
m\in\{0,1\}^{31}
\]

其中主动关节 `m_i=1`，被动关节 `m_i=0`。Residual mean 必须为：

\[
\mu_{new}=\mu_{HOPE}+m\odot\Delta\mu
\]

被动 action 维度不允许进入 Residual 学习；否则 PPO log-prob 可能因无效维度变化，而 ActionManager 随后又将这些维度置零，造成没有环境作用的虚假更新。

`m` 必须从当前 action contract/被动关节配置解析，不能在模型代码中硬编码列号。`Δq_max` 只对 active dimensions 定义；对于 `m_i=0` 的维度，`Δq_max_i=0`，且不得执行 `Δq_max_i / action_scale_i` 的除法，从而避免被动维度 `action_scale_i=0` 时除零。

active mask 必须保存到 checkpoint 和导出 metadata，并在加载时校验当前 ActionManager contract。

## 3. 总体模型路线

模型优化按以下顺序推进：

```text
HOPE Baseline
      ↓
Frozen HOPE + Residual Mean Adaptation
      ↓
Residual + Proprio/Goal/Time Encoders
      ↓
Time-Conditioned Structured Residual
      ↓
Controlled Fine-tuning Ablation
```

开发路线与实验继承关系必须分开。开发时可以按 Residual → Structured → FiLM 的顺序实现，但架构消融实验默认全部从同一个官方 checkpoint 独立开始，不默认继承上一实验已经训练过的 Residual 权重。

对于初始化原则：

- Residual 最后一层零初始化，使 `Δμ=0`；
- FiLM 使用 `γ=1+Δγ, β=Δβ` 参数化，且 `Δγ=0, Δβ=0` 零初始化；
- 新的输入编码结构首先只作用于 Residual 分支，官方 Actor 保持不变；
- 只有能够严格 identity-initialize 的增量模块（例如 FiLM）才要求初始化时等价于其直接前置结构；
- Structured Encoder 不要求、也不应被描述为初始化时严格等价于已经训练好的 Plain Residual MLP。

## 4. Residual Mean Adaptation

### 4.1 策略定义

Residual 必须作用于 Actor 的策略均值，而不是对两个已经采样的 action 做相加：

\[
\mu_{new}(o)=\mu_{HOPE}(o)+\Delta\mu_\theta(o)
\]

这里的 `o` 指经过官方 observation preprocessing 的 110D 输入：

\[
o_{proc}=N_{HOPE}(o_{raw}),
\qquad
\Delta\mu_\theta=f_\theta(o_{proc})
\]

当前 checkpoint 中 `N_HOPE` 是 identity；如果未来官方开启 normalizer，HOPE Actor 和 Residual 必须共享同一个 `N_HOPE`。

其中：

\[
\Delta\mu_i(o)=m_i\,\Delta\mu^{max}_i\tanh(f_{\theta,i}(o))
\]

其中 `m` 是当前 ActionManager contract 解析出的 active-action mask；被动维度的 Residual 输出在策略均值层面直接为零。

策略分布必须完整继承官方 HOPE 的 action distribution 实现，本文不预先将其写死为普通 Gaussian、clip 或 tanh-squashed Gaussian。

Residual 的 bound 优先用物理关节目标单位定义。第 `i` 个关节的最大物理修正量记为：

\[
\Delta q^{max}_i\quad(\mathrm{rad})
\]

根据当前动作解码关系换算到 raw action mean 单位：

\[
\Delta\mu^{max}_i=
\frac{\Delta q^{max}_i}{|action\_scale_i|}
\]

`action_scale_i` 必须从现有动作适配配置读取，不能在模型中重复硬编码。最终动作解码仍然是：

\[
q_{des}=q_{default}+action\_scale\odot a_{raw}
\]

确认训练稳定后，再比较统一物理 bound 与逐关节物理 bound：

\[
\Delta q^{max}\in\mathbb{R}^{31}
\]

逐关节限制的目的，是允许手腕等快速响应关节进行较大修正，同时限制躯干和腿部关节的过度偏移。

### 4.2 Residual-side feature preprocessing

为保证 Plain Residual、Larger Residual、Structured Residual 和 Structured+FiLM 的架构消融公平，所有 Residual 变体必须共享同一个 Residual-side feature preprocessing：

```text
official processed 110D observation
              ↓
Residual-side feature preprocessing P_R
              ├── time_to_strike → τ_hat
              └── other terms    → unchanged
              ↓
Plain / Larger / Structured Residual
```

当前唯一规定的 Residual-side 变换是：

\[
\hat\tau=clip(\tau/\tau_{scale},-1,1)
\]

Plain Residual 和 Larger Residual 将 `P_R(o)` 作为完整 110D 输入；Structured Residual 再依据 observation contract 将同一个 `P_R(o)` 拆为 proprioception、spatial goal 和 time 三部分。这样 `Plain → Structured` 只改变网络结构，不同时改变 `time_to_strike` 的尺度。

该处理不改变 HOPE Actor 的输入；HOPE 仍接收官方 preprocessing 输出。未来若增加其他 Residual-side normalization 或 feature transform，必须对所有 Residual 变体共享，并作为独立消融和 metadata 字段记录。

### 4.3 初始化与训练顺序

- 加载官方 Actor 权重；
- 官方 Actor 初始冻结；
- Residual 网络最后一层权重和偏置置零；
- 初始 Residual 输出必须为零或数值上接近零；
- 第一阶段只训练 Residual；
- 每个实验固定一组 `Δq_max`，不在 rollout 或 PPO update 中间动态改变；
- 不使用未经验证的固定 `α=0.1~0.3` 作为通用标准，修正幅度由物理单位的 `Δq_max` 定义；
- 若未来研究 bound schedule，只能在完整 PPO iteration 结束后修改，并将当前 bound 写入 checkpoint。

### 4.4 PPO 分布与 log-prob 约束

必须完整继承 HOPE 当前 PPO action distribution 的处理逻辑，只替换 Actor mean 的生成方式。Residual MVP 的参数状态必须明确为：

```text
HOPE Actor mean:   frozen
HOPE std/log_std:  frozen
Residual mean:     trainable
Critic:            trainable
```

因此，Residual 零初始化时不仅要求 `μ_new=μ_HOPE`，还要求官方策略分布的尺度保持不变。后续若开放 `std/log_std`，必须作为独立训练策略实验，不得与 Residual MVP 混在一起。

如果动作采样后还经过 clip：

```text
a_raw  ~ policy distribution
a_exec = clip(a_raw)
```

PPO 的 log-prob 必须遵循官方分布对采样动作的定义，不能简单把执行侧的 `a_exec` 重新塞回某个假定的 Gaussian 计算 log-prob。除非原实现明确使用了带 Jacobian correction 的 tanh-squashed Gaussian，否则不得自行改变这一逻辑。

当前分布和动作行为的主要参考为：

[bounded_actor_critic.py](../source/whole_body_tracking/whole_body_tracking/utils/bounded_actor_critic.py)

## 5. Proprioception/Goal/Time 编码

为保证 Structured Residual 消融不丢失 `time_to_strike`，时间信息从该阶段开始就必须保留。模型使用状态分支、空间目标分支和独立时间分支：

```text
proprioception → Proprio Encoder ─┐
spatial goal   → Goal Encoder ────┼→ Fusion → Residual Head → Δμ
time_to_strike → Time Encoder ────┘
```

定义：

\[
h_p=E_p(o_p),\qquad h_g=E_g(g_{spatial}),\qquad h_t=E_t(\hat\tau)
\]

\[
h=F([h_p,h_g,h_t])
\]

其中：

```text
o_p = base_ang_vel, joint_pos, joint_vel, actions,
      projected_gravity, base_forward_xy

g_spatial = base_target_delta_xy, racket_target_rel_base,
            racket_target_vel_w

τ_hat = normalized(time_to_strike)
```

`time_to_strike` 不放入 Spatial Goal Encoder，而是通过独立的 `Time Encoder` 输入 Fusion。这样普通 Structured Residual 阶段和加入 FiLM 后的阶段都使用完整的 110D 信息；从 Structured Residual 到 Structured Residual+FiLM，新增的只有调制机制，而不是新增输入信息。

## 6. time-to-strike FiLM

项目中的 `time_to_strike` 由：

```text
(strike_step - motion.time_steps) * step_dt
```

计算，且代码使用 `time_to_strike > 0` 判定击球前状态、使用负值判断击球后的恢复阶段。因此不能把时间简单裁剪到 `[0,1]`，否则击球后的时间信息会全部坍缩到同一个值。

时间条件采用对称归一化：

\[
\hat\tau=clip(\tau/\tau_{scale},-1,1)
\]

其中 `τ_scale` 只能由训练数据、验证场景或任务配置确定，并写入模型配置/checkpoint；不能使用 held-out final evaluation 或部署测试分布反向调参。部署阶段只统计时间饱和比例：

\[
R_{\tau,sat}=\frac{\#(|\tau|>\tau_{scale})}{\#\tau}
\]

归一化只发生在模型内部，不改变 110D observation contract。

实现依据为：

[hope_observations.py](../source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_observations.py:55)

[hope_commands.py](../source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py:5838)

其中 `time_to_strike` 由命令项直接输出，击球后的负值由现有恢复逻辑保留。

再生成 FiLM 的增量参数：

\[
[\Delta\gamma,\Delta\beta]=MLP(\hat\tau)
\]

对融合特征进行调制：

\[
h'=(1+\Delta\gamma(\hat\tau))\odot h+\Delta\beta(\hat\tau)
\]

FiLM 最后一层权重和偏置初始化为零，使得：

\[
\Delta\gamma=0,\qquad \Delta\beta=0
\]

从而 `h'=h`。这样加入 FiLM 后，模型初始行为与未加入 FiLM 的 Residual 模型一致。

## 7. Critic 策略

Critic 不做 Residual 化：

```text
Actor:  HOPE Actor + Residual Adaptation
Critic: 普通 Value Network，正常 PPO 更新
```

原因是 Critic 只用于训练辅助，不部署；Actor 改变后状态访问分布也会改变，Critic 应该允许正常适应。若官方 checkpoint 提供 Critic，可以作为初始化，但不长期冻结。

Critic 的 observation space、normalization、privileged information 和 checkpoint loading 独立遵循官方实现，不强制与 Actor 的 110D `hitter_pure` observation 一致。修改 Actor 时不得默认同步修改 Critic 输入契约。

## 8. 受控微调方案

是否解冻官方 Actor 应作为实验变量，而不是默认必做步骤。至少比较：

| 模型 | HOPE Actor | Residual |
|---|---|---|
| M1 | 全部冻结 | 训练 |
| M2 | 只解冻最后一层 | 训练 |
| M3 | 全部解冻 | 训练 |

解冻时使用比 Residual 更保守的学习率，并始终保留 Residual 的幅度限制。若 M1 已经取得最佳稳定性，则保留冻结方案，不强行进行全量微调。

### 8.1 Controlled FT optimizer contract

Controlled FT 开始时，原本冻结的 HOPE 参数变为 trainable，不能让 optimizer 通过默认行为默默决定状态迁移。首选实现为：

```text
existing Residual/Structured parameters:
    retain existing optimizer state
newly unfrozen HOPE parameters:
    add a new optimizer parameter group
    initialize optimizer moments to zero
HOPE learning rate:
    configured as a smaller scale than Residual learning rate
std/log_std:
    follow the selected experiment contract
```

`hope_unfreeze_lr_scale` 作为配置和 metadata 字段保存，具体数值由 validation 选择，不在本文档中固定。如果训练框架无法安全增加 parameter group，则 Controlled FT 必须在开始时整体重建 optimizer，并将 `optimizer_restore_mode=controlled_ft_rebuild` 写入 metadata；不得静默采用另一种方式。

### 8.2 Direct HOPE Fine-tune 定义

`HOPE Fine-tune` 是无 Residual、无新结构的直接微调对照：

```text
Actor mean:       trainable
std/log_std:      按官方 continuation-training 逻辑处理
Critic:           trainable
Architecture:     original HOPE Actor/Critic
Residual:         none
```

主对照采用 genuine continuation：加载官方 actor/critic/std、optimizer state 和 iteration metadata。该对照必须固定 learning rate、PPO epochs、rollout length 和总 environment steps，用于回答：直接继续训练官方 Actor 是否已经足够，以及 Residual 结构是否带来额外收益。

扩展消融 `Direct FT — Frozen Std` 定义为：

```text
HOPE mean:        trainable
HOPE std/log_std: frozen
Critic:           trainable
Residual:         none
Optimizer:        reinitialize
```

该扩展项与 Residual MVP 共享 std 和 optimizer 条件，只比较“直接更新 HOPE mean”与“通过 bounded Residual 更新 mean”的参数化差异，不作为首个 MVP 必做项。

## 9. 实验与消融矩阵

每个实验必须固定环境、reward、motion 文件、环境数量、训练步数和评估协议；每一步只回答一个模型问题。架构消融之间不得默认继承上一实验已经训练过的 Residual 权重；除明确的 `Controlled HOPE FT` 外，各架构对照均从同一个官方 checkpoint 独立开始。开发阶段每个配置至少使用 3 个训练 seed（例如 `0, 1, 2`），最终报告使用 `mean ± std`；最终候选模型建议使用 5 个 seed。

实验拓扑为：

```text
                           Official HOPE
                                │
               ┌────────────────┼────────────────┐
               ↓                ↓                ↓
          Direct FT      Plain Residual    Larger Residual
                                                │
                                ┌───────────────┘
                                ↓
                    Structured Residual
                    Proprio / Goal / Time
                                ↓
                   Structured Residual + FiLM
                                ↓
                      Controlled HOPE FT
```

其中 `Structured Residual + FiLM` 也从同一个官方 checkpoint 和零初始化 Residual 开始，不从已训练的 Structured Residual checkpoint 继续训练。只有 `Controlled HOPE FT` 明确作为最终候选架构的后续微调实验，才允许从该架构的选定 checkpoint 开始。

| 实验 | 目的 |
|---|---|
| HOPE Baseline | 确认官方模型能力 |
| HOPE Fine-tune | 普通微调对照 |
| Direct FT — Frozen Std | 在相同探索尺度下比较直接 mean 微调与 Residual mean 微调 |
| HOPE + Residual | 验证残差适配是否有效 |
| HOPE + Larger Residual MLP | 排除参数量增加因素 |
| HOPE + Structured Residual | 验证结构化输入是否有效；该阶段包含独立 `Time Encoder` |
| Structured Residual + FiLM | 在保持完整 110D 信息不变的前提下，验证显式击球时间调制是否有效 |
| Last-layer FT + Residual | 验证受控解冻是否进一步有效 |

其中 `Larger Residual MLP` 必须尽量匹配 Structured Residual 的 trainable parameter count：

\[
N_{trainable}^{LargeMLP}\approx N_{trainable}^{Structured}
\]

官方 Frozen HOPE 参数不计入这一公平性比较。建议同时记录 trainable parameters、总 inference parameters 和推理 latency/FLOPs。如果 Structured Residual 优于 trainable parameter count 匹配的普通 MLP，才可以认为提升主要来自输入结构，而非单纯增加容量。

蒸馏不放入第一版主线。只有当解冻官方 Actor 导致明显遗忘时，再考虑加入：

\[
L=L_{PPO}+\lambda_{KL}D_{KL}(\pi_{new}\|\pi_{HOPE})
\]

作为策略正则项。

### 9.1 开发、验证和最终评估隔离

模型选择不能依据最终汇报用的同一组评估场景反复调参。实验流程固定为：

```text
training seeds
      ↓
validation scenarios
      ↓
选择 Δq_max、网络宽度和 FiLM 配置
      ↓
held-out final evaluation scenarios
```

验证场景用于选择 bound、网络宽度和训练配置；最终评估场景必须与验证场景分离，并且在模型选择完成后不再参与调参。若当前评估器只有随机 seed 而没有显式 scenario split，也必须预先固定两组不重叠的 evaluation seeds，并在记录中标明用途。

### 9.2 Validation best-checkpoint 规则

所有模型使用相同的 training environment steps budget，并按固定间隔运行 validation。最终用于 held-out evaluation 的不是默认最后一个 checkpoint，而是在相同 budget 内按预先固定的字典序选择 validation 最优 checkpoint：

1. 第一排序：最大化 `strike_composite_success_exact`；
2. 第二排序：最小化稳定性风险，优先比较 `pre_strike_fall_rate`、`post_strike_fall_rate`，再最大化 `safe_recovery_rate`；
3. 第三排序：最小化 Residual magnitude；无 Residual 的 Direct FT 记为不适用。

排序规则、validation 间隔、候选 checkpoint 和最终选中的 checkpoint 必须写入实验 metadata，不能在看到 held-out 结果后改变。

### 9.3 Training budget contract

所有实验的训练预算定义为实验开始之后新增的 environment interactions：

\[
B_{train}=\text{additional environment steps after experiment start}
\]

比较 Direct FT、Residual、Structured 和 FiLM 时，必须使用相同的 `B_train`，不能用 checkpoint 的 absolute iteration 直接比较。Direct FT 即使恢复了 `model_21800.pt` 的 iteration metadata，也只从实验开始时重新计量新增 environment steps；Residual 等新实验从 iteration zero 开始，但预算定义相同。

Validation 也按新增 environment steps 定期执行：

\[
\text{evaluate every }K\text{ additional environment steps}
\]

`B_train`、`K`、rollout length、environment 数量和 validation 次数必须写入 experiment metadata。

## 10. 评估指标

### 10.1 主指标

- `strike_composite_success_exact`；
- `strike_pos_pass_exact`；
- `strike_vel_pass_exact`；
- `strike_normal_pass_exact`；
- 击球位置、速度和法向误差。

### 10.2 稳定性指标

- `swing_completion_rate`；
- `pre_strike_fall_rate`；
- `post_strike_fall_rate`；
- `safe_recovery_rate`；
- `rally_success_run_max`。

### 10.3 控制质量指标

- 动作绝对值和动作变化率；
- 关节力矩均值、最大值；
- 关节限位比例；
- Residual 的平均幅度和最大幅度；
- 手臂、躯干、腿部的分组 Residual 幅度；
- action saturation/clip 比例；
- 物理关节目标 Residual 幅度。

Residual 需要区分“名义修正”和“真正执行修正”，并同时记录 raw action mean 单位和物理关节目标单位。

名义 Residual 表示模型希望施加的修正：

\[
\|\Delta a_t\|_2,
\qquad
\frac{1}{31}\sum_i|\Delta a_{t,i}|
\]

\[
\Delta q^{nom}_t=action\_scale\odot\Delta\mu_t
\]

执行 Residual 必须通过官方 deterministic action transform 计算。记该变换为 `T(·)`；如果官方实现是 raw action clip，则 `T(a)=clip(a)`：

\[
\Delta q^{exec}_t=
action\_scale\odot
\left[T(\mu_{HOPE}+\Delta\mu)-T(\mu_{HOPE})\right]
\]

其中 `Δq_nom` 表示模型想修多少，`Δq_exec` 表示最终真正改变了多少关节目标。

动作饱和比例分为均值饱和和采样饱和：

\[
R_{\mu,sat}=\frac{\#(|\mu_{new}|>a_{limit})}{\#\mu}
\]

\[
R_{sample,sat}=\frac{\#(|a_{raw}|>a_{limit})}{\#a}
\]

如果官方实现不是 raw action clip，则使用其等价的官方 saturation event 定义。`R_{μ,sat}` 主要判断 Residual 是否把策略均值推到执行边界之外，`R_{sample,sat}` 还会受到探索噪声影响。

最终希望验证：击球性能提升的同时，Residual 仍然保持相对较小，说明模型是在官方技能上进行紧凑适配，而不是重新学习整套动作。

## 11. 阶段性验收门槛

### 阶段一：官方基线

- 权重结构和 110→31 接口检查通过；
- 官方评估流程可运行；
- PyTorch/ONNX 基线一致；
- 记录 checkpoint hash、配置、随机种子和指标。

### 阶段二：Residual MVP

- 零初始化时与 HOPE 输出数值等价；
- `μ_new=μ_HOPE` 且 `std/log_std_new=std/log_std_HOPE`，或对应官方分布参数完全一致；
- PPO distribution 和 log-prob 通过单步数值检查；
- 只训练 Residual 可以正常收敛；
- 训练初期没有明显成功率、摔倒率或动作幅度恶化。

### 阶段三：Structured Encoder 和 FiLM

- FiLM 的 identity initialization 能退化为同一 Structured Residual 结构；Structured Encoder 只要求其 Residual Head 零初始化时组合策略等价于 HOPE，不要求等价于已训练的 Plain Residual；
- 输入切片来自观测契约，不使用重复或新增输入；
- 消融实验使用相同训练和评估协议；
- 性能提升不能以明显稳定性恶化为代价。

### 阶段四：最终模型

- PyTorch 和 ONNX 的输入输出契约保持不变；
- 关节顺序、动作缩放和被动关节行为一致；
- 主指标相对 HOPE 基线有明确改善，或在相同性能下具有更小的 Residual/更好的稳定性；
- 结果可以由固定实验矩阵和记录文件复现。

### 11.1 完整等价性测试

Residual 初始化时必须对同一批 observation 检查完整链条，而不只比较 Actor mean：

```text
same observation
        ↓
official observation normalization
        ↓
HOPE mean / new mean
        ↓
official std or log_std
        ↓
official raw action distribution
        ↓
official action clipping/adaptation
        ↓
q_des
```

至少验证：

\[
\mu_{new}=\mu_{HOPE}
\]

\[
\text{official distribution parameters}_{new}
=
\text{official distribution parameters}_{HOPE}
\]

\[
q_{des,new}=q_{des,HOPE}
\]

ONNX 主要部署 deterministic Actor，因此 PyTorch/ONNX 的首要比较对象是 deterministic policy output；随机采样一致性按官方分布实现单独测试。ONNX 的 contract 不仅是维度不变，还必须保持输出语义不变：

```text
input semantic:  official hitter_pure observation after official preprocessing
output semantic: official deterministic actor output, raw_action[31]
```

新模型不得为了导出方便，把 `action_scale`、`q_default` 或 `q_des` 解码悄悄包进 ONNX；除非官方导出本身明确包含该步骤。当前导出仍由 [scripts/export_onnx.py](../scripts/export_onnx.py:5) 生成 `raw_action`，动作适配由独立的 ActionAdapter contract 负责。

### 11.2 Model metadata contract

每个新模型 checkpoint 和导出 manifest 至少记录以下字段：

```text
model_variant
observation_contract_version
observation_normalization
base_checkpoint_sha256
residual_bound_physical_rad
action_scale_source
resolved_action_scale_31d
residual_active_mask_31d
tau_scale
structured_split_indices
film_enabled
hope_frozen_layers
std_trainable
optimizer_restore_mode
hope_unfreeze_lr_scale
training_seed
validation_best_checkpoint
```

其中 `structured_split_indices` 必须由 `actor_observation_contract.py` 的 term layout 推导，禁止在多个文件中散落 `obs[:, :101]`、`obs[:, 101:109]`、`obs[:, 109]` 这类硬编码切片。当前官方导出 manifest 的 `observation_normalization` 值为 `none`；新模型如果改变该值，必须同步更新导出、部署和等价性测试。

`action_scale_source` 只记录来源路径，`resolved_action_scale_31d` 保存当时实际解析出的 31 个数值；`base_checkpoint_sha256` 和 `residual_active_mask_31d` 用于保证模型在配置文件后续变化后仍可复现。

## 12. 首个实现里程碑

第一版只实现以下内容，不提前加入 Structured Encoder、FiLM、GRU 或蒸馏：

```text
官方 HOPE Actor
        ↓
Frozen HOPE + Zero-init Residual Mean Actor
```

必须依次完成：

1. 正确加载官方 Actor；
2. Residual 输出零初始化；
3. 官方 observation preprocessing/normalization 语义确认，并保证 Residual 使用同一处理结果；
4. 新旧 Actor 输出等价性测试；
5. PPO distribution/log-prob 验证；
6. 验证 HOPE 与 Residual 使用同一 ActionManager 后，`actions` 观测仍是官方 `applied_raw_actions` 语义；
7. Residual-only 训练，使用模型权重加载但 optimizer state 重建；
8. ONNX `110 → 31` 导出、输出语义和一致性测试。

只有该里程碑全部通过后，才进入 Structured Encoder 和 FiLM 实现。

## 13. 主要代码边界

后续实现主要关注以下位置：

- `source/whole_body_tracking/whole_body_tracking/utils/bounded_actor_critic.py`：策略分布和 Actor-Critic 行为；
- `source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py`：训练器、checkpoint 和训练流程；
- `cfg/algo/ppo.yaml`：官方 PPO 基线配置；
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/actor_observation_contract.py`：观测契约和输入切片依据；
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_observations.py`：观测 preprocessing 和 `actions`/`applied_raw_actions` 语义；
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_actions.py`：31D action manager、被动关节处理和动作缓存；
- `scripts/export_onnx.py`：模型导出；
- `source/whole_body_tracking/whole_body_tracking/utils/exporter.py`：导出 metadata 和部署契约；
- `tests/`：模型契约、导出和数值等价性测试。

官方基线配置应保持可复现。新模型建议使用独立配置或独立模型字段，避免覆盖官方基线。

## 14. v1.4 Engineering Freeze

v1.4 之后不再扩展主算法路线。下一步只实施并验证：

```text
Official HOPE
      ↓
Frozen HOPE + Zero-init Bounded Residual Mean Actor
```

GRU、Transformer、Attention、蒸馏和其他新模型组件不进入当前实现周期。后续工作顺序固定为：源码与 checkpoint 核查 → Residual MVP 实现 → 等价性与 PPO contract 测试 → Residual-only 训练 → ONNX 语义验证。
