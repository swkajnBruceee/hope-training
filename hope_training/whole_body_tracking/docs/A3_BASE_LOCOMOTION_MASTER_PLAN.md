# A3 Base Locomotion 主计划

状态：**当前开发主线（冻结）**  
冻结日期：2026-07-18  
适用范围：A3 下肢平衡、横移、挥拍承载、Base/Strike 合成、SIL 与真机门控  
变更规则：架构级修改必须先更新本文件；实现细节和暂定阈值可在阶段评审后版本化调整。

## 1. 最终决策

本项目后续按以下路线推进：

> 不更换现有 Isaac Lab + RSL-RL 训练栈，不迁移 G1/H1 等异构机器人权重；在现有 A3 工程内新增独立的 Base Locomotion 任务，参考 HugWBC 的速度控制、非对称 Actor-Critic 和上肢干预思想进行洁净重实现；参考 Decoupled WBC 定义上下身 ownership，由唯一 Command Composer 生成完整 31 DOF 命令。

官方 A3 MOTION 已验证横移速度能力不满足任务要求，因此：

- 不再作为正常运行时的平衡或横移控制器；
- 不再作为新 Base Policy 的性能上限或训练依赖；
- 仍可用于对照实验、启动/恢复流程研究和官方仿真模型校验，但这些用途不能重新取得正常控制 ownership；
- `PD_STAND`、`GET_UP` 等启动/恢复能力暂时保留为 Base 之外的状态机能力，不等同于使用 MOTION 执行击球补位。

异构开源策略的使用边界固定为：

- 复用当前 BeyondMimic 派生工程的环境、配置、RSL-RL、导出和评估基础设施；
- 从 HugWBC 论文和公开实现中吸取任务分解、奖励和干预课程思想，不复制无明确许可证的代码；
- 从 GR00T Decoupled WBC 吸取 ownership、接口和部署编排思想，不把其权重作为依赖；
- 第一版不引入 SONIC 的全身参考编码，也不加载现有 SONIC-like ONNX 作为站立策略。

这不是“从零重写强化学习栈”。自研范围只包括 A3 不可直接复用的部分：机器人 MDP、动作合成、观测合同、执行器/接触模型、课程、部署 Composer 和安全门。

## 2. 对输入意见的批判性结论

### 2.1 直接采纳

以下建议与当前工程和真机接口一致，直接作为主线要求：

1. 将 `vx/vy/yaw_rate/body_height/body_pitch` 定义为 Command，将挥拍时序和上肢参考定义为 Context。
2. Base 始终在线，Strike 只发送任务参考，不直接接管腿部。
3. Base Policy 从随机权重开始训练，重新统计 normalization，使用 A3 专项 action scale 和随机化范围。
4. Actor 只使用可部署信息；仿真真值线速度、足底接触力等只进入 privileged critic。
5. 速度、命令变化率、上肢干预和 Domain Randomization 使用相互独立的课程门。
6. 站立、挥拍承载、完整移动使用共享 MDP 和同一策略合同的三个环境配置，不训练三套最终策略。
7. 训练基础设施复用，PPO 核心和 exporter 第一阶段不修改。

### 2.2 必须修正

原建议中有七处不能直接照搬：

1. **14 DOF action 不是 14 DOF 最终命令。** 当前机器人后端接收完整 31 DOF 命令，仿真也必须使用同样的合成语义。因此需要自定义复合 Action Term，在每个控制周期把 Base 的 14 维输出、Strike/干预参考和固定关节基线合成为唯一 31 DOF 目标。
2. **当前 Actor 不能加入完美 foot contact。** `RobotIOBackend` 当前状态合同没有可靠足底力；第一版 Actor 不使用接触传感器，接触力只给 Critic 和 Reward。
3. **当前 Actor 也不能直接使用仿真真值 base linear velocity。** 现有 body-drive 同步状态提供双 IMU 和关节状态，但尚无经过合同验证的真机线速度估计。第一版依靠时序 proprioception；以后若 mocap/估计器通过验证，必须升级 observation schema。
4. **三个任务必须保持完全相同的 Actor、Critic、Action 维度和顺序。** Stand 中未使用的移动命令、Strike Context 必须置零并带有效性掩码，不能删掉字段，否则 checkpoint 不能安全续训。
5. **future reference 必须有 validity mask。** Reference dropout 后的全零向量不能与真实零姿态混淆。
6. **`feet air time` 不能默认成为主奖励。** 乒乓球横移可能更接近短步或 shuffle，过早强化腾空时间容易诱发跳步。Stand 阶段关闭；只有步态数据证明必要时才在 Locomotion 阶段引入。
7. **部署 Composer 不能只写 Python。** 当前生产路径是 C++ `RobotIOBackend`。Python 版本只作为合同参考和 golden-vector 生成器，生产合成与安全检查必须有 C++ 实现并做逐元素一致性测试。

### 2.3 暂不采纳

- 不把 `±0.8 m/s` 当作修改一个上限就能达到的结果；它是待实验验证的任务目标。先验收 20/30 cm 补位时间、启动、刹停、反向和挥拍后稳定时间。
- 不在第一阶段修改 PPO 加入专用 symmetry loss。先做左右镜像命令采样、镜像评估和数据均衡；只有出现持续左右性能差异时再引入算法级对称增强。
- 不把 HugWBC 奖励权重原样复制到 A3。仅复用奖励结构，权重由 A3 资产、执行器响应和分阶段消融决定。
- 不把现有 `model_step_098000_a3.onnx` 当作站立初始化。当前 SIL 已证明它适合作为 motion-tracking 影子推理检查，但直接接管后快速触发倾角安全停机，不能作为 Base 起点。

## 3. 当前项目基线与已知风险

### 3.1 可直接复用的基础设施

现有代码的真实位置是 `hope_training/whole_body_tracking/`，而不是意见中的仓库根 `tasks/`：

- 训练任务：`training/tasks/`
- A3 资产和执行器配置：`training/robots/agibot_a3.py`
- Hydra 任务配置：`cfg/task/`
- PPO 配置：`cfg/algo/ppo.yaml` 和各任务 `agents/ppo.py`
- 训练入口：`scripts/train.py`
- 合同：`contracts/`
- 生产后端：仓库根 `a3_deploy_example/` 与官方参考树 `agibot/code_deployment/a3_deploy_example/`

当前 `tracking` 任务本质上是全身 motion tracking，不应改名或硬改成 Base Locomotion。新任务必须作为 `training/tasks/base_locomotion/` 的平级任务存在，避免继续携带全身 Motion Command、真值线速度 Actor 观测和全身动作 ownership。

### 3.2 必须先消除的模型风险

在大规模 PPO 前完成 A3 模型审计：

- Isaac URDF 与官方 MuJoCo/AimRT 的关节轴、顺序、限位、默认站姿和坐标符号；
- 腿部和腰部质量、惯量、COM、足底碰撞几何；
- 当前 actuator stiffness/damping/effort/velocity 配置与官方执行器响应；
- 200 Hz Isaac physics、50 Hz policy 与官方约 500 Hz 状态/伺服链的延迟和保持方式；
- 当前为避免腕部/球拍网格重叠而关闭 self-collision 的影响；
- 足底摩擦、恢复系数和接触稳定性；
- pelvis/torso 两套 IMU 的轴、符号、重力投影和时间对齐。

官方 MuJoCo/AimRT 在此阶段是 sim-to-sim 和部署合同的验证权威，但官方 MOTION 不是正常控制器。

## 4. 冻结的控制架构

```text
Ball prediction / Strike planner
             │
             ├── Strike reference + future reference + timing context
             ├── Base motion command
             ▼
         Command Composer  ← Base Policy 14-D action
             │
             ├── ownership / limits / rate limits / safety
             ▼
       one complete 31-DOF RobotCommand
             │
             ▼
         RobotIOBackend / A3
```

任何时刻只能有一个最终 31 DOF 发布者。Base、Strike、启动状态机和安全层不能各自向 body-drive 发布部分关节命令。

### 4.1 Command 与 Context

Base 必须跟踪的 Command：

```text
vx                  # heading frame 前向速度
vy                  # heading frame 横向速度；正方向在 Phase 0 基础扫描后冻结
yaw_rate            # 绕竖直轴角速度
body_height         # pelvis 目标高度
body_pitch_reference
```

Base 用于预判扰动的 Context：

```text
intervention_active
strike_phase_sin
strike_phase_cos
time_to_hit
future_upper_body_q/dq
future_reference_valid_mask
```

当前上肢 `q/dq` 已包含在全身 proprioception 中，不在 Context 中重复一份。`strike_phase` 和 `time_to_hit` 没有跟踪 Reward。

Command 坐标系固定为 pelvis-yaw 对齐的 heading frame，单位为 SI。实际 `x/y` 正方向必须通过 Isaac 和官方 MuJoCo 的正负基向量测试后写入合同，不能凭名称推断。

### 4.2 第一版 14 维 Base Action

Actor 输出顺序固定为：

```text
left_leg:  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
right_leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
waist_roll_correction
waist_pitch_correction
```

动作采用非积分式位置目标：

```text
q_leg_cmd = q_leg_nominal + scale_by_joint_group * action_leg
q_waist_roll_cmd = q_waist_roll_nominal + scale_roll * action_roll
q_waist_pitch_cmd = q_waist_pitch_strike
                    + clip(scale_pitch * action_pitch, -pitch_residual_limit,
                           +pitch_residual_limit)
```

禁止把 action 累加到上一帧目标。每个关节组分别设置 scale，并由 Phase 0 的关节限位、步响应、最大安全速度和稳定性测试确定；在这些测试前不冻结具体数值。

### 4.3 31 DOF ownership v1

| 关节组 | 正常运行主控制方 | 合成规则 |
|---|---|---|
| 双腿 12 | Base | nominal + Base action |
| waist roll | Base | nominal + Base action |
| waist pitch | Strike + Base | Strike reference + 有界 Base residual |
| waist yaw | Strike | Strike reference，经限位和速率限制 |
| 右臂 7 | Strike | Strike reference，经限位和速率限制 |
| 左臂 7 | Composer baseline | v1 固定安全基线；扩展必须升级合同 |
| 头部 2 | gaze/baseline | 不进入 v1 policy view，仍由唯一 Composer 汇总 |

这会改变当前 Native Strike 的腰部合同。现有 Strike 数据、目标、命令证据和评估结果全部保留，但集成前必须发布 `strike_policy_contract_v2`：Strike 不再输出 waist roll 的最终命令，只输出 waist yaw、waist pitch 和右臂 7 关节参考。旧 10 DOF Strike 输出不能与新 Composer 静默混用。

### 4.4 仿真与部署必须同构

新增 `A3BaseCompositeActionTerm`：

1. 接收 Actor 的 14 维动作；
2. 从 `UpperBodyInterventionCommand` 读取当前上肢参考；
3. 应用与部署相同的 ownership、scale、clip、rate limit 和 baseline；
4. 每个 policy tick 形成 31 DOF 目标；
5. 在仿真子步内使用明确定义的保持或插值；
6. 输出 Composer debug fields，便于逐关节审计。

Python 参考 Composer、Isaac Action Term 和 C++ 部署 Composer 必须共享同一份合同数据，并通过 golden vectors 验证。不能维护三套手工常量。

### 4.5 Base 返回给 Strike 的状态

协同不是单向发送速度。Composer/runtime 每个 policy tick 向 Strike/planner 提供版本化 `BaseStatus`：

```text
timestamp / state_age
command_tracking_error_vx_vy_yaw
torso_roll_pitch / angular_velocity
pelvis_height_error
base_action_saturation_ratio
composer_limit_hit_mask
intervention_context_valid
readiness_state          # warming_up / ready / degraded / safety_stop
```

第一版不伪造动力学意义不明确的单一 `balance_margin`。上层先用可测量的姿态、跟踪误差、饱和率和状态新鲜度决定是否推迟、降速或取消挥拍；以后若实现并验证 ZMP/捕获点或接触估计器，再通过 BaseStatus schema 升级加入。

## 5. Policy IO v1

### 5.1 Actor observation

第一版只使用真机当前可获得或可稳定构造的信息：

- pelvis IMU：角速度、projected gravity；
- torso IMU：角速度、projected gravity；
- 29 DOF policy view 的 `q - q_nominal` 和 `dq`，头部 2 DOF 排除；
- previous Base action 14；
- Command 5；
- Strike Context；
- proprioception 历史窗口。

明确排除：仿真真值 base linear velocity、完美 contact state/force、world-frame pose、仅仿真可见的外力和物理参数。项目已有 mocap base pose，但它尚未进入当前 body-drive policy 的同步与可用性合同，因此不能作为 Actor v1 的隐含依赖。

### 5.2 历史窗口和归一化

第一版采用 50 Hz policy rate 和 10 帧 proprioception 历史，与当前部署已有的时序缓存能力对齐。每帧候选字段为：

```text
pelvis angular velocity       3
pelvis projected gravity      3
torso angular velocity        3
torso projected gravity       3
29-DOF q relative            29
29-DOF dq                    29
previous Base action         14
                              --
per-frame                    84
```

Command 和 Context 只追加当前值，不重复进每个历史帧。按 9 个 Strike 参考关节、4 个 horizon、q+dq 计算，v1 Actor 候选总维度为：

```text
10 * 84 proprio history      840
Command                        5
Context                       80
                              ---
candidate total              925
```

Phase 0 的 schema generator 必须从字段定义计算维度并生成 JSON，禁止手写 `925` 到网络配置而不校验。历史 reset 时用当前实测 proprioception 填满状态槽、用零填 previous action；禁止用全零历史制造部署中不存在的启动瞬态。

所有连续量的 scale、clip 和 normalization 必须写入 schema。Normalization 只从 A3 当前训练分布重新统计，不能继承现有 tracking/SONIC 模型。

### 5.3 Future upper-body reference

第一版 horizon 固定为：

```text
t, t + 0.10 s, t + 0.20 s, t + 0.30 s
```

在 50 Hz 下对应 offset `[0, 5, 10, 15]` policy ticks。每个点提供以下 9 个参考关节的 q 和 dq：

```text
waist_yaw, waist_pitch,
right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw,
right_elbow,
right_wrist_roll, right_wrist_pitch, right_wrist_yaw
```

同时提供 4 维 horizon validity mask。`intervention_active=false` 时 phase/time/reference 采用合同规定的 neutral encoding；future packet 超时或缺失时置 invalid，Base 继续保持稳定，不能复用陈旧挥拍参考。

每个 future packet 还必须携带 `sequence_id`、源时间戳、reference `dt`、Strike contract ID 和 joint-order hash。Composer 以状态时间对齐，不允许用“收到消息的墙钟时间”代替轨迹时间。

当前 Strike runtime 只产生即时控制，不具备上述 future packet，因此 future context 是 Phase 3 必须补齐的 Strike v2 接口，不是假定已经存在的能力。只有相对“无 future context”消融有稳定收益时才保留 q+dq 全量编码；否则优先删减 dq 或 horizon，而不是扩大网络。

### 5.4 Privileged critic observation

Critic 在 Actor observation 基础上可以增加：

- 真值 base linear velocity；
- 双足接触力、接触状态和足底滑移速度；
- 摩擦系数、质量/COM 偏移、执行器强度；
- 注入的外力/力矩；
- action/communication delay 与控制频率随机量；
- body/foot world-frame dynamics。

Stand、StrikeSupport、Locomotion 三个配置必须保持相同 Actor/critic schema。允许数值分布不同，不允许字段数量或顺序不同。

### 5.5 合同元数据

每个训练、导出和部署 artifact 必须记录：

```text
base_policy_contract_id
strike_policy_contract_id
command_composer_contract_id
joint_order_sha256
actor_observation_schema_version
critic_observation_schema_version
action_schema_version
policy_rate_hz
future_reference_offsets_s
action_scale_sha256
actuator_gain_sha256
normalization_sha256
robot_asset_sha256
onnx_sha256
git_commit
```

缺任一必要字段的 checkpoint/ONNX 只能用于诊断，不能进入 SIL 或硬件候选。

## 6. 仿真和部署必须共用的 Composer 语义

### 6.1 仿真

新增 `A3BaseCompositeActionTerm`：

1. 接收 14 维 Base action；
2. 从 `UpperBodyInterventionCommand` 取得当前上肢参考和 future context；
3. 按 ownership、限幅、速度/加速度限制和 baseline 合成 31 DOF q target；
4. 每个 control tick 为所有 31 DOF 写入明确目标；
5. 输出合成诊断：每关节 owner、pre/post-clip 值、limit hit、context validity。

Stand 阶段也必须走同一 CompositeActionTerm，只是上肢参考为 baseline。禁止为了先跑起来而绕过 Composer，否则 Stand checkpoint 的动作语义不能继承到 StrikeSupport。

### 6.2 部署

部署 C++ Composer 运行在 `RobotIOBackend` 之前，并满足：

- 一个进程/模块拥有最终 `RobotCommand`；
- 输入超时后按合同退化到 hold/blend-to-safe，而不是继续外推挥拍；
- Base/Strike schema、joint order、policy rate 或 normalization 不匹配时拒绝启动；
- takeover 前填满 observation history，并从当前实测 q 平滑 blend 到 policy target；
- safety stop、关节限位、倾角/角速度/通信超时优先级高于策略输出；
- 记录输入、合成前后命令、限幅和实际状态，支持完全复盘。

Python reference composer 与 C++ composer 使用同一组 JSON golden vectors。31 DOF q/dq/tau_ff/kp/kd 必须逐元素一致，浮点容差写入测试，不靠人工比对。

## 7. 新任务和代码落点

实际目录遵循当前仓库结构，不采用脱离现状的顶层 `tasks/` 示例：

```text
hope_training/whole_body_tracking/
  training/tasks/base_locomotion/
    __init__.py
    base_env_cfg.py
    mdp/
      __init__.py
      actions.py
      commands.py
      observations.py
      rewards.py
      events.py
      terminations.py
      curricula.py
    config/agibot_a3/
      __init__.py
      stand_env_cfg.py
      strike_support_env_cfg.py
      locomotion_env_cfg.py
      agents/ppo.py
  cfg/task/
    A3BaseStand.yaml
    A3BaseStrikeSupport.yaml
    A3BaseLocomotion.yaml
  contracts/a3_base_locomotion_v1/
    base_policy_contract.json
    strike_policy_contract.json
    command_composer_contract.json
    actor_observation_schema.json
    critic_observation_schema.json
    action_schema.json
    golden_composer_vectors.json
  tools/
    validate_a3_base_contract.py
    compose_a3_command_reference.py
```

部署实现遵循当前“项目侧扩展注入 vendor build”的做法：`agibot/code_deployment/` 保持 Agibot vendor/reference 不改；项目侧新增 C++ Base runtime/Composer，通过构建注入或独立 target 接入官方 `RobotIOBackend`。若使用根目录 `a3_deploy_example/` 作为集成工作树，必须在实现前写明它与 vendor 副本的单向关系，禁止两处手工同步修改。

同时补充以下基础常量，而不是在各任务重复正则：

- `A3_LEFT_LEG_JOINTS`、`A3_RIGHT_LEG_JOINTS`、`A3_BASE_ACTION_JOINTS`；
- 31 DOF command order、29 DOF policy order、Base 14 DOF order；
- mirror map/sign、nominal pose、group action scales 和 joint limit margins。

`scripts/train.py` 当前包含面向现有 strike/motion 的硬编码 override 映射。新任务接入时应新增 Base 专用的 typed config adapter，且 `UpperBodyInterventionCommand` 为可选数据提供者；不能因为有上肢轨迹文件就误选 `MotionOnPolicyRunner`。

## 8. 三个环境配置

### 8.1 `A3BaseStand-v0`

- 速度和 yaw command 为零；body height/pitch 先固定 nominal；
- 上肢从 baseline 逐渐过渡到小幅随机 intervention；
- 训练站立、姿态、恢复和窄范围外力抗扰；
- context 字段保留，未使用部分置零且 mask=false。

### 8.2 `A3BaseStrikeSupport-v0`

- 零速或低速 command；
- 播放可追溯的 FH/BH 上肢参考；
- 启用 strike phase、time-to-hit、future reference、dropout 和 perturbation；
- 目标是挥拍期间不跌倒、抑制躯干失稳并在击球后恢复，不奖励 Base 跟踪手臂。

### 8.3 `A3BaseLocomotion-v0`

- 完整 vx/vy/yaw/body command；
- 训练启动、停止、反向和 20/30 cm 任务化横移；
- 从无上肢动作逐步加入真实 FH/BH；
- 最终 checkpoint 来源只有这一配置，但必须从通过前级门的 checkpoint 继续，而不是从零重开。

三个配置注册时同时增加一个 schema-equality 测试；任何配置变更造成 observation/action contract hash 不同都应让测试失败。

## 9. Reward 和 termination 设计

### 9.1 第一版 reward 组

主要任务：

- heading-frame base velocity tracking；
- yaw-rate tracking；
- body-height 和 body-pitch tracking；
- upright/projected-gravity；
- 在 intervention 后回到稳定姿态。

接触与步态：

- foot slip penalty；
- undesired contact penalty；
- foot clearance 下限/碰撞约束；
- stance width regularization；
- 左右对称命令分布和镜像评估。

动力学与安全：

- joint-limit、torque、power；
- joint acceleration、action rate；
- waist residual magnitude/rate；
- policy command 超出执行器安全 envelope 的惩罚。

击球承载：

- survive intervention；
- torso roll/pitch 和 angular velocity suppression；
- post-strike recovery time；
- 足底滑移与失稳余量。

Reward 权重从结构上参考公开工作，但数值由 A3 reward-component statistics 决定。所有权重进入版本化配置；禁止直接复制 H1 权重。

### 9.2 第一版 termination

- 非允许 body contact；
- pelvis/torso 高度低于阈值；
- roll/pitch 超过安全阈值；
- 非有限状态/动作；
- 关键关节越界；
- episode timeout。

训练 termination、SIL safety stop 和硬件 safety stop 是三套不同严格度的门，必须分别记录。训练阈值不能直接当硬件安全阈值。

## 10. 四条独立课程

### 10.1 速度课程

```text
stand
-> |vy| <= 0.15
-> 0.30
-> 0.45
-> 0.60
-> 评估性探索 0.80 m/s
```

vx 和 yaw 在横移稳定后再打开较小范围；每级同时评估速度 RMSE、横移完成时间、刹停距离、到位稳定时间和力矩余量。

### 10.2 命令变化率课程

```text
常值/缓慢变化 -> 正常启动 -> 急停 -> 反向 -> 随机短脉冲
```

命令采样器要直接生成位移任务和可控 slew rate，不只生成长时间恒定速度。乒乓球晋级指标以 20/30 cm 补位和停止为主。

### 10.3 上肢干预课程

```text
baseline
-> 小幅随机关节动作
-> 慢速 FH/BH
-> 原速 FH/BH
-> 时序/幅度/单关节扰动
-> 横移与挥拍部分重叠
```

Reference dropout 随机缩短 horizon 或置 invalid；reference perturbation 覆盖起始时刻、播放速度、幅度和有限关节偏差。扰动范围必须保留真实轨迹包络，不能生成机器人不可能执行的上肢命令。

#### 10.3.1 按挥拍相位的动态短窗口课程

`A3BaseStrikeSupport-v0` 正式采用 phase-conditioned reset + overlapping short-window curriculum，但它只在 deterministic Stand checkpoint 通过后开启，不替代 Stand 本身。

每次 reset 必须是动态状态，不是将某帧冻结成静态姿势：

```text
phase phi
upper-body q_ref(phi), dq_ref(phi)
root pose/velocity and leg/waist q,dq with declared provenance
previous effective Base action and observation history
future q/dq reference [0, 0.1, 0.2, 0.3] s + validity mask
```

上肢干预播放器必须在 Isaac 中同时传递 position target 和 velocity target。只设 `q_ref` 且默认 `dq_target=0` 会在高速相位 reset 后制造虚假制动冲击，这类结果不得当作 Base 抗扰证据。

窗口起始相位连续随机采样，禁止使用互不重叠的离散硬分块。候选课程是：

```text
0.20 s -> 0.35 s -> 0.50 s -> 0.75 s -> full strike + recovery
```

每个窗口带随机、不向 policy 显式暴露结束时刻的 `0.10–0.20 s` recovery tail，防止 policy 在窗口末帧透支稳定性。从第一个短窗口阶段起就保留长/完整 rollout，候选混合为 `70% short / 20% medium / 10% full`，比例只根据 held-out 相位成功率调整。

相位采样使用“均匀底座 + 失败加权 + 完整起点”，不允许完全追逐最差 bin：

```text
50% uniform phase
40% bounded failure-weighted phase
10% full-action start
```

第一版允许用挥拍上肢 `q/dq` + 统一 Stand 下肢的人工状态做小规模启动试验，但这些状态必须标记 `synthetic_reset`，不得单独用于晋级。策略能跑通长窗口后建立 `phase_reset_bank`，从成功 rollout 保存 root/leg/waist/upper-body 动态状态、history 和 previous action，后期训练优先从这些可达状态采样。

每个 phase bin 独立记录 survival、tilt、height、foot slip、hard/soft limit、action saturation 和 recovery。只有所有 bin 通过、关键击球 bin 通过更高门槛，且完整 rollout 没有累积失稳，才能增加窗口长度。

### 10.4 Domain randomization 课程

```text
窄执行器/摩擦
-> 质量与 COM
-> action/state delay 与频率抖动
-> 地面摩擦
-> 外力/外力矩
-> 经消融验证的联合随机化
```

每次只提升一条主课程，其余保持在已通过级别。晋级依据评估集，不依据训练 reward 均值。

## 11. 分阶段实施和硬验收门

下面的性能数字是 **v0 工程目标**，Phase 0 标定后可以通过一次有记录的计划修订调整；安全阈值不允许用训练结果倒推放宽。

### Phase 0：资产、执行器和合同冻结

交付：

- Isaac A3 与官方 MuJoCo 的 joint order/axis/limit、nominal pose、足底碰撞、质量/惯量、kp/kd/effort 和延迟对照报告；
- 31/29/14 DOF 常量、mirror map 和正负 basis scan；
- Base/Strike/Composer v1 JSON contract 及 hash；
- 50 Hz policy、状态抽样、hold/interpolation 和超时语义；
- vendor tree 与项目侧部署扩展的边界说明。

退出门：合同 validator、joint mapping、mirror involution、14->31 composer golden tests 全部通过。未通过不得启动正式 PPO。

### Phase 1：任务骨架和确定性 Stand

交付：

- `base_locomotion` MDP、三个同 schema config 和注册；
- CompositeActionTerm 与 Python reference composer；
- 无随机化、无干预的 `A3BaseStand-v0`；
- reward-component、action saturation、contact 和 torque 诊断面板。

退出门：不同 seed 的评估中，1000 个 10 s episode 生存率目标 >=99%；无非法接触，非有限值为零，tilt p95 目标 <5 deg，关节/动作持续饱和不能被 reward 掩盖。

#### Phase 1a：静态工作点资格门（2026-07-18 审计后新增）

任何延长 PPO 前先完成 `static_working_point_qualification_v1`。它不要求 normal-policy gains 下全零 action 自然稳定 10 s，但必须证明存在可达、可部署且不靠持续顶限的静态控制工作点：

```text
versioned contact candidate
+ normal-policy kp/kd/effort contract
+ nominal pose
+ bounded 14-DOF target offset
-> zero/scripted deterministic rollout
```

候选工作点必须同时满足：左右初始载荷无未解释的大偏置、COM 在声明的几何支撑域内、腰/踝 target 与 actual 不向 hard limit 迁移、所需 PD effort 有余量、所有 offset 落在候选 action scale/clip 可表达范围。原始接触资产上的后续消融已推翻“必须使用 normal-policy gains”的早期假设：当前仿真 plant 需要全部 14 个 Base-owned 关节使用 `PD_STAND` 候选增益才能被动稳定。因此 Recovery 仿真任务允许使用这组增益，但在真机后端被证明确实执行命令中的 Kp/Kd 及同一组 effort/velocity limit 之前，它不是部署合同。

同一门内增加 reward-v2 离线资格：显式 failure/termination term 必须按 Isaac `weight * term * policy_dt` 语义设计，并用固定 horizon + auto-reset 重算证明“主动快速 reset”不优于维持可恢复站立。只比较 completed episode return 不构成通过证据。

下一次 Stand smoke 的 PPO 候选保持网络和优化器不变，只将 `init_noise_std` 从 `1.0` 降为 `0.15`。原因是 RSL-RL 保存未裁剪 Gaussian sample 及其 log-prob，而环境执行前才裁剪到 `[-0.25,0.25]`；零均值、单位标准差初始化时单维理论越界概率约 `80.3%`。低噪声候选必须记录 sampled-action、actor-mean 和 effective-action 三种 clip fraction，不能把确定性 actor mean 顶限误归因于探索噪声。晋级参考为 sampled/effective clip fraction `<10%`，稳定后目标 `<5%`。

#### Phase 1b：Stand Recovery-A（2026-07-19）

零命令静态 PPO 已停止。当前通过的仿真 plant 是 `Base14 PD_STAND + passive nominal`；RL 只输出有界、非积分的 recovery residual，健康状态下必须优先零输出。

Recovery-A 第一版混合 `35%` 无扰动 episode 和 `65%` reset 扰动 episode。受扰 reset 的 root roll/pitch 范围为 `+/-0.035 rad`，roll/pitch 角速度范围为 `+/-0.20 rad/s`；扰动 mask 不进入 Actor，只用于 reward 掩码和审计分组。Reward-v3 保留 survival/termination 语义，全局惩罚 pre-clip action，在无扰 slice 附加更强的 action 惩罚，并仅对受扰 slice 奖励 reset-safe 的倾角误差下降。Progress term 返回误差下降率，避免 Isaac `weight * term * policy_dt` 再次重复缩放时间步。

首次 256 环境零 residual 审计中，无扰和受扰两组都完成了 10 秒。使用临时严格恢复包络（`tilt <= 0.01 rad`、root roll/pitch 角速度 `<= 0.05 rad/s`、连续 10 个 policy step）时，受扰组恢复率为 `85.33%`，恢复时间中位数/均值为 `2.16 s / 3.01 s`。这证明恢复时间存在可学空间，不代表学习策略成功。严格包络尚未用 clean steady-state tail 标定，因此扰动合同和恢复包络仍不批准。Reward-v3 的掩码、reset、clean 零动作优先性和 progress 解析尺度已通过运行审计。

已生成版本化的 paired disturbance trace：clean/candidate/medium/upper 各 1024 条，每个受扰 profile 的姿态和角速度四个符号象限各 256 条。Passive 与 Policy 评估必须复用同一 `trace_index`，不允许只声明相同 seed。标定器只有在恰好 500 个 policy step 时才能产生 `calibration_measured=true`；短运行必须显式使用 `--runtime-smoke`，且永不能晋级。Clean tail 将记录 pelvis roll/pitch、root 线/角速度、高度误差与腰踝速度的 p90/p95/p99，包络候选采用 p99 + margin、dwell 与 hysteresis，但分析器不得自动批准训练。

Runner 构建后的完整链审计已通过：925 维原始观测经 empirical normalization 后 Actor mean 仍严格为 `0`，14 个关节的初始 std 均为 `0.15`，20 万组 Gaussian 样本的总 clip fraction 为 `9.508%`。这只通过 zero-mean 初始化语义，不代表未训练随机策略对 plant 安全。成对安全审计器已实现 Passive/Random 同 trace 对照，必须完成 500 step 才能设置 `untrained_stochastic_policy_safety_verified=true`。

第一次有界 Recovery PPO smoke 前仍必须完成：无并发仿真器负载时的三档 500-step 扰动强度对照、基于 clean tail 的恢复包络选择、未训练随机策略的完整安全回归，以及版本化的 iteration/environment 预算。在此之前 `bounded_recovery_smoke_approved=false`。

### Phase 2：鲁棒 Stand

交付：小幅随机手臂动作、窄执行器/摩擦随机化、push/torque 扰动和恢复评估。

退出门：维持 Phase 1 生存目标；推扰后在预设时间窗内恢复高度/倾角，且无明显左右偏置。必须提供无 DR、单项 DR、联合 DR 消融。

### Phase 3：真实挥拍承载

交付：

- 由现有可追溯 FH/BH 数据生成 Strike v2 的 9 关节参考；
- future reference `[0, 0.1, 0.2, 0.3] s`、mask、dropout、时序/幅度 perturbation；
- `A3BaseStrikeSupport-v0` 课程和独立 held-out FH/BH 集。
- 按挥拍相位的动态 reset、连续重叠短窗口、隐藏 recovery tail 和始终保留的长/完整 rollout 混合；
- phase-bin 独立成功率与有界失败加权采样，以及从成功长 rollout 建立的 `phase_reset_bank`。

退出门：原速 FH/BH 和扰动版本上 10 s 生存率目标 >=99%；击球后约 0.5 s 内回到计划规定的 tilt/angular-rate envelope；相对“无 future context”基线有可复现收益。若 future context 无收益，应删减，而不是保留无效复杂度。

### Phase 4：低速横移和快速停止

交付：`|vy| <= 0.30 m/s`、20/30 cm 位移命令、急停和左右反向；保持上肢 baseline。

退出门：速度跟踪、位移时间、超调、停止漂移、settling time 和生存率同时达标。v0 参考目标为 `vy RMSE <=0.10 m/s`、生存率 >=99%；位移/刹停阈值由 Phase 0 的 A3 动力学标定后冻结。

### Phase 5：中高速横移

交付：依次晋级到 `0.45`、`0.60 m/s`，最后才探索 `0.80 m/s`；加入少量 vx/yaw 和更强 slew/reversal。

退出门：每一级都必须改善真实补位时间而不破坏停止稳定性和力矩/功率余量。若 0.8 只提高峰值速度却增加总补位时间或无法稳定击球，则冻结在更低的可用上限。

### Phase 6：移动中挥拍

交付：横移、减速、挥拍和恢复的重叠课程；Strike 上层能够读取 Base stability margin 和 command tracking error。

退出门：held-out 球路/动作时序上，移动 + FH/BH 的生存、接触前稳定、击球后恢复均通过；Base 不得依靠固定记忆某一条轨迹。

### Phase 7：ONNX、C++ Composer 和官方 SIL

交付：

- 带完整 metadata/hash 的 ONNX；
- 新 Base observation history builder；
- C++ Composer、single publisher、warm-up/blend/timeout/safety；
- Python/C++ golden parity；
- 官方 MuJoCo 中的 shadow -> gated takeover -> 延时/扰动回放。

退出门：合同完全匹配、50 Hz 推理 deadline 无丢失、31 DOF 命令一致、历史预热正确；多次 SIL takeover 无 safety stop，并复现 Isaac 的站立/横移/挥拍趋势。SIL 失败时回到相应训练/模型阶段，不通过调大安全阈值放行。

### Phase 8：硬件门控

顺序固定为：悬挂/支撑低增益检查 -> 短时原地 Stand -> 小扰动 -> 低速短横移 -> 急停 -> 慢速挥拍 -> 原速挥拍。每一步都需要 E-stop、通信超时、姿态门、关节/力矩门和 safe halt 证据。

未经独立评审，不进行高速横移中挥拍。硬件失败 artifact 必须保留，不得覆盖成功日志。

### Phase 9：可选联合微调

只有 Base、Strike 和 Composer 各自通过 SIL 后才允许：冻结大部分 Base、用小学习率解冻末层或添加有界 residual，并约束其偏离已验证 Base。必须保留原 Base checkpoint 和一键回退。此阶段不是 v1 上线前置依赖。

## 12. 测试与实验治理

第一批自动测试至少包含：

- 精确 joint order、hash 和 31/29/14 映射；
- mirror map 两次应用回到原值；
- Composer ownership 无冲突、所有 31 DOF 均被赋值；
- waist roll/pitch clip 和 Strike/Base 合成规则；
- future reference horizon、mask、dropout 和过期语义；
- 三个环境 actor/critic/action schema hash 相同；
- observation history 初始化、reset 和时间顺序；
- Actor 不含 privileged contact/velocity 的静态检查；
- ONNX input/output 名称、维度和 metadata；
- Python/C++ Composer golden parity；
- single command publisher 和超时退化的 SIL 检查。

每个实验 manifest 必须记录 git SHA、seed、资产/合同 hash、课程级别、DR 范围、reward 配置、normalization 和 checkpoint hash。候选状态只允许按以下方向晋级：

```text
diagnostic -> candidate -> isaac_qualified -> sil_qualified -> hardware_candidate
```

不允许把失败/诊断 artifact 通过复制或重命名晋级。所有阶段均保留固定 seed 回归集和 held-out 轨迹集，训练 reward 不能代替验收指标。

## 13. 后续开发顺序

从本计划生效起，提交按以下顺序推进：

1. Phase 0 资产/执行器审计和 v1 合同；
2. CompositeActionTerm、reference composer 与合同测试；
3. 三个同 schema 环境骨架；
4. deterministic Stand；
5. robust Stand；
6. StrikeSupport；
7. 低速横移、急停、反向；
8. 中高速横移；
9. 移动中挥拍；
10. ONNX/C++ Composer/SIL；
11. 硬件门控；
12. 可选联合微调。

第一批实现不得提前改 PPO、引入 SONIC、迁移他机权重、直接编辑 vendor deployment tree，或把官方 MOTION 重新放回正常控制 ownership。任何需要改变 Command/Context、14 DOF Action、腰部 ownership、Actor 可部署性或 single-publisher 原则的修改，都必须先更新本文件和合同版本，再写代码。

当前执行状态：**Phase 0 有界 fixture 门已通过；Phase 1a 已得到“被动站立 plant 通过、学习策略未通过”的分裂结论；Phase 1b Recovery-A 环境和 reward 语义已实现，但 PPO 仍关闭。** 普通增益即使使用完整 schema 内的静态预载仍在 `107` step 由 base height 失败。原始接触资产上的逐组增益消融显示：只提高 waist pitch、整个 waist、或 waist+ankles 均不能稳定；仅当全部 14 个 Base-owned 关节使用 PD_STAND 候选增益时，零动作才能完成 `500 step / 10 s`，而上肢无需切换高增益。三轮零命令 PPO 的 raw-action clip 分别为 `77.87% / 75.93% / 51.00%`，所有 checkpoint 仍被拒绝。Recovery-A 现已拥有独立任务/runner 命名空间、零 actor-mean 初始化、隐藏的 reset 扰动 mask、clean slice do-no-harm reward、仅对受扰环境生效的 recovery shaping，以及持续越界安全包络。完整的 256 环境被动 baseline、reward-v3 运行审计、固定 paired trace 和完整 runner 零均值审计已写入 `stand_recovery_a_gate_v1.json`。因三档扰动合同、clean-tail 恢复包络和未训练随机策略安全回归尚未完成，gate 继续 fail-closed。`bounded_recovery_smoke_approved=false`、`stand_recovery_policy_approved=false`、`stand_long_training_approved=false`、`locomotion_command_approved=false`、`deployment_approved=false`；真实后端是否实际执行这组 Kp/Kd 仍是部署前硬门。

## 14. 开源参考边界

- 当前 BeyondMimic/whole-body-tracking 工程继续作为基础设施底座，遵守仓库已有许可证和 NOTICE。
- HugWBC 仅用于理解论文中的 velocity/body/gait command、upper-body intervention 和训练课程；采用 clean-room 式重新实现，不复制无明确许可代码。
- GR00T WholeBodyControl 仅参考 Decoupled WBC 的上下身编排和部署思想；代码和模型权重许可分别审查，v1 不依赖其权重。
- SONIC 只保留为以后 future-motion encoding、全身协调和部署结构的研究参考，不进入 Base v1 关键路径。

参考：

- BeyondMimic / whole_body_tracking: https://github.com/HybridRobotics/whole_body_tracking
- HugWBC paper: https://arxiv.org/abs/2502.03206
- HugWBC repository: https://github.com/InternRobotics/HugWBC
- GR00T WholeBodyControl: https://github.com/NVlabs/GR00T-WholeBodyControl
