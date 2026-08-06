# A3 Base Locomotion Phase 0 审计

日期：2026-07-18  
状态：**有界 fixture 矩阵已通过；仅 Stand smoke 开放，长训/移动/部署关闭**  
主计划：[`A3_BASE_LOCOMOTION_MASTER_PLAN.md`](A3_BASE_LOCOMOTION_MASTER_PLAN.md)

## 已确认事实

### 2026-07-18 Stand smoke 更新

- `stand_fixture_gate_v1` 批准的 89 个逻辑 case 已在 Isaac/MuJoCo 各重复 3 次，534 次执行均在 fixture 安全包络内，同引擎重复 metrics/evidence 逐位一致。
- `A3BaseStand-v0` 已实现并通过真实 Isaac manager 实例化：14 维 action、925 维 Actor、970 维 Critic、200/50 Hz causal ZOH、完整 31 DOF 唯一目标。
- 确定性 reset/zero/scripted/random 审计无 nonfinite、非脚接触或硬限位错误；零动作首次 torso-tilt 重置在第 79 个 policy step，是未训练 plant 基线，不是站稳证据。
- 首次 64-env/100-iteration bounded PPO smoke 完成，但 `model_99` 仍不合格：无 10 s timeout episode，平均完成 episode `98.96` step，`53.0%` 有效动作达到 clip，主要因 waist-pitch hard-limit 终止。
- free-base waist working-point 扫描确认 `Kp=50` 时 waist pitch 会漂到约 `+0.419 rad`，而 `Kp=200--500` 可抑制该漂移；但增益单变量并未改善 Stand episode。
- 已完成 `waist pitch Kp/Kd = {50/2, 350/7}` × `raw clip = {0.25, 0.5}` 的 2×2、同 seed/同训练预算对照。四格确定性平均 episode 分别为 `98.96 / 97.49 / 74.61 / 101.78` step，全部仍为 `0` 个 10 s timeout。组合格仅有小幅交互收益，不足以批准 gain 或 action 合同。
- 零动作支撑审计显示初始 COM 近似支撑裕量约 `+9.8 cm`；先发生的是 torso 在低增益 waist pitch 处前折，约 `0.8 s` 到达上机械限位，随后 COM 前移并在 `1.56 s` 越出近似支撑区。因此默认 COM 一开始就在支撑区外不是当前主解释。
- 参数枚举到此停止。组合格仅保留为诊断候选；在 reward/termination return 对齐、right-ankle-pitch 目标/实际/力矩/载荷和接触几何审计前，不批准 500-iteration 延长 smoke。`stand_long_training_approved=false`、`locomotion_command_approved=false`、`deployment_approved=false` 保持不变。
- causal timeline 已完成。baseline 先发生 waist-pitch 偏离并最终越限，right ankle 没有触及 hard limit；组合候选则在约 `1.38 s` 越出近似支撑域，right-ankle target 从首帧偏离 nominal、随后正负摆动并触及 action clip，而 actual 在约 `2.0 s` 到 hard limit。终止时 target 仍在约 `0.175 rad`、actual 约 `0.524 rad`，施加力矩约 `-17--20 Nm`，符合 `Kp=50/Kd=2` 的误差响应而非 `118.2 Nm` effort saturation。
- 冻结组合候选的 policy waist-pitch residual 没有改善，平均完成长度由约 `102.7` 降为 `100.0` step；因此 policy 主动 waist-pitch residual 不是下游 ankle/height 失败的必要条件。高 waist-pitch `350/7` 配置下全零 14 维 action 也只维持 63 step，失败迁移到接近 `+0.349 rad` hard limit 的 waist roll。
- reward manager 的逐项 `rate * 0.02 s` 与环境返回 reward 在 `1e-8` 内一致，`base_height` 是目标高度平方误差惩罚，不会直接奖励下蹲。但当前没有显式 termination penalty；Isaac 自动 reset 后固定 PPO rollout 会立刻采新 episode，因此“单条 episode 累计 return 随时间增加”不足以证明策略偏好完整 10 s 生存。
- 接触语义已纠正：Isaac URDF importer 当前使用 `convex_hull`，不是动态刚体上的原始非凸三角 mesh。现状 zero-action 初始左右竖向载荷约 `2.7%/97.3%`；只替换两个 foot collision 的 conservative sole box 后为 `47.5%/52.5%`，但 zero-action 失败时间仍为 `79/78` step，组合 policy 平均完成长度也只由约 `102.7` 到 `103.7` step。接触不对称是真问题，但不是唯一根因。
- 路线因此增加 `static_working_point_qualification_v1`：在任何 reward v2 或 PPO 前，先冻结接触候选，并在可部署 normal-policy gains 下标定 nominal pose 与 bounded target offset。生产 `PD_STAND` 的高增益只能产生诊断参考，不能直接成为 Base policy 合同。

### 关节和接口顺序

以下三个来源的31个主动关节名称、顺序、轴、关节限位和effort limit已经由dependency-free validator交叉检查：

1. Agibot源URDF；
2. 为Isaac生成的`training/assets/agibot_a3/urdf/model.urdf`；
3. 官方`a3_pingpong.xml` MuJoCo模型。

唯一backend顺序为：

```text
waist 3 -> head 2 -> left arm 7 -> right arm 7 -> left leg 6 -> right leg 6
```

29 DOF policy view严格等于31 DOF顺序删除head两个slot，不是另一套“legs first”顺序。Base action严格固定为`left leg 6 -> right leg 6 -> waist roll -> waist pitch`。对应hash已经写入合同并由validator重算。

### 控制与物理频率

项目中存在四个不同频率，不能继续统称为“控制频率”：

| 层 | 当前值 | 来源/含义 |
|---|---:|---|
| Isaac physics | 200 Hz | `tracking_env_cfg.py`: `sim.dt=0.005` |
| Isaac/Base policy | 50 Hz | `decimation=4` |
| 官方MuJoCo physics | 1000 Hz | `a3_pingpong.xml`: `timestep=0.001` |
| A3 backend sync默认值 | 100 Hz | 默认`2 * policy_hz` |
| Policy runtime | 50 Hz | `a3_runtime_config.yaml` |

历史SIL日志中约500 Hz的raw state采样是传输/状态更新率，不是MuJoCo physics timestep。后续报告必须分别记录physics、raw state、backend sync和policy/command rate。

### 执行器增益不是同一套

当前Isaac隐式执行器和ONNX policy path使用`a3_kps/a3_kds`训练侧增益；`PD_STAND`使用明显更高的生产stand增益。例如leg hip pitch是policy `Kp=80`，而PD_STAND为`Kp=1500`。二者不能混用：

- Base训练和正常policy命令使用经Phase 0标定后的policy gain合同；
- PD_STAND只用于启动状态机；
- 不能用PD_STAND稳定性证明Base policy的sim-to-SIL一致性；
- takeover必须显式blend，并记录切换前后gain和q target。

### 接触模型仍是高风险项

当前Isaac URDF保留完整足部mesh作为 collision source，但 Isaac Lab 2.1 URDF converter 默认将动态 mesh 转成 `convex_hull`；生成 USD 已确认带 `convexHull` 标记。它不是原始三角 mesh 刚体接触。self-collision 因腕部/球拍重叠被关闭；prepare脚本只专门修补了球拍碰撞面。这适合 starter smoke，但还不足以证明高速横移接触质量。

在Base正式训练前至少需要：

- 足底mesh/接触patch可视化和静态压力分布；
- 摩擦扫描和左右脚滑移对称性；
- mesh collision与简化foot collision的对照；
- self-collision关闭对腰臂挥拍承载的风险说明；
- Isaac与官方MuJoCo相同站姿下的COM、足底高度和自由落体/落地响应。

自动化审计进一步确认：Isaac 的 collision source 是完整 `left/right_ankle_roll_Link.STL`，导入结果为各自 convex hull；官方 MuJoCo 使用 `collision_left/right_ankle_roll_hull`，两者不能因名称都含 hull 就视为同一几何。此外，当前 Isaac tracking nominal static/dynamic friction 为 `1.0/1.0`，旧任务随机化范围为 static `0.3–1.6`、dynamic `0.3–1.2`；官方 MuJoCo 地面 sliding friction 为 `1.5`。Base 任务不能直接继承旧范围后宣称两端一致，必须单独做 foot hull 和摩擦消融。

### 质量、COM与惯量审计结果

validator现在将URDF中通过fixed joint连接的link质量、COM和惯量合并到对应32个active body，并与官方MuJoCo body inertial比较。当前结果：

```text
URDF/Isaac total mass: 58.27723163 kg
official MuJoCo mass:  58.25727231 kg
difference:            0.01995932 kg
```

主要差异来自`imu_in_pelvis_Link`：URDF包含`0.02 kg`并固定到pelvis，官方MuJoCo虽然保留IMU site/visual geom，但`pelvis_link` inertial没有合并这`0.02 kg`。pelvis最大COM差约`0.354 mm`，惯量不变量最大相对差约`1.58%`。`torso_Link`还有约`0.009`的惯量不变量相对差，主要处于MuJoCo合并/截断后的量级，但在自动审计中仍保留为mismatch。

这不是立即导致跌倒的巨大质量误差，但它证明两端物理模型并非逐项完全相同。Phase 0必须明确选择：在Isaac中去掉pelvis IMU质量、在MuJoCo中补齐，或将其登记为有消融证据的允许偏差。在选择前保持训练门关闭。

### 低幅命令basis探针

在当前官方SIL保持`MotionControlAction_MOTION`时，执行了两次只读证据范围内的低幅接口探针；每次结束均发送三次零速度，输出保存在`/tmp`而非训练artifact：

| command | 1.5 s命令段横移 | 峰值横移速度 | 结论 |
|---:|---:|---:|---|
| `vy=+0.1 m/s` | `-0.000435 m` | `0.000409 m/s` | 响应低于可辨识范围 |
| `vy=-0.1 m/s` | `-0.000189 m` | `0.000270 m/s` | 响应低于可辨识范围 |

两侧均未产生足以判断符号的横移，可能涉及官方MOTION deadband、模式或高层命令接收语义。因此本次结果不能冻结Base command的`+vy`定义。新Base task中的basis scan必须使用策略自己的Command sampler和heading-frame位移测量重新完成；禁止用上述噪声级结果强行指定符号。

## 本批已实现

- 新增精确的A3左右腿、Base action、Strike v2和31/29 DOF常量；
- 新增Base、Strike、Composer、Actor、Critic、Action合同；
- 新增不依赖Isaac/NumPy/SDK的Python reference Composer；
- 新增两组31 DOF golden Composer vectors；
- validator检查合同hash、ownership、维度、镜像involution、golden vectors和三份机器人资产；
- validator默认报告资格状态；使用`--require-training-approved`时当前预期exit 2，阻止提前训练。
- 新增固定Phase 0标定协议和240次运行矩阵：80个逻辑case，每个重复3次，覆盖command basis、14关节低/中幅step、waist pitch叠加和200/1000 Hz下ZOH/linear transport。
- 标定artifact validator要求矩阵hash、完整case覆盖、分类指标和统一安全envelope；即使安全门通过也不会自动晋级，action scale仍需人工审查和版本化选择。
- 新增单case 31 DOF标定命令生成器；它只接受能被现有50 Hz RobotIO replay如实表达的关节step和waist residual ZOH case，明确拒绝尚无Base policy的command basis及必须在仿真器native substep执行的transport对照。
- 生成的NPZ带matrix/contract/joint-order/command hash及单发布者、隔离仿真和禁止真机标记；本批未对当前共享SIL执行直接关节命令。
- 新增只读同一 causal trace 的 MuJoCo 3.1.6 与 Isaac Sim 4.5 / Isaac Lab 2.1 native-substep runner。经等限位、固定根和接触下状态恢复三轮消融后，`single_joint_fixture_v1` 冻结为无地面接触的状态约束 fixture；原因和首个双端结果见[`A3_BASE_PHASE0_CROSS_SIM_FIXTURE.md`](A3_BASE_PHASE0_CROSS_SIM_FIXTURE.md)。旧 MuJoCo 单端 pilot 已标记为历史证据。
- 最终repeat-1低幅/waist/transport `46/46`和中幅step `28/28`通过fixture安全门；6组代表case的3次metrics hash逐位一致。这些结果不是free-base平衡证据，也不自动批准scale。

## 当前候选值及其含义

合同中的leg action scale来自现有`0.25 * effort / stiffness`规则；waist roll/pitch暂用现有保守Strike scale `0.12/0.14 rad`，pitch residual暂限`0.12 rad`。这些值只用于冻结Composer计算结构和golden vectors，不表示已经适合快速横移。

尤其是hip pitch/yaw的候选scale达到`0.6875 rad`。这可能对motion tracking合理，却可能对乒乓球短距离急停过大。必须进行单关节和成组step response、饱和率与落地冲击检查后再决定是否缩小。

左右镜像只对`intervention_active=false`的Locomotion采样有效。右手持拍挥拍Context不能简单镜像成左手动作，因此StrikeSupport不使用该镜像增强。

## Phase 0剩余硬阻塞项

1. 正负`vx/vy/yaw`基础扫描，冻结heading frame符号；
2. 首个 hip-roll 同 trace 双端 fixture 已通过，14个 Base action 的代表性双端 step response 和 Base Policy 下的 free-base 安全证据仍缺；
3. waist pitch residual 尚缺现行 causal trace v3 下的 working-point 双端扫描和 Isaac/StrikeSupport 对照；
4. 质量、惯量、COM和足底collision的Isaac/MuJoCo对照；
5. 50 Hz target 的 causal transport 已实现，尚缺代表性关节双端 ZOH/linear 证据，hold/interpolation 不冻结；
6. Base/Strike消息超时、hold和blend-to-safe时序；
7. C++ Composer对同一golden vectors的逐元素一致性；
8. Actor normalization重新统计方案。

当前标定矩阵semantic hash为`a3ce2b267eacffd98e98f4df86a947fc914df71918ad68cb117854df4b5f89e1`。矩阵内嵌五份完整合同payload hash；合同或case定义变化后旧结果不得复用。

以上未完成前，`training_approved=false`和`deployment_approved=false`必须保持不变，也不创建长期PPO run。

当前回归结果为`68 passed`；`--require-training-approved`资格门预期返回exit code 2，这是阻止候选标定值被提前用于长期训练的正常行为，不是测试失败。部署源码已确认 normal-policy `a3_kps/a3_kds` 经 `ExpandToBackend -> RobotCommand -> JointCommand stiffness/damping` 下发；这证明软件路径可表达训练增益，但尚未证明真机固件实际接受并执行这些逐关节增益，因此 deployment gain contract 仍未冻结。

## 执行命令

```bash
cd hope_training/whole_body_tracking
python3 tools/validate_a3_base_contract.py
python3 tools/build_a3_base_calibration_matrix.py --output /tmp/a3_base_phase0_matrix.json
python3 tools/build_a3_base_calibration_command.py \
  --matrix /tmp/a3_base_phase0_matrix.json \
  --case-id step__a0.10__left_hip_pitch_joint__pos__r01 \
  --output /tmp/a3_base_left_hip_pitch_step.npz
python3 -m pytest -q tests/test_a3_base_contract.py
```

资格门检查当前应失败并返回2：

```bash
python3 tools/validate_a3_base_contract.py --require-training-approved
```
