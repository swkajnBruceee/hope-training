# A3 Base Phase 0 跨仿真 Fixture v3

日期：2026-07-18  
状态：**Isaac/MuJoCo runner 与首个同 trace case 已通过；代表性矩阵未完成；Stand/PPO 未批准**

## 1. 冻结边界

本阶段只验证单关节控制语义、符号、候选量级和 50 Hz target transport。它不验证：

- 自由基座稳定性；
- 足底接触下的最终 action scale；
- 横移、刹停或速度命令；
- Strike Support 或真机部署。

两个 runner 都只读同一个 `a3_base_command_trace_v1` NPZ。200 Hz 物理步长为
`0.005 s`，50 Hz policy 命令每 4 个物理步更新。linear 模式只允许从上一 policy
target 因果插值到当前 target，四个 substep 的比例为 `1/4, 2/4, 3/4, 1`，不读取
未来命令。

当前矩阵 semantic hash：

```text
3a06767eb877918695a696568a33cf0be21bb2fe271188063a74340b915dd851
```

首个对照 trace：

```text
case_id:
step__a0.10__left_hip_roll_joint__pos__r01

trace_sha256:
abdd88a2bb9eaf0b2e7bd6a126783923ab51f660009421d0905ab7411d82c893
```

## 2. Fixture 语义为何调整

Isaac fixture 经过三种实现的受控消融。

| 版本 | 根/关节约束 | 地面 | 观察 | 决策 |
|---|---|---:|---|---|
| 探针 A | 根固定，非被测关节等上下限 | 开 | 非被测关节漂移 `0.0113 rad`；`0.08 rad` hip-roll 只响应 `0.00079 rad` | 淘汰；PhysX 等限位不是本任务所需的硬锁 |
| runner v1 | 根固定，非被测关节状态恢复 | 开 | hip-roll 基线漂到约 `0.112 rad` | 淘汰；固定根反力污染唯一自由关节 |
| runner v2 | 根与非被测关节都逐 substep 恢复 | 开 | hip-roll 仍在约 `0.17–0.22 rad` 强振荡 | 淘汰；状态恢复与足底接触求解产生夹具冲量 |
| runner v3 | 根与非被测关节都逐 substep 恢复 | 关 | 1 秒基线稳定；双端响应进入可比较范围 | 当前冻结版本 |

单变量无地面消融中，Isaac 的 hip-roll 在 200 个 baseline step 后为
`0.00559997 rad`；随后 `0.08 rad` target step 的末段增量约 `0.07186 rad`。
这证明此前大幅漂移主要是 fixture/contact 耦合伪影，不支持通过放大 action scale 处理。

v3 中碰撞几何仍随模型加载，但 Isaac 不生成地面且 self-collision 关闭；MuJoCo 显式
禁用 contact solver。Fixture reaction 继续标记为不可用：Isaac 会保存原始 incoming
joint force，但状态恢复不是物理约束，不能把该量冒充夹具反力。

## 3. 首个双端结果

两端均完成 `480/480` physics step，trace 未修改，无非有限值、触限、饱和或 safety stop。

| 指标 | Isaac v3 | MuJoCo v3 |
|---|---:|---:|
| commanded delta | 0.045833 rad | 0.045833 rad |
| steady response ratio | 0.9021 | 1.1276 |
| steady error | 0.00468 rad | 0.00816 rad |
| peak delta | 0.06433 rad | 0.05169 rad |
| selected effort RMS | 2.0504 Nm | 2.1174 Nm |
| selected peak torque | 5.5000 Nm | 5.2233 Nm |
| saturation duration | 0 s | 0 s |

配对比较结果：

```text
active delta trajectory RMSE       = 0.01572 rad
normalized RMSE / command delta    = 0.3429
steady response-ratio abs diff     = 0.2255
effort RMS relative diff           = 0.0316
peak torque relative diff          = 0.0503
```

响应方向一致，且力矩量级高度接近。当前将剩余过冲/响应比差异临时分类为：

```text
expected_integrator_difference
expected_actuator_difference
```

依据是 Isaac 使用 PhysX implicit drive，而 MuJoCo runner 使用手写 PD，并叠加 XML 中的
被动 damping/frictionloss。该分类是工程推断，只适用于此无接触 fixture；代表性关节扩展后
若出现同类指标显著恶化，必须改标为 `unexplained` 并阻塞 Stand。

## 4. 当前 Gate

```text
fixture_runner_implemented = true
first_cross_sim_pair_passed = true
representative_fixture_matrix_complete = false
final_action_scale_frozen = false
target_transport_frozen = false
stand_task_approved = false
locomotion_command_approved = false
deployment_approved = false
```

下一步只扩展代表性关节的正负低/中幅、waist working point 和 ZOH/linear 对照。
在这批对照完成前不跑全 240 case，不创建 Stand 环境长期 PPO，也不调整 waist residual limit。
