# A3 V1.3B 三模型真机部署包

本目录用于复现 `model_5000_student.pt` 在已审计 CompletePriors 条件下的动作效果。三份权重必须作为一个运行时组合使用：

```text
model_5000_student.pt  = 98D reference-free 学生策略，输出 26D
model_3396_lower_prior.pt = 下肢/支撑先验，历史 Stage-A，输出 14D（当前有效腿部通道 12D）
model_900_upper_prior.pt  = 上肢击球先验，固定基座，输出 10D
```

只加载 `model_5000_student.pt` 不能复现历史结果；它不包含运行时 3396/900 先验的输出。

## 1. 复现模式

要复现 model_5000 的历史审计结果，使用与训练一致的 `CompletePriors` runtime coordinator，不使用纯 reference-free 或 Precision Rescue 退火配置：

```text
lower prior alpha = 1.0（约）
upper prior alpha = 0.9
reference-free actor observation = 98D
student action = 26D
reset perturbation probability = 0
episode = one strike opportunity, up to 10 s
```

历史固定测试集上的参考结果（不是安全认证）：10 s survival 100%，位置误差约 4.55 cm，法向误差约 5.06°，速度误差约 1.25 m/s，combined success 约 14.8%。这些数值只有在三模型和相同坐标、时间、目标采样合同下才有可比性。

## 2. 每个模型的职责

### `model_3396_lower_prior.pt`

- 只作为下肢支撑/平衡的 additive prior，不是最终策略。
- checkpoint actor observation 126D，actor action 14D；当前 A3 合同实际使用其中 12 个腿部通道。
- 必须保留它自带的 observation normalizer、历史 observation 顺序和 joint mapping。
- 不要把它当作 V1.3B 的 warm-start 输入层，也不要把 14D 直接当作 A3 的 26D action。

### `model_900_upper_prior.pt`

- 作为上肢/腰部击球的冻结先验，输出 10D。
- 必须使用 `upper` 私有 observation contract、ready prelude、12-frame shoulder pitch/yaw lookahead 和 task-phase velocity feed-forward（beta=0.75）。
- 不要把它当作 V1.3B actor，也不要直接把原始 10D 拼到 26D。

### `model_5000_student.pt`

- CompletePriors 训练链中的 V1.3B 学生 actor checkpoint。
- actor observation 98D，critic 99D（真机只需要 actor）；action 26D。
- 98D 顺序：

```text
base_lin_vel(3)
base_ang_vel(3)
projected_gravity(3)
joint_pos(22)
joint_vel(22)
racket_pos(3)
racket_lin_vel(3)
racket_normal(3)
goal_10d(10)
previous_action(26)
```

- `goal_10d` 必须是 pelvis/root 的 base-yaw local frame，并使用训练合同中的固定 target-bank 归一化：`target racket position(3), target racket linear velocity(3), target racket normal(3), signed time-to-hit(1)`；不得改成球中心速度或世界坐标。
- 使用 checkpoint 中的 actor normalizer；不要重新拟合或丢弃 normalizer。

## 3. 运行时组合

概念组合为：

```text
q_lower = q_ready_lower
        + alpha_lower * (q_3396 - q_ready_lower)
        + lower_scale * student_action[0:12]
        + microstep_delta

q_upper = q_ready_upper
        + alpha_upper * upper_prior_target(model_900)
        + upper_scale * student_action[12:22]

student_action[22:26] = microstep command
```

实际部署不得自行重写上述 blending；应复用项目中的 A3 coordinator/action term，以确保 ready pose、frame、joint order、clipping、velocity feed-forward 和 safety gate 一致。

历史复现时 `alpha_lower≈1.0, alpha_upper=0.9`。当前 Precision Rescue 训练会从该状态继续退火，不能用来复现 model_5000 的原始审计数字。

## 4. 动作 scale

使用 `config/direct_action_scale_v13b_annealed_prior.yaml`，不要使用统一 scale，也不要硬编码旧的 0.24/0.55：

```text
lower (12): [0.192, 0.048, 0.192, 0.192, 0.144, 0.192,
             0.192, 0.048, 0.096, 0.072, 0.144, 0.192] rad
upper (10): [0.440, 0.022, 0.110, 0.440, 0.0132, 0.110,
             0.440, 0.440, 0.440, 0.440] rad
```

该 scale 已通过 PhysX dynamic envelope probe；当前 qualification 不覆盖 self-collision，真机仍需硬件限位、扭矩、速度、急停和碰撞保护。

## 5. 真机启动顺序

1. 加载并校验三个权重 SHA256。
2. 初始化机器人到项目定义的右脚前、左右宽站、屈膝 READY 姿态；不要用默认站姿覆盖 READY。
3. 启动 lower prior 和 upper prior，各自加载自己的 normalizer 与 observation contract。
4. 启动 student actor，加载其 actor normalizer；critic、optimizer 不部署。
5. 每个控制周期按同一 joint order 构造 98D observation，并更新 `previous_action`。
6. 将目标 10D 转换到当前 pelvis/root local frame；`signed_time_to_hit = t_hit - t`，击球后为负值。
7. 先运行 zero/small-action/nominal-target dry run，再开放击球目标。
8. 保持 deterministic inference；任何 alpha、scale、frame、joint index 不匹配都应 fail-safe 停机。

## 6. 严禁事项

- 不能只加载学生策略。
- 不能把 `model_3396` 或 `model_900` 的 observation 直接拼进 98D actor。
- 不能把 world-frame goal、ball-center goal 或 outgoing-ball velocity 直接喂给 student。
- 不能省略 normalizer、ready pose、lookahead、velocity feed-forward 或 previous action。
- 不能把本包当作真机安全认证；它是仿真合同和权重交付包。

详细字段见 `DEPLOYMENT_CONTRACT.yaml`，校验值见 `SHA256SUMS`，原始 probe 证据在 `evidence/`。

## 8. 目录内自包含推理工具

`runtime/` 现在包含不依赖 IsaacLab 的三模型 checkpoint loader 和
admission dry-run。它会加载三份 `.pt` 的 actor 与 observation normalizer，
并验证 126D/56D/98D 输入与 14D/10D/26D 输出：

```bash
python -m runtime.dry_run
```

现在还包含可直接对接 MuJoCo 的 `runtime/mujoco_adapter.py`，它负责：

- 从 A3 XML 按 31 个 canonical joint name 取 q/dq，并按 50 Hz policy tick 推进 MuJoCo；
- 构造 98D student、126D Stage-A、56D upper observation；
- 执行训练侧球拍挂点 FK、READY/prelude、model3396/model900 target 重建、microstep、关节/速率/扭矩限制和 `safe_halt()`。

合同 smoke：

```bash
PYTHONPATH=model_deployment/v13b_three_model_runtime_20260810 \
  python -m runtime.mujoco_contract_smoke
```

观测顺序、joint order 和 READY 数值分别见：

- `contracts/observation_contract.yaml`
- `contracts/joint_order.yaml`
- `contracts/ready_pose.yaml`

Isaac 对齐版 MuJoCo 植物位于：

```text
models/a3_v13b_isaac_compatible/mjcf/a3_v13b_isaac_compatible.xml
config/mujoco_isaac_compatible.yaml
```

它使用 Isaac tracking 的 `0.005s × 4 = 50Hz` 时序，关闭机器人自碰撞，
只保留脚底碰撞几何，并补入 Isaac A3 的 armature、Kp/Kd、摩擦和地面参数。
原始 MuJoCo XML 没有被覆盖。

这解决了“权重能否正确加载、normalizer 是否存在、三模型 shape 是否一致”
以及 MuJoCo 接线的问题。若要运行 model900 的真实击球先验，还必须提供
包内 `references/training_reference_bank_merged_20260807/training_manifest.json`
中的参考轨迹。现在可以用 `MotionManifestReferenceProvider` 按 manifest 选择并
懒加载真实 50Hz `.npz`，自动接入 50 步 READY prelude、8/12/16 步预览和三模型
先验；缺少该轨迹时只能使用 `ReadyHoldReference` 做接线/复位 smoke，适配器不会
伪造一条零轨迹。

注意：contract smoke 通过只代表状态、输入和物理步进有限，不等于所有目标都已
完成动态站立认证。当前包已经补入 MuJoCo 专用的
`isaac_passive_stable` 低层适配：它复制 source snapshot 的仿真站立增益，
在每个 `mj_step` 前做有界 COM/姿态支撑反馈，并将下肢目标限制在 READY 周围
0.12 rad 的站立安全包络。项目 candidate reference 0 的完整三模型链路在该
受限验证模式下已经跑满 250 个 50 Hz 控制步，根高约 1.04 m；这不是任意动作、
任意目标或真机稳定性的证明。`official_pd` 仍可用于对照，通常不能替代原生
MC 的平衡层。

低层配置见 `config/mujoco_isaac_compatible.yaml`，运行时可用：

```bash
PYTHONPATH=model_deployment/v13b_three_model_runtime_20260810 \
  python -m runtime.mujoco_contract_smoke \
  --manifest model_deployment/v13b_three_model_runtime_20260810/references/training_reference_bank_merged_20260807/training_manifest.json \
  --motion-index 0 --steps 250 \
  --low-level-profile isaac_passive_stable
```

真机接口由 `MujocoV13BAdapter.hardware_command()` 提供 canonical 31-DOF 的
`q_des/dq_des/kp/kd/tau_ff`。其中 `tau_ff` 默认置零，不能把 MuJoCo 的
`data.ctrl` 直接发给机器人；仍需沿用现有 SDK 的 31-DOF scatter、硬件限位和急停。

训练侧合同源文件的副本在 `source_snapshot/`，其中包含动作融合、私有
观测、A3 joint order 和 READY 姿态的原始实现，供 adapter 对照审计。
