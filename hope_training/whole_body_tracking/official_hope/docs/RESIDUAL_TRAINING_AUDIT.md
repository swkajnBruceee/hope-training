# Residual MVP 训练审计说明

## 当前结论

Residual MVP 训练具备两类审计：

1. HOPE 原有的环境、动作、击球和 q_des 审计；
2. 本项目新增的 Residual policy 审计。

两类审计都只记录指标，不改变 reward、环境状态或 PPO loss。

## 已迁移并接入训练 runner 的审计

以下内容来自官方 HOPE 训练工程，并已在当前项目中保留或迁移：

- `actor_observation_contract.py`：110D `hitter_pure` observation layout 校验；
- `hope_actions.py`：`applied_raw_actions`、passive action mask、q_des/action safety instrumentation；
- `hope_commands.py`：strike position/base error、strike pass/fall、q_des joint instrumentation；
- `one_step_contract.py`、`qdes_contract.py`：动作和 q_des contract；
- `success_metric.py`：独立评估阶段的 `success_rate` 定义；
- `MotionOnPolicyRunner._log_live_metrics()`：每个 PPO iteration 写入 command、reward、termination、action、q_des 和 policy 统计；
- checkpoint resume state：训练审计累积量随 checkpoint 保存和恢复。

训练日志中的主要前缀包括：

```text
Live/Reward/*
Live/Termination/*
Live/Action/*
Live/Policy/*
Live/<command_term>/*
Instrumentation/<command_term>/*
Instrumentation/qdes_safety/*
```

## 本次新增的 Residual 审计

新增位置：

```text
source/whole_body_tracking/whole_body_tracking/utils/residual_actor_critic.py
source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py
```

每轮 PPO 日志写入：

```text
Live/ResidualProbe/residual_mean_l2_active
Live/ResidualProbe/residual_mean_abs_active
Live/ResidualProbe/residual_mean_max_abs_active
Live/ResidualProbe/residual_q_nom_abs_active
Live/ResidualProbe/residual_q_nom_max_abs_active
Live/ResidualProbe/residual_q_raw_clip_estimate_abs_active
Live/ResidualProbe/residual_mean_saturation_rate_active
```

这些指标只统计 29 个 active action dimensions，不把 head 被动关节计入平均值。

- `residual_mean_*`：raw action policy mean 的 Residual 修正；
- `residual_q_nom_*`：乘以实际 `action_scale` 后的名义物理关节角修正；
- `residual_q_raw_clip_estimate_*`：考虑 raw action `[-1, 1]` clipping 后的确定性执行修正估计；
- `residual_mean_saturation_rate_active`：组合 policy mean 超出 raw action 边界的比例。

精确的 sampled action 执行结果仍以 ActionManager 的 safety instrumentation 为准，因为环境可能继续执行 affine decoding、safe clamp 和 passive joint handling。

## checkpoint 中保存的 Residual metadata

Residual checkpoint 额外保存 `model_metadata`，包括：

- observation contract 和 layout；
- normalization 语义；
- `residual_delta_q_max_rad`、`residual_time_scale`；
- 31D 实际 action scale；
- 31D active mask 和 raw residual bound；
- official warm-start checkpoint SHA256；
- std 是否可训练。

## 尚未迁移的官方脚本

`/home/bistu/桌面/HOPE` 中 planner、ROS 和 ball-physics falsification 审计脚本不属于 Actor PPO 训练链路，因此没有伪装成训练内审计接入。它们应在 planner、MuJoCo 或部署验证阶段单独运行。

当前项目已有的 `scripts/evaluate.py` 和 `scripts/mujoco_eval_onnx.py` 用于训练后的策略评估，不会在每个 PPO iteration 内自动执行。

## 查看方式

TensorBoard 日志：

```bash
tensorboard --logdir logs/rsl_rl
```

`ResidualProbe` 是更新后策略对本轮 rollout 最后一个 observation batch 的 probe，不应解释为本轮 sampled action 的精确平均执行量。训练结束后，优先同时检查：

```text
Live/ResidualProbe/*
Instrumentation/qdes_safety/*
Live/Termination/*
```

如果 Residual 名义修正明显增加但 `residual_q_clip_exec_abs_active` 很小，通常说明动作边界饱和；如果 q_des safety fault 或 termination rate 上升，应停止继续扩大 Residual bound。
