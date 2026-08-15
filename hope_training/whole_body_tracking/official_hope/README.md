# A3 全身跟踪训练与部署

这是一个独立的 A3 乒乓球全身跟踪训练链路，包含训练、评估、仿真回放和模型导出工具。
当前运行契约如下：

- actor observation：110 维
- actor action：31 维 Agibot A3 关节位置残差
- 控制频率：50 Hz
- 任务：`HOPE-HitterPingPong-AgibotA3-v0`
- PPO 网络：`[512, 256, 128]`
- 基线权重：`checkpoints/model_21800.pt`

## 1. 环境准备

需要 Isaac Sim / Isaac Lab、CUDA、PyTorch、`rsl_rl`、Hydra 等运行环境。在能运行
Isaac Lab 的 shell 中执行：

```bash
cd /home/bistu/桌面/HOPETableTennis/hope_training/whole_body_tracking/official_hope
source setup_train_env.sh
```

如果 Isaac Sim 不在默认位置，先设置：

```bash
export ISAAC_PYTHON=/path/to/isaacsim/python.sh
export ISAACLAB_ROOT=/path/to/IsaacLab   # 只有源码安装时需要
```

本地 A3 URDF 资产由项目已有的 `agibot/URDF/` 或 `a3_deploy/URDF/` 提供。首次运行可执行：

```bash
hope_isaac_py scripts/prepare_a3_isaac_asset.py --force
```

## 2. 先验证基线权重

先做不启动 Isaac 的权重检查：

```bash
python3 scripts/check_model_21800.py
```

再用小规模 viewer 验证任务注册、观测和 checkpoint 加载：

```bash
hope_isaac_py scripts/play.py task=HOPEPingPong algo=ppo num_envs=4 \
  checkpoint=checkpoints/model_21800.pt \
  motion_file=motions/preprocessed/hope_forehand.npz \
  motion_file_2=motions/preprocessed/hope_backhand.npz
```

首次正式复现时，使用原配置的环境数量和训练入口：

```bash
hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true \
  motion_file=motions/preprocessed/hope_forehand.npz \
  motion_file_2=motions/preprocessed/hope_backhand.npz
```

从 `model_21800.pt` 继续 PPO 训练：

```bash
hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true \
  checkpoint_path=checkpoints/model_21800.pt \
  motion_file=motions/preprocessed/hope_forehand.npz \
  motion_file_2=motions/preprocessed/hope_backhand.npz
```

`checkpoint_path` 是完整 resume，会加载 actor、critic、优化器和训练迭代状态；先用
上述 play 命令确认环境和权重正常，再做正式续训。新的 checkpoint 默认写入本快照的本地
训练日志目录。

## 3. 输出与后续部署

当前 actor 输出的是 31 维 raw joint-position residual，不是轨迹点。动作适配关系为：

```text
q_des = default_q + raw_action * action_scale
```

其中 head 的两个被动关节保持默认值，上一时刻实际应用动作也属于 observation。因而后续
如果接入自己的轨迹预测器，需要让预测器提供或转换成这套 110 维 actor observation；模型
本身可以直接作为“观测 -> 31 维关节目标残差”的策略模块使用。

导出 ONNX：

```bash
hope_isaac_py scripts/export_onnx.py --checkpoint checkpoints/model_21800.pt
```

不要直接把这个权重用于真实机器人；先完成仿真复现、关节顺序校验和限幅/安全检查。

## 4. 文件边界

`source/`、`cfg/`、`scripts/` 和 `motions/` 构成当前训练链路。项目中的其他 `training/`、
`model_deployment/` 和自定义任务保持独立。`model_21800.pt` 当前被项目 Git 规则
忽略，但已保存在本地快照中；其 SHA256 为：

```text
69ad47f206bb9da263102488b243bf3b750f09608078a354beee663c79f0fb6b
```

`scripts/check_model_21800.py` 是主机侧权重检查。`tests/` 中包含 ONNX、ROS 消息和部署
参考包测试；它们需要完整的 `a3_deploy`/`hope_ws`
部署树，不能替代 Isaac Lab 训练链路的验证。
