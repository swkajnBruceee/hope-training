# A3 全身跟踪训练

本目录提供 A3 乒乓球全身跟踪任务的训练、评估、动作预处理和模型导出工具。
更完整的运行说明位于 `official_hope/README.md`。

## 快速开始

在已配置 Isaac Sim / Isaac Lab 的环境中：

```bash
cd hope_training/whole_body_tracking
source setup_train_env.sh
hope_isaac_py scripts/train.py task=HOPEPingPong algo=ppo headless=true \
  motion_file=motions/preprocessed/hope_forehand.npz \
  motion_file_2=motions/preprocessed/hope_backhand.npz
```

回放或评估：

```bash
hope_isaac_py scripts/play.py task=HOPEPingPong algo=ppo num_envs=4 \
  checkpoint=checkpoints/model_21800.pt
```

## 目录约定

- `official_hope/`：完整的训练、评估、仿真回放和导出链路。
- `training/`：当前项目的训练包及任务实现。
- `cfg/`：环境和算法配置。
- `scripts/`：训练、动作预处理、检查与导出脚本。
- `motions/`：本地动作数据及预处理结果。

训练任务使用 Hydra 配置，默认任务名为 `HOPEPingPong`，策略观测为 110 维，动作
为 31 维 A3 关节位置残差，控制频率为 50 Hz。真实硬件运行前必须完成仿真、关节顺序、
限幅和急停检查。
