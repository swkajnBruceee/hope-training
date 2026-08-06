# HOPE 乒乓球强化学习开发流程

本文档用于指导后续把当前 HOPE/Isaac Lab 场景逐步开发成“机器人能打乒乓球”的训练任务。

当前状态不是完整乒乓智能体，而是已经具备：

- Isaac Sim / IsaacLab 运行环境。
- Agibot A3 + 桌台 + 球 + 网的场景。
- 球的重力、碰撞、反弹和空气阻力。
- PPO 训练入口和基础 RL 框架。
- 一个可验证的 `--fix_base` 场景 smoke run。

因此后续重点不是继续装环境，而是逐步补齐任务、奖励函数、动作数据和训练流程。

## 总体原则

不要一开始训练“完整人形机器人打乒乓球”。这个任务同时包含平衡、移动、挥拍、击球、落点控制、连续回合，难度太高。

推荐路线是：

```text
固定底座击球 -> 固定底座定向回球 -> 站稳 -> 站稳击球 -> 移动到位 -> 连续回合
```

每个阶段都要有明确验收标准。只有当前阶段能稳定复现，再进入下一阶段。

## 阶段 0：场景和物理验证

目标：确认环境、场景、球物理、机器人资产都能正常加载。

已验证命令：

```bash
cd ~/桌面/HOPETableTennis/hope_training/whole_body_tracking
source setup_train_env.sh
hope_isaac_py scripts/play_table_tennis.py --fix_base --steps 300
```

验收标准：

- Isaac Sim GUI 能启动。
- `/World/envs/env_0/Robot` 创建成功。
- `/World/envs/env_0/Ball` 创建成功。
- `serve_ball` reset event 生效。
- 球能被发出、飞行、反弹。
- 300 步后正常退出。

当前结论：阶段 0 已完成。

相关文件：

- `hope_training/whole_body_tracking/scripts/play_table_tennis.py`
- `hope_training/whole_body_tracking/training/tasks/table_tennis/table_tennis_env_cfg.py`
- `hope_training/whole_body_tracking/training/tasks/table_tennis/table_tennis_env.py`
- `hope_training/whole_body_tracking/training/tasks/table_tennis/ball.py`
- `hope_training/whole_body_tracking/training/tasks/table_tennis/geometry.py`

## 阶段 1：固定底座击球任务

目标：先不解决站稳，只训练上半身或手臂把球碰到。

这是最重要的起点。因为如果同时训练站稳和击球，策略很容易在两个难题之间互相干扰，训练信号也很难分析。

建议创建一个最小任务：

```text
HOPE-TableTennis-AgibotA3-HitFixedBase-v0
```

任务设定：

- 机器人 base 固定。
- 来球由 `serve_ball` 事件生成。
- action 先控制全身 31 个关节，后续可收窄到上半身/手臂。
- observation 包含机器人关节状态、球相对机器人位置、球速度。
- reward 先只鼓励手/球拍接近球和发生有效接触。

建议 reward 从简单到复杂：

```text
手/球拍靠近球 -> 击中球 -> 击球后球向 +X 方向飞 -> 球过网
```

第一版 reward 可以包含：

- `racket_ball_distance`：手/球拍离球越近奖励越高。
- `hit_contact`：球速度突然变化或球与手/球拍接近时给奖励。
- `return_velocity_x`：击球后球朝对方半台方向飞。
- `action_rate`：动作变化过大惩罚。
- `joint_limit`：接近关节极限惩罚。

验收标准：

- PPO 能跑起来并生成 checkpoint。
- TensorBoard 中总 reward 不再完全随机。
- 可视化回放时，手臂会朝球运动。
- 10 次来球中至少能碰到 1 到 3 次。

主要修改位置：

- `tasks/table_tennis/mdp/rewards.py`
- `tasks/table_tennis/mdp/observations.py`
- `tasks/table_tennis/mdp/terminations.py`
- `tasks/table_tennis/table_tennis_env_cfg.py`
- `tasks/table_tennis/config/agibot_a3/table_tennis_env_cfg.py`
- `tasks/table_tennis/__init__.py`

开发建议：

- 先不要追求球落台。
- 先不要训练走路。
- 先不要引入复杂 motion imitation。
- 先把 reward 做到能稳定产生“靠近球”和“碰球”的行为。

## 阶段 2：固定底座定向回球

目标：在固定底座下，让机器人不只是碰到球，而是把球打向目标区域。

任务设定：

- base 仍然固定。
- 球从对方半台发来。
- 击球后希望球朝 +X 方向越过网。
- 进一步希望球落到对方半台。

新增 reward：

- `ball_over_net`：球过网且高度合理。
- `landing_on_opponent_table`：球落在对方半台区域。
- `return_direction`：球速度方向朝目标区域。
- `return_speed`：球速在合理范围内，太慢或太快都不理想。

新增 termination：

- 球出界。
- 球落地。
- 击球后成功落入目标区。
- episode 超时。

验收标准：

- 策略能稳定碰球。
- 击球后球大多数情况下向对面飞。
- 少量回合能过网。
- 后续目标是提高落台率。

相关模块：

- `geometry.py`：目标区域、桌台坐标、网位置。
- `events.py`：来球初始位置和速度分布。
- `rewards.py`：回球方向、过网、落台奖励。
- `terminations.py`：成功/失败终止。

## 阶段 3：站稳任务

目标：让 A3 在不固定 base 的情况下保持平衡。

这一步可以单独训练，不要同时训练复杂击球。

任务设定：

- base 不固定。
- 不发球，或球不参与 reward。
- action 控制全身关节。
- reward 关注身体稳定性。

建议 reward：

- `alive`：持续存活。
- `base_height`：骨盆高度接近期望值。
- `upright`：身体姿态保持竖直。
- `feet_contact`：脚部接触稳定。
- `joint_vel_penalty`：关节速度过大惩罚。
- `action_rate`：动作变化过大惩罚。

termination：

- pelvis 高度过低。
- 身体倾角过大。
- episode 超时。

验收标准：

- 机器人能站立数秒不倒。
- 关节动作不过度抖动。
- 不依赖 `--fix_base` 也能保持基本姿态。

说明：

当前 `--fix_base` 只是为了验证场景和球物理，不代表真正策略会站稳。要进入真实乒乓任务，站稳能力必须补上。

## 阶段 4：站稳击球

目标：把阶段 2 的击球能力和阶段 3 的站稳能力合并。

这一步是难度明显上升的阶段。建议使用 curriculum：

```text
固定 base 击球 -> base 加小扰动 -> 不固定 base 但来球慢 -> 正常来球
```

训练策略：

- 可以从固定底座击球 checkpoint 初始化。
- 可以从站稳 checkpoint 初始化。
- 也可以先用 imitation motion 做先验，再用 PPO fine-tune。

新增 reward：

- 保留站稳 reward。
- 保留击球 reward。
- 增加上半身动作平滑。
- 增加脚底稳定或重心稳定。

验收标准：

- 机器人在来球时不立刻摔倒。
- 能做出朝球方向的上肢动作。
- 少量回合能碰到球。

## 阶段 5：移动到位

目标：让机器人根据来球位置调整身体或脚步。

这一步不应该太早做。只有站稳和基本击球都可用后，再引入移动。

任务设定：

- 来球位置分布扩大。
- 机器人需要调整站位或重心。
- observation 需要包含更明确的球预测信息，例如球未来到达击球平面的估计位置。

建议新增内容：

- `target_base_position` 或隐式目标站位。
- 脚步移动 reward。
- 不摔倒约束。
- 击球点预测辅助 observation。

验收标准：

- 机器人能朝合理方向移动或调整身体。
- 移动不会显著破坏站稳。
- 击球成功率高于固定站位时。

## 阶段 6：连续回合

目标：从单次回球扩展到多次来回。

这是完整乒乓任务，不建议在前面阶段未完成时启动。

需要补充：

- 对手或发球机策略。
- 更严格的比赛规则。
- 回合状态机。
- 连续击球奖励。
- 落点、速度、旋转控制。

验收标准：

- 能连续回球 2 次以上。
- 能把球稳定打到对方半台。
- 策略不是靠偶然碰撞获得奖励。

## 推荐的近期开发顺序

短期只做前三件事：

1. 建立 `HitFixedBase` 训练任务。
2. 加入“手/球拍靠近球”和“击中球”的 reward。
3. 用 PPO 跑小规模训练并可视化 checkpoint。

不要同时做：

- 完整比赛规则。
- 复杂步态。
- 大规模训练。
- 多球连续回合。
- 真实机器人部署。

## 建议命令

场景验证：

```bash
cd ~/桌面/HOPETableTennis/hope_training/whole_body_tracking
source setup_train_env.sh
hope_isaac_py scripts/play_table_tennis.py --fix_base --steps 300
```

headless smoke run：

```bash
hope_isaac_py scripts/play_table_tennis.py --headless --steps 300
```

当前已有 PPO smoke run 示例：

```bash
hope_isaac_py scripts/train.py task=TrackingFlat algo=ppo headless=true \
  logger=tensorboard \
  motion_file=sample_motions/agibot_a3_smoke_stand.npz \
  num_envs=32 max_iterations=10
```

后续创建 `HitFixedBase` 任务后，训练命令应变成类似：

```bash
hope_isaac_py scripts/train.py task=HitFixedBase algo=ppo headless=true \
  logger=tensorboard \
  num_envs=32 max_iterations=100
```

具体 `task=HitFixedBase` 的 Hydra 配置需要后续新增。

## 每次开发的检查清单

每做一个阶段，都按下面顺序检查：

1. 场景能启动。
2. observation shape 正确。
3. action shape 正确。
4. reward 每一项数值范围合理。
5. termination 不会过早触发。
6. PPO 能生成 checkpoint。
7. TensorBoard 曲线有可解释趋势。
8. 回放行为和 reward 设计一致。

如果策略行为异常，优先排查：

- reward 是否符号写反。
- reward 是否尺度过大或过小。
- 球和手/球拍坐标系是否一致。
- reset 分布是否太难。
- action scale 是否太大。
- episode 是否太短。

## 当前代码结构速查

场景和物理：

- `tasks/table_tennis/table_tennis_env_cfg.py`
- `tasks/table_tennis/table_tennis_env.py`
- `tasks/table_tennis/geometry.py`
- `tasks/table_tennis/ball.py`

MDP 组件：

- `tasks/table_tennis/mdp/observations.py`
- `tasks/table_tennis/mdp/rewards.py`
- `tasks/table_tennis/mdp/events.py`
- `tasks/table_tennis/mdp/terminations.py`

机器人配置：

- `tasks/table_tennis/config/agibot_a3/table_tennis_env_cfg.py`
- `robots/agibot_a3.py`

训练入口：

- `scripts/train.py`
- `scripts/play_table_tennis.py`
- `cfg/task/*.yaml`
- `cfg/algo/ppo.yaml`

## 阶段完成定义

阶段 0 完成：场景和球物理可视化正常。

阶段 1 完成：固定底座下策略能主动挥向球并有一定碰球率。

阶段 2 完成：固定底座下策略能把球朝对面打，部分回合能过网或落台。

阶段 3 完成：不固定底座时机器人能稳定站立。

阶段 4 完成：不固定底座时机器人能站稳并完成少量击球。

阶段 5 完成：机器人能根据来球位置调整站位。

阶段 6 完成：机器人能完成连续回合。
