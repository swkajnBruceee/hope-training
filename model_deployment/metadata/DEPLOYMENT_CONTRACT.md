# 部署契约摘要

## model_900 上身策略

- 固定基座 backhand policy，作为联合协调器中的冻结 upper prior。
- actor 输入维度：56；actor 输出维度：10。
- 网络：`[512, 256, 128] + ELU`。
- 训练配置：256 environments，3000 iterations 运行；交付 checkpoint 为 `model_900`。
- 需要保持 12 帧 shoulder pitch/yaw lookahead，以及 ready-pose prelude release 逻辑。
- 不能把它当成官方 A3 `obs[1570] -> action[29]` 单体策略。

## model_3396 下身策略

- 浮动基座历史 Stage-A leg-support policy，作为联合协调器中的冻结 legacy lower prior。
- actor 输入维度：126；actor 输出维度：14。
- 网络：`[256, 128, 64] + ELU`。
- 训练 lineage：Fresh Stage-A → Return-C1 → Robust-B → Unified-K8 → K17 continuation，最终 checkpoint 为 `model_3396`。
- `.pt` 中带有 126-D actor observation normalizer；不能丢弃。
- 当前 F1 公开 Base14 契约只启用其中 12 个腿部通道，但这不等于可以改变历史 checkpoint 的输出维度或 observation 顺序。

## 重要限制

`model_3396` 的历史 root frame、observation 和 data contract 与当前 retraining contract 不同。它可以作为已经验证过的历史先验交付，但不能作为当前合同重新训练的 warm start。

真机部署还需要一个 wrapper，把真实机器人状态构造成相同 observation，应用相同 normalizer/action scaling/joint mapping，并按照 prelude、lookahead、限位和安全停机规则组合上下身输出。当前包只保证模型文件及其来源信息完整，不宣称已经完成真机安全验收。
