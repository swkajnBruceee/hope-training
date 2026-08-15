# 来源与版本

## 学生策略

- 文件：`model_5000_student.pt`
- 来源运行：
  `agibot_a3_target_conditioned_reference_free_v13b_complete_priors_rightfront_v1/2026-08-09_18-10-06_v13b_resetfixed_model18900_clean_23118_rightfront_16384x50000_resume_from2300_exact`
- checkpoint：`model_5000.pt`
- 训练动作库：23,118 条（正手 9,274；反手 13,844）
- 该 checkpoint 的历史复现必须使用 CompletePriors，而非纯 V1.3B 或 Precision Rescue。

## 先验

- `model_3396_lower_prior.pt`：`checkpoints/frozen_priors/model_3396.pt`
- `model_900_upper_prior.pt`：`checkpoints/frozen_priors/model_900.pt`
- 二者均为运行时先验，不是学生网络的替代品。

## 兼容性

三模型只在以下条件同时满足时具有可比性：机器人 joint order、ready pose、root frame、goal frame、normalizer、action scale、控制频率、lookahead 和 alpha 组合均与项目合同一致。
