# P4A 正式场景全轨迹动态资格审计

## 结论

P4A 没有开始 PPO。它验证的是五条 legacy motion 在正式 P1 场景中能否作为修复起点，而不是它们是否已经构成 10D 技能中心。

现有下肢稳定器在标称动力学下能够执行完 motion 0/2/3/4/5，五条均无物理终止。但五条 reference 都越过软限位，而且正式 A3 资产关闭了 self-collision，所以当前没有 A 类安全先验。

保守的修复种子分类为：

| motion | 分类 | 主要原因 |
| --- | --- | --- |
| 0 | B | 标称动态可执行且审计窗口内恢复；仍需确定性退限位 |
| 2 | C | 审计窗口内未同时达到 ready + stable |
| 3 | C | canonical 位置误差超过修复种子筛选线，且未恢复 |
| 4 | C | 法向和速度误差大，且未恢复 |
| 5 | C | 法向和速度误差大，且未恢复 |

这些 B/C 是 P4B 的修复优先级，不是最终验收等级。

## 证据边界

`replay_canonical_prior_p1_dynamic.py` 的裸 implicit-PD 回放是 plant/controller baseline。它说明裸 PD 无法稳定执行这类浮动基座轨迹，但不能依此把 motion 判为 D。正式分类使用现有上肢执行策略和下肢稳定器的全轨迹回放。

随机化 7-env 探针中，motion 2 有 2/7 物理终止，motion 3 有 3/7，其余为 0/7。该探针包含 startup physics material randomization，因此作为鲁棒性警报，不与标称资格结果混合。

## 标称回放摘要

| motion | 位置误差 m | 法向误差 deg | 速度误差 m/s | 最大倾角 deg | 最小 reference 软限位裕量 rad |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0484 | 3.33 | 1.000 | 5.98 | -0.0454 |
| 2 | 0.0738 | 4.02 | 1.141 | 6.16 | -0.0454 |
| 3 | 0.1082 | 5.98 | 1.364 | 6.89 | -0.0454 |
| 4 | 0.0597 | 15.39 | 2.322 | 7.67 | -0.0349 |
| 5 | 0.0752 | 13.50 | 1.847 | 5.83 | -0.0349 |

以上数值由脚本从原始 JSON 重算；机器可读结果为 `eval_outputs/strike_goal_p4/p4a_policy_qualification_summary.json`。

## 当前不允许 PPO 的原因

1. 全部 reference 软限位裕量为负；
2. self-collision 尚未可观测；
3. Planner 的 ball-center 目标还需要经过显式接触几何转换成 policy TCP；
4. canonical goal、legacy calibrated center、adapted reference 和 actual execution 尚未在同一 trace 中分层；
5. 击球速度还未达到 10D 技能的可接受精度。

## 下一步

P4B 将先对 motion 0 建立低维、平滑、确定性的轨迹形变，将 waist/shoulder 等关节拉回正软限位裕量，同时约束命中时刻的 policy TCP 位置、法向和速度不退化。第一个修复样本经 MuJoCo 几何审计和正式 P1 动态回放通过后，再扩展到其他 motion。
