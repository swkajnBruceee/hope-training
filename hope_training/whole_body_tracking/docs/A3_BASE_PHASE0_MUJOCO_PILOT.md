# A3 Base Phase 0 隔离 MuJoCo Pilot

日期：2026-07-18  
状态：**历史 pilot，已被共享 causal trace + 无接触跨仿真 fixture v3 取代**

> 本文中的旧矩阵 hash、linear transport 数值和接触下单端响应不得再用于决策。
> 原 runner 自行生成 target，且接触与逐步状态恢复发生耦合；后续实证表明该组合会在
> Isaac 中制造显著 hip-roll 伪响应。现行证据和废止理由见
> [`A3_BASE_PHASE0_CROSS_SIM_FIXTURE.md`](A3_BASE_PHASE0_CROSS_SIM_FIXTURE.md)。

## 1. 目的和边界

本 pilot 使用官方 `a3_pingpong.xml`、MuJoCo `3.1.6` 和合同中的
31 DOF PD 增益，不启动 ROS、AimRT、MOTION 或网络发布者。它只回答：

- 单个 Base action 在当前 PD/接触模型下的响应、力矩和方向；
- waist pitch Strike reference + Base residual 的合成响应；
- 50 Hz target 在 200/1000 Hz native substep 下 ZOH 与线性插值的差异。

它不回答机器人能否站稳、横移或承受挥拍。这些必须由后续 Base
Policy 在 free-base Stand/StrikeSupport 门中验证。

## 2. 为什么修正 fixture

实验路线经过两次有证据的修正：

1. **free-base nominal PD**：`left_hip_pitch +0.10` 在约 `1.146 s`触发
   `6°` 倾角门。原因是尚无 Base Policy，实验将执行器响应与全身失稳混在一起。
2. **pelvis-only fixture**：46 个 repeat-1 case 中 42 个通过，4 个因
   `right_hand_finger_collision|right_hip_yaw_collision` 失败。原因是非被测上肢在低增益下下垂，污染单关节实验。

最终冻结 `single_joint_fixture_v1`：每个 native substep 之后恢复 freejoint
和所有非被测主动关节，只允许被测关节保持动力学演化。这是执行器/传输
诊断 fixture，不是平衡器。前两批结果已废止，但保留为修改实验语义的根据。

## 3. 最终候选矩阵和来源

```text
matrix semantic hash:
a3ce2b267eacffd98e98f4df86a947fc914df71918ad68cb117854df4b5f89e1

matrix file sha256:
ec06c550f9fcaa5ff62b16aff39a03522e958b31fa3e86e1ba15798a56928db9

native runner sha256:
1f77c96a7aae234e85a9b9b6906372ead988048fd18280eda8ad405e535d726d

pilot driver sha256:
d9dc821039b0772cd7a561b29866409f8f185c45a641e21b9cedcb32c6188c51
```

矩阵现在还内嵌 Base、Strike、Composer、Action 和 Calibration 五份合同的
canonical payload hash。任何 action scale、指标、fixture 或安全语义变化都会使旧矩阵失效。

## 4. Pilot 结果

| 范围 | case 数 | 安全门通过 | 用途 |
|---|---:|---:|---|
| 低幅 action + waist residual + transport, repeat 1 | 46 | 46/46 | 低幅与传输诊断 |
| 中幅 action, repeat 1 | 28 | 28/28 | 静态负载区和非线性诊断 |
| 6 组代表 case, repeat 1/2/3 | 18 | 18/18 | 确定性复核 |

6 组代表 case 的 metrics canonical hash 在三次运行中逐位一致。这证明
runner 是确定的，但不代表已覆盖随机化、传输抖动或真机噪声。

### Action response

- 中幅相对低幅总体呈单调增强，hip yaw 和 knee 响应方向左右一致。
- 踝关节的稳态 response ratio 仍很低，这与足部在 fixture 中贴地受约束一致，
  不能直接推导出应放大 action scale。
- `waist_roll ±0.25` 对应的稳态 response ratio 约为 `0.83`。
- `waist_pitch +0.25` 在零 Strike pitch 附近仍被重力静态负载压住，有效响应接近零；
  负向 residual 或带正 Strike pitch 参考时响应增强。这提示后续需要评估
  gravity compensation/姿态工作点，不支持现在直接改大 residual limit。
- 所有 step 在 `0.4 s` 观测窗内基本未持续进入 settling band，因此当前证据
  不批准最终 action scale。

### Target transport

| physics | joint | ZOH RMSE | linear RMSE | ZOH peak accel | linear peak accel |
|---:|---|---:|---:|---:|---:|
| 200 Hz | hip roll | 0.084624 | 0.084652 | 15.657 | 11.694 |
| 200 Hz | ankle roll | 0.062871 | 0.063201 | 25.179 | 25.556 |
| 200 Hz | waist pitch | 0.022504 | 0.022441 | 0.549 | 0.549 |
| 1000 Hz | hip roll | 0.084771 | 0.084830 | 15.674 | 12.464 |
| 1000 Hz | ankle roll | 0.062925 | 0.063380 | 33.538 | 33.687 |
| 1000 Hz | waist pitch | 0.022396 | 0.022354 | 0.180 | 0.180 |

线性插值对 hip roll 峰值加速度有小幅改善，但在 ankle roll 上 RMSE/峰值加速度
略差，waist pitch 基本无差异。因此 MuJoCo 单端没有证明 linear 是统一更优的选择。
在 Isaac native-substep 对照完成前，Composer 的 transport 保持 pending；不为追求“更平滑”
而无证据地提前引入插值。

## 5. 结论与下一门

本批证明 native runner、single-joint fixture、指标和安全包络可工作，但仍然：

```text
matrix_coverage_complete = false
automatic_promotion = false
free_base_stability_evidence = false
training_approved = false
```

下一步是实现 Isaac 内的同语义 single-joint fixture 对照，先比较 200 Hz ZOH，再决定
是否值得引入 linear substep。其后才能对 action scale、腰部工作点和执行器差异做版本化选择。
