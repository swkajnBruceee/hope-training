# P3：motion 去场景绑定与正式 P1 静态审计

生成日期：2026-08-03。状态：**坐标规范化已完成；仅具备五个不连续的位置局部技能，不能直接执行 Planner 10D。**

## 本阶段完成内容

六条 legacy motion 已转换为：

```text
motion_prior_base_heading_frame0/v1
```

该坐标以 motion frame 0 的 pelvis 位置为原点、pelvis yaw 为朝向，移除了源场景的
`X/Y/yaw` 放置；关节轨迹、相对高度、刚体姿态和速度不变。源 NPZ 没有被修改。输出位于：

```text
eval_outputs/strike_goal_p3/canonical_motion_prior_v1/
```

manifest 明确标记为 `coordinate_canonicalization_only`、`training_approved=false`。这一步只证明
刚体换坐标等价，不证明安全、动力学跟踪、平衡、碰撞或接触 TCP。

## 正式 P1 场景静态 FK

审计器将六个 canonical 命中状态逐条写入正式
`HOPE-PingPong-AgibotA3-v0` 场景，P1 pelvis anchor 实测为：

```text
[-0.5, -0.7625, 0.3084] m, yaw = 0
```

未推进物理时间、未调用策略动作、未启动训练。正式任务默认 racket FK 与强制
`right_wrist_yaw_Link + A3_MOUNT_OFFSET` 的位置逐条完全一致，说明该场景没有引入另一套
球拍点。

六个 task anchor 的静态 FK 位置误差为 `0.834–3.554 mm`。但 task target 与 motion 命中帧
本身并不是同一个完整接触状态：

* 法向差 `2.12–12.28°`；
* 速度差 `0.137–0.273 m/s`。

因此 manifest 的 `position/normal/velocity` 必须继续称为任务目标；NPZ FK 必须称为执行参考，
二者不能共用一个“真值”字段。

### 限位结论

六个命中姿态都越过了 Isaac soft joint limit，最差 `-0.0349 rad`，且全部发生在
`waist_roll_joint`。相对 hard limit 的剩余裕量只有：

```text
motion 0: 0.000533 rad
motion 1: 数值上位于硬限位边界
motion 2: 0.000664 rad
motion 3: 0.006760 rad
motion 4: 0.005082 rad
motion 5: 0.002104 rad
```

这意味着 motion 可以作为动作形状/相位先验，但不能原样作为“安全中心”。第一版轨迹适配器
必须拥有腰部形变权限，并把 waist roll 拉回有明确裕量的区域；不能再让 residual 长期顶限位。

结构化报告：`eval_outputs/strike_goal_p3/p1_formal_scene_fk_audit.json`。

## 当前真正验证过的能力

现有 P10 报告验证的并非六个 canonical task anchor，而是 motion `0/2/3/4/5` 五个专用校准
中心。motion 1 未放行。五个校准中心与 canonical task anchor 相差 `7.33–12.02 cm`。

在旧 tracking 执行链路内：

* 五个校准中心位置误差 `0.128–1.064 mm`；
* 每个中心的 `±1 cm` 单轴七点测试均完成且无物理终止；
* 七点位置最大误差按 motion 为 `1.98 / 1.21 / 6.06 / 4.95 / 4.67 mm`；
* 中心速度误差仍为 `1.09–2.23 m/s`；
* 五个 `±1 cm` 小盒之间最近仍有约 `6.6 cm` 空洞。

准确描述应是：

> 当前系统有五个高精度、位置局部、motion 专用的孤岛技能；尚没有连续工作空间，且速度条件
> 基本没有完成。

结构化对比：`eval_outputs/strike_goal_p3/anchor_gap_audit.json`。

## 一个现在可复现的击球点

选择 motion 3 的已验证校准中心，policy link point 为：

```text
base-heading: [0.483471, 0.066297, -0.082999] m
正式 P1 刚体放置后的 HOPE world: [-0.016529, -0.696203, 0.225401] m
```

旧 tracking 链路中心位置误差为 `0.921 mm`，法向误差 `5.79°`，无物理终止；`±1 cm` 单轴
测试的位置最大误差 `6.06 mm`。但是速度误差仍为 `1.24 m/s`，正式 P1 场景只完成静态 FK，
没有完成浮动基座动态执行。

此外，这个点是 `pingpang_red_Link origin`，不是 Planner 发布的球心。硬件 contact transform
尚未闭环，不能把它直接当成 Planner ball-center point。

## 能否直接挥拍到 Planner 给定位置

当前结论是 **不能**。原因不是位置 FK 不可达，而是：

1. Planner 10D shadow 对 action 仍为零影响；
2. Planner position 是球心，当前局部技能目标是 policy link point；
3. 硬件球心到 link 的 contact transform 未认证；
4. Planner/control 时钟映射未完成部署标定；
5. 正式 P1 场景尚无动态回放放行；
6. 现有技能只覆盖五个相互分离的 `±1 cm` 位置小邻域；
7. 即便在中心，速度误差也明显不合格。

## 下一步放行顺序

下一阶段不是立即做全局 PPO，而是：

1. 在正式 P1 场景动态回放 canonical motion，检查全轨迹桌面/自碰撞、腰部限位和浮动基座；
2. 先实现带腰部退限位约束的低维轨迹适配器，并用五个已验证中心做行为继承；
3. 将“canonical task anchor”和“实际执行参考”作为两个显式字段，禁止继续互相覆盖；
4. 通过中心回归后只训练 bridge 位置目标，先闭合五个技能岛之间的空洞；
5. 位置连续泛化通过后，再依次训练法向、速度和 time-to-hit。

