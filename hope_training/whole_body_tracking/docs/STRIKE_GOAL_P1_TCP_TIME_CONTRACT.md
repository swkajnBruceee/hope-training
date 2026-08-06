# P1：Planner TCP 与时间合同审计

生成日期：2026-08-03。状态：**部分实测通过，但物理 contact transform 与部署时钟映射
未通过；禁止接入 actor 或启动泛化训练**。

本文件将“消息可解析”与“物理目标可执行”分开。前者已经由
`training/utils/strike_goal.py` 的 `PlannerRacketCommand` 适配器和单测固定；后者仍需
运行时证据。

## 已由源码确认的消息/时钟事实

| 项目 | 结论 | 依据 |
| --- | --- | --- |
| 消息类型 | `msgs/msg/RacketCommand` | `hope_ws/src/msgs/msg/RacketCommand.msg` |
| 向量顺序 | 适配器输出固定为 `position, normal, velocity, time_to_strike` | `PlannerRacketCommand.from_ros_message()` |
| 发布坐标 | `header.frame_id = "world"` | `solver/src/solver_node.cpp:306-314` |
| Planner position 数据来源 | `out.position = plan.p_hit`，且 `plan.p_hit = strike.p_ball` | `solver/src/solver_node.cpp:309`、`solver/src/hit_plan_solver.cpp:37` |
| `strike.p_ball` 语义 | 预测击球时的**球位置/球心轨迹状态** | `trajectory/include/ball_trajectory_predictor.h:15` |
| Planner velocity 数据来源 | 理想碰撞模型 `computeRacketVelocity()` 的 `v_racket` | `solver/src/racket_target_solver.cpp:122-137`、`hit_plan_solver.cpp:60-62` |
| `time_to_strike` 数据来源 | `strike.t_strike - last_t_` | `trajectory/src/strike_prediction_node.cpp:141` |
| `last_t_` 时钟来源 | 进入 trajectory 的球消息 `header.stamp` | `trajectory/src/strike_prediction_node.cpp:317-320` |
| `RacketCommand` 时间转发 | solver 保留上游 `time_to_strike`，并把 `plan.t_hit` 放入 `strike_time` | `solver/src/solver_node.cpp:313-314` |

因此，原始合同不是“同一球拍 TCP 的 p/n/v”，而是一个混合但可明确版本化的冲击合同：

```text
strike_goal_10d/ball_center_impact_v1
  position        = predicted_ball_center_at_strike
  normal          = desired_ideal_racket_face_normal
  linear_velocity = desired_ideal_racket_impact_velocity
  time            = source time_to_strike, latched onto control clock after receipt
```

`velocity` 只是当前无旋转理想拍面碰撞模型的速度。如果将 position 从球心转到
`pingpang_red_Link origin`，则严格的点速度转换还需要 `omega × r`；现有 10D 没有
角速度，所以适配层不得声称它已获得某个刚体点的精确线速度。

### 运行时存在两套不兼容 Planner

本机同时发现：

| overlay | 消息类型 | 关键区别 | 合同状态 |
| --- | --- | --- | --- |
| 当前仓库 `hope_ws/src` | `msgs/msg/RacketCommand` | 含 incoming/outgoing ball velocity；对应当前 C++ trajectory/solver | P1 canonical candidate |
| `/home/bruce/hope_ws_hopett_ros/install` | `hope_msgs/msg/RacketCommand` | 不含 incoming ball velocity；对应旧 `hope_planner` | 明确不兼容，禁止送入 v1 adapter |

两者 interface hash 分别为 `4aeff473...` 和 `bd984aa...`。DDS 类型名本身不同，但部署启动
脚本仍必须检查 package/topic 类型，避免误把旧 overlay 当成当前 Planner。当前仓库 ROS
构建还暴露了一个工程限制：位于中文路径时 `rosidl_generate_interfaces` 会生成截断的 IDL
依赖路径；本次通过已有 ASCII symlink `/home/bruce/HOPETableTennis` 和隔离 build/install
目录完成构建。最终部署/CI 应固定 ASCII workspace 路径。

## 静态几何审计

URDF `agibot/URDF/A3T2.5-URDF-std-pingpang/urdf/URDF-JOINT-LINK.urdf` 确认：

```text
right_wrist_yaw_Link
  └─ fixed right_hand_pingpang_joint, origin = (0, 0, 0)
       └─ right_hand_pingpang_Link
            └─ fixed pingpang_red_joint,
                 origin = (0.210210, 0.032078, 0.032036) m
                 └─ pingpang_red_Link
```

`pingpang_red_Link` 的 visual/collision origin 都是零，引用同一个 STL。该 mesh 的本地
包围盒约为 `X=[-84.23, 76.20] mm`、`Y=[-2.905, 0] mm`、`Z=[-84.23, 76.20] mm`：它是一个
薄的 XZ 拍面，几何面法向近似为本地 ±Y。此结果支持当前训练 FK 的
`normal_axis=1` 选择，但**不证明** origin 是可用于球接触的中心，也不决定正负号。

另一个待修复的不一致是：Planner 的 `normalToQuaternion()` 将本地 `+X` 对齐到 normal，
而训练的 racket-state 当前从本地 axis `+Y` 取 normal。`RacketCommand` 本身只发布 normal
向量，因而 P1 可以先统一向量含义；若后续采用 Planner 的 quaternion 或全姿态目标，必须
显式插入 `R_planner_face_to_policy_racket`，不能直接复用 quaternion。

## 2026-08-03 Isaac 运行时结果

使用当前 P10 checkpoint 对 motion `0,2,3,4,5` 的五个已验证中心重新执行了确定性、无球、
单拍回放。五次运行均报告：

```text
racket FK mode = wrist_offset
wrist body      = right_wrist_yaw_Link
mount offset    = (0.2102113992, 0.0320784995, 0.0320358706) m
```

它与 URDF 的 `pingpang_red_joint` 固定平移一致。因此当前 policy/WBC 参考点已经可以正式命名
为：

```text
policy_racket_point/v1 = pingpang_red_Link origin
                       = right_wrist_yaw_Link origin + fixed mount transform
```

五个中心全部完成命中，位置误差均值 `0.816 mm`、范围 `0.128–1.064 mm`；记录时刻的法向
误差范围 `1.59–7.71°`，没有出现 180° 反向。这证明 P10 内部 FK/目标点和 normal axis/sign
在现有 motion 上自洽，但仍不证明 `pingpang_red_Link origin` 等于物理球接触 TCP，也不
证明 Planner 球心可无偏移地作为该点目标。

原始报告位于 `eval_outputs/strike_goal_p1/p10_center_motion_{0,2,3,4,5}.json`。

## 2026-08-03 球碰撞探针

为了验证球心到 policy FK 点的符号和量级，对 motion 0 做了两组无脚本反射、
PhysX 自然碰撞试验：

1. **显式诊断 proxy**：原球拍 mesh 碰撞被禁用，一个跟随同一 FK 姿态/速度的薄长方体
   作为唯一拍面碰撞体。球心目标使用
   `p_ball = p_pingpang_red_origin - 0.017 * normal`。在 step 81 实测球心到
   FK 原点距离 `22.8 mm`、法向距离 `17.1 mm`、横向距离 `14.3 mm`，球的
   X 速度从 `+2.851 m/s` 自然变为 `-2.372 m/s`。这证明对该 proxy 而言符号正确：
   `p_policy = p_ball + 0.017 * normal`。
2. **原始导入 URDF 碰撞**：不使用 proxy，配置 24 mm 球心目标后球也改变为返回
   `-X`，但最小 FK 距离为 `61.6 mm`、最小法向距离为 `53.4 mm`，而且当前
   `racket_ball_contact` sensor 实际挂在 Ball 上，会统计球与任何 robot 碰撞。因此这次反弹
   不能认证为球拍拍面命中，也不能认证 24 mm TCP。

17 mm 因此只记为 `diagnostic_proxy_only`；导入 URDF 与真机 offset 仍为
`unresolved`。当前 `cfg/strike_goal.yaml` 中 `contact_mapping.enabled=false`，不会将任何一个
诊断数字静默送入 actor。
结构化结果位于 `eval_outputs/strike_goal_p1/contact_probe_summary.json`。

## 必须做的运行时 TCP 探针

使用至少五个同步姿态：准备姿态、motion 0 命中帧、motion 2 命中帧、一个其他空间边缘
motion、一个随机安全姿态。每条记录必须同一控制 tick 读取，且给出：

```json
{
  "sample_id": "motion_2_hit",
  "timestamp_s": 12.34,
  "pose_label": "motion_2_hit",
  "planner_command_position_world": [0.0, 0.0, 0.0],
  "planner_normal_world": [0.0, 1.0, 0.0],
  "policy_tcp_position_world": [0.0, 0.0, 0.0],
  "policy_tcp_normal_world": [0.0, 1.0, 0.0],
  "racket_link_origin_world": [0.0, 0.0, 0.0],
  "world_from_racket_rotation": [[1,0,0],[0,1,0],[0,0,1]],
  "policy_tcp_name": "pingpang_red_Link origin"
}
```

保存为 JSON list 或 `{"samples": [...]}`，运行：

```bash
cd hope_training/whole_body_tracking
python tools/audit_strike_goal_contract.py tcp tcp_samples.json
```

输出同时给出 world 偏差与 `R_world_racket^T (p_planner - p_policy_tcp)` 的球拍局部偏差，
以及法向夹角。由项目的真实接触/控制容差决定是否接受；工具不会把任意常数阈值写进
代码。仅当局部偏差在多姿态下稳定，才可将它提议为固定刚体 TCP transform；即使稳定，
仍需确认 raw Planner point 是球心还是面接触目标后才选择球半径方向补偿。

## 必须做的运行时时间探针

每次接收和每个随后的控制 tick 记录下列字段；`source_clock_domain` 与
`control_clock_domain` 必须如实填写（例如 `mocap_ros`、`isaac_sim`、`robot_steady`），不得
都填成 `world`：

```json
{
  "command_id": "planner-strike-42",
  "source_clock_domain": "mocap_ros",
  "control_clock_domain": "isaac_sim",
  "header_stamp_s": 101.0,
  "strike_time_s": 101.52,
  "message_time_to_strike_s": 0.52,
  "received_control_time_s": 15.20,
  "current_control_time_s": 15.24,
  "policy_time_to_strike_s": 0.48,
  "simulation_time_s": 15.24,
  "control_step": 1524
}
```

分析命令：

```bash
python tools/audit_strike_goal_contract.py time time_samples.json
```

该命令只检查两个不会混时钟域的关系：

```text
strike_time - header_stamp == message_time_to_strike
policy_tts == max(received_message_tts - elapsed_control_time, 0)
```

若已通过独立时钟同步测得 `control_time = source_time + offset`，才允许额外传
`--source-to-control-offset-s OFFSET` 评估映射后的剩余时间。工具绝不从样本数据猜 offset。

P1 时间通过还要求完成静态时钟一致性、人工消息延迟和不同 Isaac real-time factor 三个
实验。Isaac 训练倒计时绑定 sim/control time；真机倒计时绑定单调控制时钟；两者通过一个
环境外的 time provider 实现，不进入 actor。

## 2026-08-03 ROS 时间与字段实测

当前仓库五个核心包在隔离 ASCII build 中成功编译；`common/trajectory/solver/decision` 共
`47 tests, 0 errors, 0 failures`。随后启动真实 `solver_node` 并发布三条有效
`msgs/msg/PredictedStrike`：

* `RacketCommand.header.stamp`、`strike_time`、`time_to_strike` 均原样转发；
* `RacketCommand.position` 与输入预测球位置逐值相等；
* 真实生成消息可由 `PlannerRacketCommand.from_ros_message()` 解析为固定 `p,n,v,t` 顺序；
* solver 计算 velocity/normal，但不会按发布/通信延迟更新 `time_to_strike`。

受控注入 `0/50/150 ms` 发布延迟后，接收端实测消息年龄为
`7.77/56.30/154.36 ms`，而消息中的剩余时间始终为 `0.5 s`；其陈旧量与消息年龄逐值相等。
报告位于 `eval_outputs/strike_goal_p1/planner_ros_delay_probe.json`，可由
`tools/probe_planner_racket_command_ros.py` 重复生成。

为此新增 `LatchedStrikeGoal`：只接受显式 `control_clock_domain`，可扣除**已测得**的
pre-receipt delay，随后按同一个 control clock 每 tick 递减，并拒绝时钟倒退。它不会猜测
mocap→sim/robot 的时钟 offset；真实部署的时钟同步仍是 P1 未完成项。

## 回归结果

P1 Python 合同、TCP/time 审计器、external request、target adapter 和 actor-critic 兼容测试共
`36 passed`；当前 C++ Planner 四个核心包共 `47 passed`。YAML/JSON 合同文件与所有新工具
均通过解析/语法检查。

对修改后的 `scripts/play.py` 再运行一次无球 motion 0 中心回放：自动选择 motion 0，
位置误差 `0.912 mm`，法向误差 `2.45°`，无物理终止。这证明诊断修复没有改变
现有 P10 中心能力；新 10D 和 contact mapping 仍未启用。

P2 只读 shadow 已经接入 Isaac 并通过运行检查，详见
`docs/STRIKE_GOAL_P2_SHADOW.md`。它另外发现 legacy tracking motion 的 world placement
与 Planner P1 击球平面不是同一场景放置；该问题必须由正式 match scene 迁移解决，
不能并入 TCP offset。

## P1 放行条件

在以下四项均有记录、报告和明确版本化定义以前，`strike_goal.enabled` 必须保持 false：

1. policy FK 点现已命名为 `pingpang_red_Link origin`；仍需决定它是否就是最终 TCP，并给出
   Planner 球心到该点的显式 contact transform；
2. normal 的 local axis 与符号已通过多姿态比较，且没有偷用 Planner `+X` quaternion；
3. Planner source clock 到训练/控制 clock 的映射及通信/推理/执行延迟归属已测量；
4. actor shadow trace 中每 tick 的剩余时间按照接收时刻递减，过期目标 fail-closed。

在此之前，10D 仅可作为消息审计对象，不能用于 P2 observation 或任何训练损失。
