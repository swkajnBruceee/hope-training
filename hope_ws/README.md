# HOPE 乒乓球 ROS2 工作空间

本目录是 HOPE 乒乓球机器人项目的 ROS2 colcon 工作空间根目录。核心代码已经从旧的 Python / `hope_planner` 结构迁移到 ROS2 C++ / `ament_cmake` 多包结构,所有运行时业务逻辑均使用 C++ 实现。

- 工作空间根目录:`/home/gzy/robot_pingpang/HOPE/hope_ws`
- ROS 发行版:Humble(已在 Humble 上验证)
- 构建系统:colcon + ament_cmake
- 当前状态:trajectory / decision / solver 基础闭环已就绪,共 15 个单元测试全部通过

---

## 1. 当前状态概览

- 这是一个标准 ROS2 colcon workspace,根目录为 `hope_ws`
- `src/` 下按职责拆成多个独立的 `ament_cmake` 包
- 当前主要完成 trajectory(来球估计与轨迹预测)→ decision(目标决策)→ solver(击球解算)的闭环
- 运行时核心链路已 C++ 化:`decision_node`、`solver_node` 均为 C++ 可执行文件
- `msgs` 包提供 `RacketCommand` 与 `TargetDecision` 两条自定义消息
- 测试状态(最近一次本地验证):

  ```
  Summary: 15 tests, 0 errors, 0 failures, 0 skipped
  ```

- 物理常量与默认值集中在 `common/include/constants.h` 与 `decision/src/target_selector.cpp`
- 当前 `decision_node` 以 `fixed_center` 策略发布固定目标决策,作为后续真实策略的占位实现

---

## 2. 工作空间结构

实际目录布局(基于只读扫描,非凭空):

```
hope_ws/
├── build/                     # colcon 构建产物
├── install/                   # colcon 安装产物,source install/setup.bash 使用
├── log/                       # colcon 构建/测试日志
├── README.md                  # 本文件
└── src/
    ├── common/                # 公共常量与四元数工具(C++)
    ├── trajectory/            # 球状态估计 + 轨迹预测 + UDP overlay 节点(C++)
    ├── solver/                # 击球解算核心(C++)
    ├── decision/              # 目标决策节点,当前固定策略(C++)
    ├── msgs/                  # 自定义 ROS2 消息
    ├── bringup/               # launch 文件、坐标系配置、mocap 中继
    ├── calibration/           # 标定相关(C++)
    └── tools/                 # 离线工具(bag_to_csv 等,C++)
```

`build/`、`install/`、`log/` 由 colcon 生成,通常不进入版本控制。源码全部位于 `src/`。

---

## 3. 包职责说明

### 3.1 common

公共基础库,所有其他 C++ 包均依赖。

- 球台尺寸(`TableParams`)、球物理参数(`BallPhysics`)、解算调参(`PlannerConfig`)
- 四元数与法向量之间的转换工具

关键文件:

- `include/constants.h` — `TableParams`、`BallPhysics`、`PlannerConfig` 结构定义
- `include/quaternion_utils.h` — `normalToQuaternion`
- `src/quaternion_utils.cpp`

依赖:`Eigen3`(`geometry_msgs` 由 solver/decision 间接使用)。

### 3.2 trajectory

球的来球状态估计、轨迹预测,以及可选的可视化 overlay 节点。

- `BallStateEstimator`:对最近 N 个采样点做二阶多项式拟合,解析求导得到平滑位置/速度;每次检测到桌弹跳清空缓存
- `BallTrajectoryPredictor`:1 kHz 显式欧拉前向积分,带二次阻力 + 重力 + 桌弹跳(`v+ = diag(C_h, C_h, -C_v) @ v-`),输出到虚拟击球面的 `StrikeTarget`
- `trajectory_overlay_udp_node`:可选的 UDP overlay 节点,用于把预测轨迹广播到可视化端

关键文件:

- `include/ball_state_estimator.h`
- `include/ball_trajectory_predictor.h`
- `src/ball_state_estimator.cpp`
- `src/ball_trajectory_predictor.cpp`
- `src/trajectory_overlay_udp_node.cpp`

### 3.3 decision

目标决策节点,负责告诉 solver "打到哪里、用多长滞空时间、限速多少"。

- 当前以 `fixed_center` 策略发布 `/target_decision`
- 内部通过 `TargetSelector::selectDefault()` 给出固定默认值
- 未来只需替换 `decision` 包内部的策略实现,无需修改 solver

关键文件:

- `include/target_selector.h`
- `src/target_selector.cpp`
- `src/decision_node.cpp`

可执行文件:`decision_node`(10 Hz 默认发布频率,可通过参数 `publish_rate_hz` 调整)。

当前默认决策(由 `TargetSelector::selectDefault()` 给出):

| 字段 | 值 |
| --- | --- |
| `target_land` | `[2.055, -0.7625, 0.0]` |
| `delta_t_flight` | `0.5` s |
| `net_clearance_margin` | `0.03` m |
| `max_racket_speed` | `6.0` m/s |
| `desired_ball_speed` | `-1.0`(未启用) |
| `max_ball_out_speed` | `-1.0`(未启用) |
| `valid` | `true` |
| `mode` | `"fixed_center"` |

> 注:`net_clearance_margin = 0.03` 同样在 `solver/include/hit_plan.h` 的 `makeDefaultSolveTarget(config)` 与 solver 参数 fallback 中出现,保证 solver 在没有收到 decision 时也使用一致的过网安全裕度。

### 3.4 solver

击球解算核心节点,消费 trajectory 的击球候选与 decision 的目标,输出 `RacketCommand`。

- 输入话题:`/ball/point`(主,PointStamped,best-effort)、`/poses`(兼容回退,PoseArray)
- 输入话题:`/target_decision`(`msgs/msg/TargetDecision`)
- 输出话题:`/racket/command`(`msgs/msg/RacketCommand`,reliable)
- 诊断话题:`/planner/diagnostics`(10 Hz)
- 没有收到 decision 时,使用由参数构造的 `SolveTarget`(`mode = "default_fixed_center"`、`valid = true`)作为兜底
- 收到 `valid = false` 的 decision 时,记录 `last_target_reason_ = "invalid_decision_ignored"` 并保留当前目标,不覆盖
- 物理模型:无旋转(Magnus 暂未引入);出球通过同阻力模型反向积分落到 `target_land`

关键文件:

- `include/hit_plan.h` — `SolveTarget`、`HitPlan`、`makeDefaultSolveTarget(config)`
- `include/hit_plan_solver.h` — `HitPlanSolver`(阶段 3 适配层)
- `include/racket_target_solver.h` — `RacketTargetSolver`(`computeOutgoingVelocity`、`computeRacketVelocity`、`checkNetClearance`)
- `include/solver_pipeline.h` — `HOPESolverPipeline`(串联 Stage 1/2/3)
- `src/hit_plan_solver.cpp`
- `src/racket_target_solver.cpp`
- `src/solver_pipeline.cpp`
- `src/solver_node.cpp`
- `config/solver.yaml` — `solver.launch.py` 加载的默认参数
- `config/hope_solver.yaml` — 与 `solver.yaml` 等价的另一份参数文件,保留给外部/历史调用方使用;两份文件当前内容一致

可执行文件:`solver_node`。

### 3.5 msgs

ROS2 自定义消息包(`ament_cmake` + `rosidl_generate_interfaces`)。

- `RacketCommand.msg` — solver 的输出:`position`、`velocity`、`normal`(球拍状态)、`strike_time`、`time_to_strike`、`ball_velocity_outgoing`、`valid`、`clears_net`、`bypasses_net_posts`、`predicted_bounces`
- `TargetDecision.msg` — decision 的输出:详见第 6 节

仅含消息定义,无运行时逻辑;由 `ament_cmake` 的 `rosidl_generate_interfaces` 生成对应的 C++/Python 头文件。

### 3.6 bringup

系统启动入口,只放 launch 文件、坐标系配置、mocap 中继桥,不放置业务逻辑。

当前 launch 文件:

- `launch/solver.launch.py` — 启动 `solver_node`,加载 `solver/config/solver.yaml`(solver 的最终唯一入口)
- `launch/hope_world.launch.py` — 世界坐标系 / 静态 TF
- `launch/avatar_pro_vrpn_relay.launch.py` — mocap 中继入口
- `launch/avatar_pro_hope_bridge.launch.py` — avatar_pro ↔ HOPE 桥接入口
- `src/avatar_pro_vrpn_relay.cpp`、`src/ball_truth_udp_bridge.cpp` — mocap / 球真值 UDP 桥接 C++ 节点

### 3.7 calibration

C++ 标定包,提供标定 CLI 与单元测试。

关键文件:

- `include/calibration.h`、`include/split_calibration_csv.h`
- `src/calibration.cpp`、`src/calibration_cli.cpp`
- `src/split_calibration_csv.cpp`、`src/split_calibration_csv_cli.cpp`
- `test/test_calibration.cpp`、`test/test_split_calibration_csv.cpp`

是否完整可用、是否被运行时调用,以实际代码为准;当前为 C++ 实现。

### 3.8 tools

离线工具包,`bag_to_csv` 把 ROS bag 转换为 CSV 供离线分析。

关键文件:

- `include/bag_to_csv.h`
- `src/bag_to_csv.cpp`、`src/bag_to_csv_cli.cpp`

当前为 C++ 实现。

---

## 4. 当前数据流

```
                    ┌────────────────────┐
                    │   mocap / 视觉     │
                    │  /ball/point       │
                    │  /poses (回退)     │
                    └─────────┬──────────┘
                              │
                              ▼
            ┌──────────────────────────────────┐
            │  solver_node (solver_pipeline)   │
            │                                  │
            │  BallStateEstimator  (Stage 1)   │
            │           │                      │
            │           ▼                      │
            │  BallTrajectoryPredictor (Stage2)│
            │           │                      │
            │           ▼                      │
            │  HitPlanSolver (Stage 3)         │
            │     ├─ RacketTargetSolver        │
            │     ├─ computeOutgoingVelocity   │
            │     ├─ computeRacketVelocity     │
            │     └─ checkNetClearance         │
            └────────────────┬─────────────────┘
                             │
                             ▼
                       /racket/command
                       /planner/diagnostics

   ┌─────────────────────────┐
   │  decision_node (10 Hz)  │
   │  TargetSelector         │
   │  selectDefault()        │
   └────────────┬────────────┘
                │  /target_decision
                ▼
        solver_node latest_target_
```

职责划分:

- **trajectory**:来球在哪里、未来怎么飞、哪里可以击球(产出 `StrikeTarget`)
- **decision**:打哪里、用多长滞空时间、速度限制是多少(产出 `/target_decision`)
- **solver**:给定来球候选和目标,解出球拍应该处于什么状态(产出 `RacketCommand`)

---

## 5. 当前解算逻辑

`HitPlanSolver::solve(strike, target)` 是阶段 3 的入口,定义在 `solver/src/hit_plan_solver.cpp`。

**输入**:

- `trajectory::StrikeTarget` — 由 Stage 2 预测得到的击球点/击球时刻/来球速度
- `solver::SolveTarget` — 由 decision 给出,或在没有 decision 时由 `makeDefaultSolveTarget(config)` 兜底

**输出**:`solver::HitPlan`,含 `p_hit`、`t_hit`、`v_in`、`target_land`、`flight_time`、`v_out`、`racket_velocity`、`racket_normal`、`racket_orientation`、三组 `bool(valid / clears_net / bypasses_net_posts)`、`score`、`reason`。

**执行流程**:

1. 复制 `p_hit / t_hit / v_in / target_land / flight_time` 到 `plan`
2. 检查 `strike.valid` —— 否则 `reason = "invalid_strike"`
3. 检查 `target.valid` —— 否则 `reason = "invalid_target"`
4. 检查 `target.delta_t_flight > 0` —— 否则 `reason = "non_positive_flight_time"`
5. 调用 `RacketTargetSolver::computeOutgoingVelocity` 求 `v_out`
6. 调用 `RacketTargetSolver::computeRacketVelocity` 求 `(racket_velocity, racket_normal)` 并写回 `plan.racket_orientation`
7. 调用 `RacketTargetSolver::checkNetClearance`,传入 `target.net_clearance_margin`
8. NaN/Inf 防护 —— 否则 `reason = "non_finite_solution"`
9. 检查 `clears_net` —— 否则 `reason = "net_not_clear"`
10. `target.max_ball_out_speed > 0` 且 `|v_out|` 超限 → `reason = "ball_speed_limit"`
11. `target.max_racket_speed > 0` 且 `|racket_velocity|` 超限 → `reason = "racket_speed_limit"`
12. 全部通过 → `valid = true`、`reason = "ok"`

**关键设计点**:

- 固定目标值(`[2.055, -0.7625, 0.0]`、`delta_t_flight = 0.5`、`net_clearance_margin = 0.03`、`max_racket_speed = 6.0`)不是写死在 `HitPlanSolver` 的物理解算内部,而是来自:
  - `decision::TargetSelector::selectDefault()`(运行时正常路径)
  - `solver::makeDefaultSolveTarget(config)` 和 `solver_node` 的参数 fallback(没有 decision 时的兜底路径)
- 未来真实策略只需要在 `decision` 包内替换 `TargetSelector` 实现并通过 `/target_decision` 发布,solver 侧的物理解算无需改动

---

## 6. 目标决策接口预留

当前 `decision_node` 以 `fixed_center` 策略周期发布 `/target_decision`;未来真实策略(包括落点选择、目标球速、滞空时间、限速)只需替换 `decision` 包的内部实现即可,solver 不需要改核心物理解算。

`msgs/msg/TargetDecision` 字段(`ros2 interface show msgs/msg/TargetDecision` 实测):

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `header` | `std_msgs/Header` | 时间戳与坐标系 |
| `target_land` | `geometry_msgs/Point` | 期望落点(`x`, `y`, `z`) |
| `delta_t_flight` | `float64` | 出球到落点的滞空时间(s) |
| `desired_ball_speed` | `float64` | 期望出球速度;<0 表示不约束 |
| `max_ball_out_speed` | `float64` | 出球速度上限;<0 表示不约束 |
| `max_racket_speed` | `float64` | 球拍速度上限;<0 表示不约束 |
| `net_clearance_margin` | `float64` | 过网安全裕度(m) |
| `valid` | `bool` | 该决策是否可用;`false` 时 solver 不会覆盖当前目标 |
| `mode` | `string` | 决策模式标签(如 `fixed_center`),用于诊断 |

**solver 侧行为约定**:

- 没有收到过任何 decision 时:使用由参数 fallback 构造的 `SolveTarget`,`valid = true`,`mode = "default_fixed_center"`
- 收到 `valid = false` 的 decision:忽略,记录诊断 `target_reason = "invalid_decision_ignored"`,保留当前目标
- 收到 `valid = true` 的 decision:覆盖 `latest_target_`,记录 `target_reason = "decision_update"`

---

## 7. 构建方法

```bash
cd /home/gzy/robot_pingpang/HOPE/hope_ws
source /opt/ros/humble/setup.bash

# 全量构建
colcon build

# 或只构建运行时闭环相关的几个包(推荐在迭代时使用)
colcon build --packages-select msgs common trajectory decision solver

# 编译完成后加载工作空间
source install/setup.bash
```

构建产物位于 `build/`、`install/`,日志位于 `log/`。

---

## 8. 测试方法

```bash
cd /home/gzy/robot_pingpang/HOPE/hope_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

colcon test --packages-select decision solver
colcon test-result --verbose
```

**最近一次本地验证结果(实跑)**:

```
Summary: 15 tests, 0 errors, 0 failures, 0 skipped
```

---

## 9. 常用检查命令

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

# 消息接口
ros2 interface show msgs/msg/TargetDecision
ros2 interface show msgs/msg/RacketCommand

# 可执行文件
ros2 pkg executables solver
ros2 pkg executables decision

# launch 参数
ros2 launch bringup solver.launch.py --show-args

# 启动 solver(最终入口)
ros2 launch bringup solver.launch.py

# 节点直接启动(带默认配置)
ros2 run solver solver_node --ros-args --params-file src/solver/config/solver.yaml
ros2 run decision decision_node
```

**残留检查**:

```bash
# Python 运行时文件:运行时核心链路已 C++ 化,
# 除 ROS2 launch 文件外,src 下无项目自定义 Python 运行时文件
find src -name "*.py" -not -path "*/launch/*" -not -path "*/.pytest_cache/*"

# 项目自定义 .hpp:目前 src 下无项目自定义 .hpp,
# 只允许 ROS2 自动生成消息头使用 .hpp
find src -name "*.hpp"
```

实测两个命令当前均无输出。

---

## 10. 编码规范

- 所有 ROS2 包统一使用 `ament_cmake`,禁止运行时使用 `ament_python`
- C++ 源码放在 `src/<package>/src/`
- 头文件放在 `src/<package>/include/`,**直接在 `include/` 下放 `.h` 文件**
- **不使用** `include/<package_name>/xxx.h` 这种嵌套路径
- **不使用**项目自定义 `.hpp`;只允许 ROS2 自动生成的消息头使用 `.hpp`
- **不保留**运行时 Python wrapper;只有 ROS2 launch 文件可以是 Python
- 物理常量集中在 `common/include/constants.h`,不要在 solver/decision 内散落硬编码
- `solver` 不直接决定目标:目标必须来自 `decision` 包发布的 `/target_decision`,或在缺数据时来自 `makeDefaultSolveTarget(config)` 兜底;`HitPlanSolver` 的物理解算内部不允许再写一份"固定目标"
- include 顺序建议:本包头文件 → 其他 ROS2 包头文件 → 第三方库 → 标准库;每个区块之间空一行
- 命名空间:每个包使用 `<package>` 小写命名空间(`solver`、`decision`、`trajectory`、`common`)

---

## 11. 下一步开发计划

1. **decision 从 fixed_center 改成真实策略**
   - 替换 `TargetSelector` 内部实现,基于轨迹预测结果选择落点、滞空时间、速度限制
   - 保留 `mode` 字段用于诊断与回放
2. **solver 多候选 / 多目标评分**
   - 在 `HOPESolverPipeline` 内部枚举多个 `StrikeTarget` 候选,选择 `HitPlan::score` 最高的一个
   - 输出时附带候选索引/分数,便于调试
3. **接入可达性 / IK / 球拍速度约束**
   - 在 `HitPlanSolver` 之后或之内加入 IK 可达性检查
   - 把可达性失败作为新的 `reason`(例如 `unreachable`)
4. **下游控制对齐**
   - 把 `RacketCommand` 与真实机器人控制接口(轨迹插值器、低层控制器)对齐
   - 引入时间同步(根据 `strike_time` / `time_to_strike` 触发)
5. **trajectory_overlay 与 C++ solver 的关系收尾**
   - 确认 `trajectory_overlay_udp_node` 的输出格式与可视化端的协议;必要时升级为 `trajectory_overlay_node.cpp` 并加入新的 ROS2 接口
6. **calibration / tools 补全**
   - 视实际需要继续把 calibration 工具、tools/bag_to_csv 与新的物理参数对齐
   - 在物理常数变更后,重新跑决策/解算单元测试与 ROS2 集成测试
7. **质量门**
   - 新增回归用例覆盖 `target_decision` 多种模式、`valid=false` 行为、ball 输入边界
   - 在 CI 上固化 `colcon build`、`colcon test`、`colcon test-result --verbose`

---

## 12. 不确定项 / 已知边界

- 本 README 基于一次性只读扫描;如果之后修改了固定默认值或包结构,请同步更新对应章节
- solver 的最终 launch 入口已收敛为 `bringup/launch/solver.launch.py`;`hope_solver.launch.py` 已删除
- `calibration`、`tools` 的运行时调用入口暂未在 `bringup` 中串联,后续接入时需要回填第 11 节
- 测试统计 `15 tests` 仅覆盖 `decision` 与 `solver`;其他包的测试覆盖情况以各自 `colcon test` 输出为准
