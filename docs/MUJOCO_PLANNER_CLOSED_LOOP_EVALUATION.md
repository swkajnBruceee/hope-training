# MuJoCo + Planner 闭环评测说明

本文档固定 HOPE 策略在本项目中的真实闭环 MuJoCo 评测流程。评测链路使用
AimRT MuJoCo、HOPE ROS 2 planner、Gate3 物理球和 native C++ runner，策略通过
与真实部署一致的 110D observation / 31D action 接口运行。

## 1. 一键启动入口

统一入口为：

```text
scripts/run_mujoco_planner_motion.sh
```

脚本会自动完成：

```text
AimRT MuJoCo
    ↓
native runner（planner + policy-native）
    ↓
schema-2 base-pose relay
    ↓
HOPE planner
    ↓
Gate3 物理球
    ↓
MOTION level=0 准备态
    ↓
连续来球与策略击球
```

不需要手动分别启动 MuJoCo、planner 或 runner。脚本会清理本项目启动的旧进程，
并在退出时关闭本次链路。

当前状态：单球官方闭环已经实际确认 `RACKET CONTACT`；随机多球模式已经完成
发球与连续 planner 压力链路，但尚未被认证为“十球全部成功”的正式成功率协议。
多球结果必须逐球检查 `launch`、`planner engage` 和 `RACKET CONTACT`，不能只看到
脚本正常结束就判定十球成功。

## 2. 当前已经验证的启动顺序

启动顺序必须保持如下：

1. 启动 AimRT MuJoCo，使用 1 ms 物理步长和显式 body-drive PD；
2. 重置机器人到指定站位；
3. 启动 native runner，先通过 `PD_STAND` 完成静态稳定；
4. 启动 base-pose relay 和 planner；
5. 让 runner 进入 `MOTION level=0`；
6. 确认机器人保持 upright 且站位稳定；
7. 只有在策略准备完成后才启动 Gate3 物理球；
8. planner 发布 schema-2 racket command；
9. runner 根据 ball clock 从 level 0 进入 level 1 swing；
10. MuJoCo 记录球拍与球的真实碰撞。

`PD_STAND` 只是启动和重置阶段的静态保持，不是训练策略的准备动作。真正与官方
策略一致的准备态是 `MOTION level=0`。

## 3. 坐标契约

当前闭环中必须保持以下坐标转换，不可重复或遗漏：

| 数据 | 坐标约定 |
|---|---|
| MuJoCo pelvis / ball | floor-origin，地面为 z=0 |
| planner `/poses` | table-origin，桌面为 z=0 |
| planner 内部预测 | table-origin |
| `/racket/command_flat` | policy/floor-origin |
| `/a3/base_pose_flat` | policy/floor-origin |

具体转换为：

```text
Gate3 ball floor z → planner /poses z = floor_z - 0.76
planner intercept table z → racket_flat z = table_z + 0.76
MuJoCo pelvis floor z → base_flat z，不再额外加 0.76
```

因此启动脚本中：

```text
gate3_state_to_poses: table_surface_z=0.76
base relay: policy_z_offset=0.0
planner: policy_z_offset=0.76
```

如果 planner 的 `policy_z_offset` 错设为 `0.0`，击球目标会整体低约 0.76 m，
策略会收到越界的 racket target，可能表现为不击球、异常动作或机器人失稳。

## 4. 官方基线评测

从项目根目录执行：

```bash
cd /home/bistu/桌面/HOPETableTennis

bash scripts/run_mujoco_planner_motion.sh \
  --duration 45 \
  --shots 10 \
  --randomize \
  --seed 20260814 \
  --flight-window 2.5 \
  --inter-shot 0.8 \
  --stand-x -0.65 \
  --stand-y -0.565 \
  --policy-dir /home/bistu/桌面/HOPETableTennis/a3_deploy/a3_deploy_example/models/model_21800/policy \
  --label official_random10
```

官方模型目录：

```text
a3_deploy/a3_deploy_example/models/model_21800/policy
```

## 5. Residual 模型评测

当前最新正式训练的 `model_2999` 已导出为 native runner 可直接使用的 policy bundle：

```bash
cd /home/bistu/桌面/HOPETableTennis

bash scripts/run_mujoco_planner_motion.sh \
  --duration 45 \
  --shots 10 \
  --randomize \
  --seed 20260814 \
  --flight-window 2.5 \
  --inter-shot 0.8 \
  --stand-x -0.65 \
  --stand-y -0.565 \
  --policy-dir /home/bistu/桌面/HOPETableTennis/a3_deploy/a3_deploy_example/models/model_residual_2999/policy \
  --label residual_random10
```

Residual bundle：

```text
a3_deploy/a3_deploy_example/models/model_residual_2999/policy
```

该目录必须同时包含：

```text
exported/policy.onnx
exported/policy_manifest.json
params/deploy.yaml
```

native runner 不直接读取 Isaac Lab 的 `.pt` checkpoint；`.pt` 必须先导出为上述
policy bundle。

## 6. 随机十球协议

使用：

```text
--shots 10 --randomize --seed 20260814
```

随机序列由 `gate3_ball_launcher.py` 根据 seed 可复现生成。当前随机测试固定在已验证的
反手可达走廊；每个候选在发布前都会做物理预检，不合格候选会被拒绝并重新采样。若
运行环境安装了 Python `mujoco`，预检直接加载当前 AimRT MuJoCo 的 `a3_pingpong.xml`；
正式 `hope-ros` 环境没有该 Python wheel 时，启动器使用同一几何、时间步长和重力包络
的保守纯 Python 后备预检。运行时球物理始终由 AimRT MuJoCo 负责。预检要求：

* 初始位置位于官方 opponent-half 球台范围内；
* 来球从正向负 X 运动，穿越球网时球心高于网顶加球半径，并额外保留 30 mm 净空；
* 在当前反手击球平面到达，时间约为 0.50–0.72 s，Y/Z 位于击球走廊；
* 到达前发生近侧台面接触且没有撞网；
* 发射和击球速度均位于官方 1–7 m/s 物理包络内；
* 轨迹最高点不超过 1.65 m，避免生成明显高过机器人头部的来球。

当前通过预检后实际使用的随机化范围为：

```text
x0: 2.64 ~ 2.72 m（官方 opponent-half 球台范围内）
y0: -0.54 ~ -0.50 m（反手区域）
z0: 1.14 ~ 1.18 m（floor-origin）
vx: -5.05 ~ -4.95 m/s
vy: -0.22 ~ -0.18 m/s
vz:  0.97 ~ 1.03 m/s
```

启动日志中的 `[gate3 preflight]` 行会逐球打印预检模式、过网高度、击球时间、击球
位置、击球速度和轨迹最高点；只有全部检查通过的十个球才会进入 ROS 发送队列。预检
不会改变运行时的球物理或 planner。

检测到球拍接触后，发射器默认保持当前球 1.5 s，用于观察击球后的过网和对方台面事件，
再进入下一球；该值可通过 `--contact-hold` 调整。如果没有检测到接触，则使用
`--flight-window` 超时后继续。整个多球过程不重置机器人、runner 或 planner 的策略历史。

`--contact-hold` 不能设得过短。过短时仍可统计球拍碰撞，但球会在官方
`post_racket_table_events` 产生前被停到场外，导致合法回球率被系统性低估。

这组随机球首先用于检查连续生命周期、状态恢复和球路输入链路。当前实验中如果
出现 planner 预测高度异常、`telemetry-not-met`、fall guard 或只有前一球接触，
应把它判为多球协议未通过，而不是归因于模型成功率。

## 6.1 正手/反手混合随机协议

当前项目新增：

```bash
--mixed-random --seed 20260814 --stand-y -0.7625
```

它复用官方 V17-R10 的 side-neutral 原则，但按照当前 `hitter_pure` 的
floor-origin 坐标和约 0.55 s 击球时间重新生成来球。发球命令不携带 side 标签，
由 planner 根据实测轨迹和机器人站位自行选择正手或反手。

100 球生成协议严格成对采样：

```text
forehand lane：世界 y 约 -1.22 ~ -1.18 m
backhand lane：世界 y 约 -0.87 ~ -0.83 m
```

每个候选仍必须通过过网、近侧台面接触、击球时间、击球高度、速度和最高点预检；
预检失败会在同一 side lane 内重新采样。100 球默认得到 50 个正手 lane 和 50 个
反手 lane，实际 planner side 以 `runner.log` 中的 `[pp engage]` 为最终记录。

混合 100 球命令示例：

```bash
cd /home/bistu/桌面/HOPETableTennis
bash scripts/run_mujoco_planner_motion.sh \
  --duration 360 \
  --shots 100 \
  --mixed-random \
  --seed 20260814 \
  --flight-window 2.5 \
  --contact-hold 1.5 \
  --inter-shot 0.8 \
  --stand-x -0.65 \
  --stand-y -0.7625 \
  --policy-dir /home/bistu/桌面/HOPETableTennis/a3_deploy/a3_deploy_example/models/model_a5_2800/policy \
  --label model_a5_2800_mixed_random100
```

原来的 `--randomize` 仍保留，表示旧的单侧反手随机协议；它不能与
`--mixed-random` 混用。

## 7. 成功判据

单球闭环至少应在日志中出现：

```text
ENTER_MOTION result=APPLIED
mode=MOTION level=0 ... gravZ=-1.00
planner_valid ...
[pp engage]
RACKET CONTACT shot=N count=M
```

其中最重要的是：

```text
RACKET CONTACT
```

它来自 MuJoCo Gate3 物理碰撞状态，不是 planner 的预测命令，也不是训练 reward。
因此它可以证明“球真正接触了球拍”。

但是，`RACKET CONTACT` 不等于合法回球。当前证据记录器在球拍接触后继续保留球体，
因此可以进一步统计击球后的过网和对方台面落点。完整评估应统计：

```text
strike attempts
virtual hits
net clears
legal landings
landing error
reset rate
```

当前一键脚本主要固化“真实闭环接触”和策略执行链路；合法落点统计应从后续球轨迹/落点
审计中读取，不能仅凭 planner command 判断。

### 7.1 官方 Gate3 物理证据审计

当前项目已经迁移并接入官方 Gate3 物理证据记录器：

```text
a3_deploy/a3_deploy_example/scripts/pp_gate3_ball_evidence.py
```

一键脚本会在发球前启动记录器，在退出前停止记录器并生成：

```text
physical_evidence.json
physical_evidence.log
closed_loop_audit.json
```

`physical_evidence.json` 是逐球的物理真值来源，检查 1 kHz Gate3 球状态中的：

* incoming table bounce；
* racket contact edge；
* outgoing net/table events；
* sample gap、counter monotonicity 和 shot_id 完整性。

`closed_loop_audit.json` 只负责把它和 launcher、planner、runner、AimRT 日志连接起来，
不会把 preflight、planner engage 或 command 发布误报成击球成功。手动复核命令：

```bash
python a3_deploy/a3_deploy_example/scripts/pp_closed_loop_audit.py \
  --log-dir logs/mujoco_planner_motion/<本次日志目录> \
  --physical-evidence logs/mujoco_planner_motion/<本次日志目录>/physical_evidence.json \
  --output logs/mujoco_planner_motion/<本次日志目录>/closed_loop_audit.json
```

官方的 `pp_gate3_core.py`、`pp_gate3_ball_evidence.py`、`pp_rally_conductor.py`、
`pp_rally_report.py`、`pp_mujoco_plant_report.py` 和 `pp_planner_envelope_audit.py` 均已迁移到
当前项目的 `a3_deploy/a3_deploy_example/scripts/`。其中当前一键闭环已经自动接入
`pp_gate3_ball_evidence.py` 和本地 join adapter；`pp_rally_conductor.py`、plant report
和 obs report 仍需 native runner 提供官方格式的 trace/obs CSV 后才能自动生成，不应在
缺少这些输入时伪造完整 rally 结论。

当前 smoke test 已验证：物理证据能够完整闭合 shot 生命周期。一次 residual 模型运行
中，`incoming_bounce_pass=1`、`telemetry_complete=1`、`racket_contact_count=0`，因此
审计结果明确为“来球有效但本次未接触”，不是“记录器失效”。

## 8. 日志位置与检查命令

每次运行产生独立日志目录：

```text
logs/mujoco_planner_motion/<label>_<timestamp>/
```

主要文件：

| 文件 | 用途 |
|---|---|
| `runner.log` | native runner 状态、planner engage、策略 level |
| `planner.log` | planner 输入频率、有效命令、预测失败原因 |
| `gate3_launcher.log` | 发球参数与 `RACKET CONTACT` |
| `gate3_state_to_poses.log` | 球坐标桥接状态 |
| `base_relay.log` | schema-2 base pose 发布/拒绝统计 |
| `mujoco_plant.csv` | 1 ms MuJoCo plant 的关节、球拍、球和接触诊断 |
| `serve_sequence.txt` | 本次随机或固定发球序列 |
| `physical_evidence.json` | 官方 Gate3 逐球物理证据与 fail-closed verdict |
| `physical_evidence.log` | 物理证据记录器运行日志 |
| `closed_loop_audit.json` | launcher/planner/runner/AimRT 与物理证据的机器可读汇总 |

快速查看击球结果：

```bash
rg -n "launch shot|RACKET CONTACT|\[pp engage\]|mode=MOTION level" \
  logs/mujoco_planner_motion/<本次日志目录>/
```

## 9. 常见问题

### 没有 `RACKET CONTACT`

按以下顺序检查：

1. 是否出现 `mode=MOTION level=0`；
2. 是否在该日志之后才出现 `launch shot`；
3. 是否有 `planner_valid`；
4. planner 目标 z 是否约为 `0.85~1.30` 的 floor-origin 高度；
5. 是否有 `[pp engage]`；
6. runner 是否出现 fall guard、command fault 或 stale target；
7. `mujoco_plant.csv` 是否确认球和球拍碰撞 geom 已启用。

### planner 有效但 runner 不 engage

重点查看：

```text
time_to_strike
station readiness
target gate
base pose schema=2
```

planner 命令太晚、base pose 不新鲜、站位不在 readiness 范围，都会导致 planner
有输出但 runner 不释放 swing。

### 机器人弹飞或突然失稳

优先检查：

```text
planner policy_z_offset 是否为 0.76
base relay policy_z_offset 是否为 0.0
是否先进入 MOTION level=0
是否仍在使用 explicit PD
```

不要先修改模型或增加 residual；先恢复上述控制契约。

### 终止时出现 `ExternalShutdownException`

如果该异常只出现在脚本退出清理阶段，而之前已经出现 `RACKET CONTACT`，它通常是
ROS 进程被统一关闭时的退出噪声，不代表本次击球失败。真正的评测结果以运行期间的
contact、engage、plant CSV 和 planner 日志为准。

## 10. 固化原则

以后比较官方模型和 Residual 模型时，必须保持以下配置一致：

```text
MuJoCo XML
PD mode
robot station
planner config
policy_z_offset
ball physics
random seed
shots
flight window
inter-shot interval
评测日志与判定口径
```

只替换：

```text
--policy-dir
```

这样才能把差异归因到模型，而不是闭环启动方式或球路配置。
