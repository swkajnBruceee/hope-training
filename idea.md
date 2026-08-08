我认真过了一遍。我的判断是：

> **这套 6-stage pipeline 的工程骨架是对的，适合做 fixed-base humanoid ping-pong motion library；但目前 Stage 2/3 的优化目标还稍微“太像动作重定向”，不够“击球任务导向”。**

我大概会给它：

**工程 pipeline：8.5/10**
**retarget 数据生产：8/10**
**乒乓球 impact fidelity：目前约 6/10**

不是要推翻重做。恰恰相反，我觉得你现在应该**保留整个六阶段结构，只改几个关键接口和 quality gate**。

---

# 一、我认为你现在做得([arXiv][1])re → target_npz + target_spec`

彻底拆出来，我非常赞成。

特别是你明确存：

* frame convention
* quaternion convention
* unit
* fps
* robot
* robot_side
* base pose
* active joints
* phase windows
* quality thresholds

这是典型的“以后救自己狗命”的设计 😂

我甚至建议继续把这个思路贯彻到底，加上：

```text
spec_version
source_dataset_version
robot_asset_hash
joint_order_hash
racket_body_name
racket_mount_transform
retarget_code_git_sha
```

尤其是：

```text
racket_mount_transform
```

**一定建议显式保存。**

因为以后换 racket USD、改 wrist adapter、改 body origin，最容易发生的事情就是：

> CSV 看着正常，FK 看着差 3 cm，所有人开始调 optimizer 权重。

结果两天以后发现 mount transform 改过。

---

Stage 2 / Stage 3 分离也对。

你没有直接：

```text
human motion
    ↓
giant nonlinear optimization
    ↓
pray
```

而是：

```text
target
 ↓
per-frame IK feasibility
 ↓
trajectory optimization
```

这非常合理。

近期 humanoid retargeting 工作也反复指出，逐帧几何 IK 本身是高度非凸的，容易受初始化影响，并产生 joint discontinuity、local optimum 等问题；因此，把 IK 当 initializer、后面再做 temporal / physics refinement，比把 frame-wise IK 当最终 motion 更靠谱。([arXiv][2])``text
geometry
dynamics
schema
replay_precheck

````

四层 quality，以及：

```text
bad_source_data
fixed_base_reach_fail
fixed_base_hit_pose_fail
fixed_base_dynamic_fail
schema_fail
fixed_base_pass
````

这个分类我也很喜欢。

**请保留。**

甚至我觉得这是整个 pipeline 里最成熟的一部分。

---

# 二、P0 问题：你缺的不是 velocity direction，而是 impact velocity

这是我认为目前**最重要的逻辑问题**。

你 Stage 3 的 hit residual 是：

```text
position
normal
velocity_direction
tangent
```

我的第一反应是：

> **为什么没有 racket velocity magnitude？**

或者更直接一点：

> **为什么没有完整的 contact-point linear velocity vector？**

乒乓球不是“拍子沿正确方向经过正确位置”就够。

下面两条 trajectory：

```text
trajectory A:
hit pos = identical
normal  = identical
velocity direction = identical
speed = 2 m/s

trajectory B:
hit pos = identical
normal  = identical
velocity direction = identical
speed = 8 m/s
```

按照你现在描述的主要 hit geometry residual，它们可能非常接近。

但击球结果完全是两码事。

机器人乒乓球研究通常直接把 impact 时刻的 racket orientation / velocity 作为关键 action 或规划量；近年的高速击球和 humanoid table-tennis 系统，也明确围绕 striking position、velocity、timing 构造规划器或控制器接口。([arXiv][1])Stage 3 的核心 target 从：

```text
racket trajectory geometry
```

提升成：

```text
impact state
```

即至少：

```text
strike_state = {
    hit_time,
    contact_position,
    racket_orientation,
    contact_linear_velocity,
}
```

更完整一点：

```text
strike_state = {
    t_hit,

    p_contact,
    R_racket,

    v_contact,
    omega_racket,

    ball_v_in,
    ball_v_out,
}
```

注意我这里特意写的是：

```text
v_contact
```

而不一定是：

```text
racket_center_velocity
```

因为严格来说：

[
v_{contact}=v_{origin}+\omega\times r_{contact}
]

如果球不是永远打在 racket origin 上，angular velocity 会影响接触点速度。

### 我会改 residual 为类似：

```python
r_hit = [
    w_pos * position_error,
    w_rot * rotation_error,
    w_vel * velocity_vector_error,
]
```

其中：

```python
velocity_vector_error =
    (v_contact - v_contact_target) / velocity_scale
```

或者为了避免 magnitude 压过方向：

```text
velocity direction error
+
velocity magnitude error
```

比如概念上：

```python
r_vel_dir = 1.0 - dot(normalize(v), normalize(v_target))
r_vel_mag = (norm(v) - norm(v_target)) / v_scale
```

**你现在的 `velocity_direction` 建议不要删，但必须补 magnitude。**

我的评价非常直接：

> **没有 impact velocity magnitude 的 trajectory optimizer，对 table tennis 来说是不完整的。**

SMASH 这类近期 humanoid ping-pong 工作也是只在 strike event 附近重点监督 racket position 和 velocity，并将 task execution 与整体 motion prior 区分开；这其实与你 Stage 3 的 phase-window 思路非常接近，只是它把 racket velocity 明确放在 task-level strike objective 里。([arXiv][3])、P0 问题：Stage 4 的 200 → 50 Hz 很可能损失 hit event

这里我觉得有一个很隐蔽的坑。

你的：

```text
hit window = [-3, +3]
fps = 200
```

所以：

```text
dt = 5 ms
```

hit window 相当于：

```text
-15 ms ~ +15 ms
```

总共大概 30 ms。

现在降到：

```text
50 Hz
dt = 20 ms
```

那么这个关键 hit window，在最终 motion NPZ 里面通常只剩：

```text
1 frame
```

或者：

```text
2 frames
```

而且取决于 resampling phase。

举个极端例子：

```text
200 Hz:

... -15 -10 -5  0 +5 +10 +15 ... ms
                 ↑ hit
```

50 Hz 若刚好：

```text
... -20  0 +20 ...
          ↑
```

还不错。

但若采样格点偏了：

```text
... -10 +10 +30 ...
```

你的真正 hit 就位于两个 reference frame 中间。

## 50 Hz 本身不是错

这里我要非常强调。

已有 humanoid table-tennis 系统确实使用 50 Hz policy 输出 joint position targets；但它的控制器会额外收到 racket target position、racket target velocity 和 time-to-strike。也就是说，**strike event 是显式 command，不完全依赖 reference motion 的离散帧来表达。** ([arXiv][1])正问题是：

> **Stage 4 是否把 hit semantic side-channel 丢掉了。**

从你的描述看：

```text
optimized_motion_npz:
joint_pos
body_pos_w
...
```

我没有看到：

```text
hit_time_s
hit_phase
hit_frame
racket_target_velocity
racket_target_position
```

### 我强烈建议 NPZ 里永久保留：

```text
hit_time_s
hit_frame_source_200hz

hit_frame_50hz
hit_subframe_alpha

strike_contact_pos
strike_racket_quat
strike_racket_normal
strike_racket_tangent

strike_contact_vel
strike_racket_angular_vel

ball_vel_pre_hit
ball_vel_post_hit
```

其中：

```text
hit_subframe_alpha
```

很重要。

例如：

```text
hit occurs at:
frame_50 = 42
alpha = 0.37
```

表示：

```text
t_hit = lerp(t[42], t[43], 0.37)
```

### 另一种简单方案

在 `csv_to_npz` 前做 retiming：

```text
shift trajectory time
```

让：

```text
t_hit % 0.02 == 0
```

也就是保证 hit 精确落在某个 50 Hz frame。

我个人其实建议：

> **两个都做。**

即：

```text
1. retime to nearest 50 Hz strike frame
2. still preserve continuous hit_time
```

这点我会列为 **P0**。

---

# 四、P0 问题：你现在的 dynamics quality 不是真正的 dynamic feasibility

你现在 Stage 3 的 dynamics：

```text
joint velocity
joint acceleration
joint jerk
```

这叫：

> kinematic dynamic proxy

它很好，但它不等于：

> robot dynamically executable

比如一条轨迹：

```text
qdot OK
qddot OK
jerk OK
```

依然可能：

```text
torque saturation
PD tracking lag
large racket velocity error
waist actuator overload
large reaction wrench
self collision
```

特别是乒乓球这种高速 swing。

近期 high-speed robot table-tennis 系统会直接围绕 paddle impact state 做 optimal control；面向更高竞技速度的工作甚至显式考虑关节动力学、delay compensation 以及 ball-racket interaction model。([arXiv][4]) retargeting 的近期工作同样把“纯 geometric retargeting 缺少 physical feasibility”作为核心问题，并通过 physics rollout/refinement 获得更适合 downstream tracking 的 motion。([arXiv][2])问题：

### `csv_to_npz.py` 到底是哪种 replay？

如果是：

```python
robot.set_joint_state(q_ref)
step()
record_body_pose()
```

或者本质上的 state teleport：

> **它完全不能验证 tracking feasibility。**

如果是：

```python
action = q_ref
PD controller
physics.step()
```

然后记录实际：

```text
q_actual
body_actual
```

那就好很多。

## 我建议增加 Stage 3.5

```text
Stage 3.5:
Physics Tracking Validation
```

pipeline 变成：

```text
IK
 ↓
trajectory optimization
 ↓
physics tracking validation
 ↓
motion library packaging
```

需要测：

```text
joint_tracking_rmse
joint_tracking_p90

racket_hit_position_error
racket_hit_normal_error
racket_hit_velocity_error

torque_saturation_ratio
velocity_limit_ratio

self_collision_count

base_drift
foot_slip
```

对于你当前 fixed-base：

```text
base_drift
foot_slip
```

可以暂时不作为主要 gate。

但：

```text
racket_hit_velocity_error
```

我认为必须有。

最终：

```text
optimized_replay_ready
```

不应该仅意味着：

```text
smooth + finite + joint limit OK
```

而应该意味着：

```text
the actual controlled A3 can track the required strike state
```

这两个概念差得非常大。

---

# 五、Stage 2：我会把 IK 改成 “hit-anchored IK”

你现在是：

```text
frame 0 solve
 ↓
warm start
frame 1
 ↓
warm start
frame 2
...
 ↓
hit
```

这个设计正常。

但对乒乓球我不认为它是最佳方案。

因为最重要的 frame 是：

```text
hit frame
```

假如 frame 0 找到了一个不太好的 IK branch：

```text
shoulder branch A
```

warm start 很可能一路：

```text
A → A → A → A
```

然后 hit frame：

```text
position barely reachable
normal bad
joint near limit
```

实际上另一个：

```text
shoulder branch B
```

可能特别适合 hit。

Frame-wise IK 的非凸和 initialization sensitivity 是优化式 retargeting 的已知痛点；warm start 可以提高连续性，但本质上也可能传播一个差的局部解分支。([arXiv][2])

```text
              hit frame
                  ↓
          multi-seed IK solve
                  ↓
        select best hit posture
             ↙          ↘
backward solve          forward solve
hit-1, hit-2...         hit+1, hit+2...
```

即：

```python
solve_hit_frame_multistart()

for t in reversed(pre_hit):
    warm_start = q[t + 1]

for t in post_hit:
    warm_start = q[t - 1]
```

### hit frame 用 8～16 个 seed

例如：

```text
A3 default
forehand prior
backhand prior

previous successful forehand cluster centers
previous successful backhand cluster centers

random small perturbations
```

评分：

```text
hit position error
+ orientation error
+ joint limit margin
+ posture prior
```

我觉得这个改动的收益会很高。

甚至我会说：

> **对于 strike retargeting，hit-first 比 frame-0-first 更符合任务结构。**

---

# 六、Stage 2 的 orientation residual 还需要再想一下

目前：

```text
FK_normal - target_normal
```

normal 只能约束 racket orientation 的两个自由度。

绕 normal 的 rotation：

```text
racket roll
```

是自由的。

数学上：

```text
normal identical
```

并不意味着：

```text
racket orientation identical
```

你 Stage 3 又加了 tangent，这说明你其实已经意识到了这个问题。

那么我会问一句：

> **为什么 IK init 不同样用 tangent？**

我建议 Stage 2：

```text
position
normal
tangent
regularization
```

或者直接：

```text
SO(3) Log error
```

例如：

[
r_R=\log(R_{target}^{T}R_{FK})
]

但这里有个任务设计选择。

### 如果 racket normal 才是 task-critical

那可以：

```text
normal strong
tangent weak
```

比如：

```text
normal_weight = 1.0
tangent_weight = 0.2
```

我反而不推荐无脑 full orientation 强追人类 racket pose。

因为 humanoid morphology 不同。

SMASH 的设计也是将 racket face normal 与 desired strike velocity 做 task-oriented alignment，而不是强行要求所有 racket orientation 自由度全程复现参考动作。([arXiv][3])

```text
normal = task constraint
tangent = weak disambiguation constraint
```

这比：

```text
full quat exact tracking
```

更合理。

---

# 七、Stage 2 的 pass/reject 逻辑有一点不一致

你写的是：

```text
评估：
position error
orientation error
p50/p90
...

pass/reject:
based on hit_position_reject_m
```

这里我觉得不够。

例如：

```text
position error = 5 mm
normal error = 65 deg
```

按照描述：

```text
PASS
```

然后交给 Stage 3。

理论上 Stage 3 可以修。

但这实际上不再是：

```text
IK initialization
```

而是在说：

```text
good luck optimizer
```

😂

我的建议：

Stage 2 不要只有 `pass/reject`。

改为：

```text
ik_status:
    unreachable
    position_reachable
    pose_reachable
    seed_ready
```

例如：

```text
position_reachable:
    hit_pos_error < 0.03

pose_reachable:
    pos OK
    normal_error < 15 deg

seed_ready:
    pose OK
    joint_margin OK
    temporal_jump OK
```

Stage 3 默认只吃：

```text
seed_ready
```

debug 可以：

```text
--include-pose-reachable
```

这样失败分类会干净很多。

---

# 八、7 控制点方案能用，但 hit 周围自由度偏少

你现在：

```text
boundary_start
pre_far       -20
pre_near       -6
hit             0
post_near      +6
post_far       +24
boundary_end
```

200 Hz 下：

```text
-20 = -100 ms
-6  = -30 ms
+6  = +30 ms
+24 = +120 ms
```

这个布局总体其实挺合理。

但有个问题：

> **hit velocity 是 spline derivative。**

你只有：

```text
q_hit
```

没有：

```text
qdot_hit
```

所以 hit speed 是由：

```text
pre_near
hit
post_near
```

共同“间接生成”的。

如果以后加入我前面说的：

```text
strong impact velocity constraint
```

optimizer 很可能发现：

```text
position easy
orientation easy
velocity magnitude hard
```

因为它必须扭三个 control point 才能改 `qdot_hit`。

## 我的建议有两个方案

### 方案 A：最小修改

控制点改成：

```text
boundary_start
pre_far
pre_near
hit_enter       -3
hit              0
hit_exit         +3
post_near
post_far
boundary_end
```

9 points。

在 hit 附近直接增加 knot density。

我觉得这个已经够实用。

### 方案 B：我更喜欢

使用 Hermite-style parameterization：

```text
opt variable:
q_control_points
+
qdot_hit
```

然后显式保证：

```text
q(t_hit) = q_hit
q'(t_hit) = qdot_hit
```

这样：

```text
impact pose
impact velocity
```

就是直接可控变量。

对于乒乓球，这非常漂亮。

---

# 九、`CubicSpline(natural)` 我建议重新审视

这里是一个比较细的点。

SciPy 的 `CubicSpline(..., bc_type="natural")` 意味着：

```text
second derivative at start = 0
second derivative at end = 0
```

**它并不意味着 start/end velocity 为 0。** ([Scipy 文档][5])你的 motion slice boundary 是什么语义？

如果：

```text
boundary_start = normal preparation posture
boundary_end = recovery posture
```

而你期望：

```text
roughly stationary
```

那么 natural boundary 不一定是你想要的。

可能应该：

```python
bc_type=((1, qdot_start), (1, qdot_end))
```

其中 derivative 来自 IK/source。

或者明确 stationary：

```text
clamped
qdot_start = 0
qdot_end = 0
```

当然，`clamped=0` 也不能无脑用。

如果 clip 是从连续 rally 中切出来：

```text
start/end 并不 stationary
```

强制零速度会造成人工 preparation/recovery。

所以我的推荐是：

> **把 boundary velocity 存进 Stage 1 target spec，然后 Stage 3 使用显式 derivative BC。**

即：

```text
boundary_start_qdot_target
boundary_end_qdot_target
```

至少来自 IK finite difference。

比固定 `natural` 更有语义。

---

# 十、Stage 1 的 top-score selection 容易选出“一堆简单球”

这也是数据 pipeline 一个很常见的坑。

当前：

```text
preferred bonus
+ confidence
+ ball-racket distance
+ success
```

然后：

```text
sort
top 10 forehand
top 10 backhand
```

非常容易出现：

```text
10 个 forehand
其实全是差不多的 contact point
差不多的 incoming velocity
差不多的 swing speed
```

也就是：

> quality 很高，coverage 很烂。

而 humanoid ping-pong 数据工作已经明显开始强调 strike-point coverage 和 physically executable motion coverage，而不仅仅是保留最高质量的少数 reference。([arXiv][3])election 分两步

先：

```text
quality filtering
```

例如：

```text
success
confidence
contact plausibility
```

然后：

```text
diversity selection
```

feature 可以是：

```python
[
    hit_pos_x,
    hit_pos_y,
    hit_pos_z,

    ball_vin_x,
    ball_vin_y,
    ball_vin_z,

    racket_speed_at_hit,

    racket_normal_x,
    racket_normal_y,
    racket_normal_z,
]
```

再用：

```text
farthest point sampling
```

或者简单 KMeans。

流程：

```text
quality filter
 ↓
candidate pool
 ↓
diversity sampling
 ↓
per_stroke_target=10
```

如果只有 10 个样本，我宁愿要：

```text
8 个覆盖空间的 good samples
+ 2 个 exceptional quality
```

而不是：

```text
10 个 score top
```

### 我的 hot take

对于 motion library：

> **排序 top-k 往往是 dataset diversity 的天敌。**

它特别喜欢给你选十胞胎。

---

# 十一、我会在 Stage 1 顺手重建 ball impact semantics

你已经有：

```text
ball position
ball velocity
racket pose
racket velocity
hit time
```

我会多做一步：

```text
estimate pre-impact ball state
estimate post-impact ball state
```

保存：

```text
ball_pos_hit
ball_vel_in
ball_vel_out
```

如果有条件：

```text
landing_xy
```

也保存。

为什么？

因为以后你会很想问：

```text
这个 human sample 为什么成功？
```

真正有价值的数据不是：

```text
human racket moved like this
```

而是：

```text
incoming ball state
    ↓
human generated this strike state
    ↓
outgoing ball state
```

即：

```text
(ball_in, strike_state) -> ball_out
```

这实际上已经是一个非常宝贵的 impact dataset。

现代高水平 robot table-tennis 工作越来越重视准确的 ball-racket interaction，特别是速度和旋转提高以后，contact model error 会直接影响 stroke planning 和 sim-to-real。([arXiv][6])spin：

```text
ball_v_in
strike velocity
racket normal
ball_v_out
```

也很值得存。

### 注意

不要在 contact 前后做一个 centered velocity filter。

因为 impact 是 discontinuity。

应该类似：

```text
pre-hit local fit
post-hit local fit
```

分开估计。

HITTER 在球桌 bounce 时会清空历史拟合 buffer，避免跨越离散碰撞事件做平滑；SMASH 的消融也显示，显式处理碰撞不连续性对位置和速度估计很重要。([arXiv][1])*ball-racket impact 前后的 velocity estimation 不要跨 hit 平滑。**

---

# 十二、Stage 5 很好，但 manifest 应该开始承担 dataset QA

现在：

```text
stroke counts
smoke pick
```

我会继续加：

```text
hit_position_range
hit_position_histogram

incoming_ball_speed_range
racket_hit_speed_range

hit_normal_distribution

motion_duration_range

max_joint_vel_distribution
max_joint_acc_distribution

physics_tracking_pass_count
```

manifest summary 最终应该让我一眼看出：

```text
forehand: 57
backhand: 43
```

之外，还能看到：

```text
卧槽，forehand 57 条全在 table center-right
```

这种问题。

按 `episode_id` 去重没问题。

但以后如果 competition data 是 rally 切 episode，我建议再加：

```text
source_rally_id
source_session_id
```

避免 train/test：

```text
episode_001 from rally A → train
episode_002 from rally A → validation
```

造成 near-duplicate leakage。

---

# 十三、Stage 6 的 smoke test 太弱

你现在：

```text
TrackingFlat
num_envs=8
max_iterations=1
```

它验证的是：

```text
motion loads
env creates
policy/training loop can step
```

很好。

但不要把它叫：

```text
trajectory verified
```

😂

我建议三层 smoke：

### Smoke A：schema smoke

```text
num_envs=8
max_iterations=1
```

你现在这个。

### Smoke B：tracking rollout smoke

选：

```text
1 forehand
1 backhand
```

完整 rollout。

测：

```text
joint tracking
racket position at hit
racket velocity at hit
termination
NaN
torque saturation
```

### Smoke C：strike smoke

Isaac 里放一个 ball。

甚至不用完整 perception。

直接：

```text
initialize ball trajectory
replay/task track strike
```

检查：

```text
contact occurred
contact timing error
ball outgoing direction
```

我觉得 **Smoke C 才是这个 pipeline 的最终 truth test**。

因为你这个项目不是：

```text
人形机器人优美挥拍
```

而是：

```text
人形机器人打乒乓球
```

这个区别非常残酷 😂

---

# 如果是我，我会把 pipeline 微调成这样

```text
Stage 1
Competition sample selection
+
Impact semantic reconstruction
    ball_in
    ball_out
    continuous hit_time
    contact state
        ↓

Stage 2
Hit-anchored multi-seed IK
    solve hit first
    backward / forward warm start
        ↓

Stage 3
Strike-state trajectory optimization
    impact position
    impact orientation
    impact velocity vector
    phase corridor
    smoothness
        ↓

Stage 3.5
Closed-loop physics tracking validation
    PD tracking
    actuator limits
    actual racket state at impact
        ↓

Stage 4
Semantic-preserving 200 → 50 Hz packaging
    motion
    hit_time
    hit_subframe
    strike state
        ↓

Stage 5
Motion library merge
+
coverage QA
        ↓

Stage 6
Schema smoke
Tracking smoke
Ball-contact smoke
```

---

# 最终评价

**不用重构六阶段。**

你的大方向是合理的，而且 pipeline engineering 已经比很多“一份 retarget.py 跑天下”的项目成熟多了。

我会按这个优先级改：

> **P0-1：Stage 3 加 racket/contact velocity magnitude 或完整 velocity vector。**
> **P0-2：200→50 Hz 后保留 continuous hit time / hit subframe / strike target。**
> **P0-3：增加 actual closed-loop physics tracking validation。**
> **P1-1：Stage 2 改 hit-first、multi-seed、前后双向 IK。**
> **P1-2：IK pass gate 加 normal/tangent，不只看 position。**
> **P1-3：hit 附近增加控制点或显式优化 qdot_hit。**
> **P1-4：重新考虑 natural BC。**
> **P1-5：Stage 1 从 score top-k 改 quality-filter + diversity sampling。**

**一句话总结：你的“数据工厂”逻辑已经对了；下一步应该把 optimizer 的中心从 `reproduce racket motion` 移到 `reproduce executable strike state`。**

把 `cli_generate_a3_fixed_base_ik_init.py` 和 `cli_optimize_a3_fixed_base_trajectory.py` 的核心函数贴出来或上传后，下一步最值得直接做一次 **逐 residual / 权重 / Jacobian / quality gate 的 code-level review**；我怀疑 Stage 3 里面还能抓出几个比架构问题更具体的坑。

[1]: https://arxiv.org/html/2508.21043v2 "HITTER: A HumanoId Table TEnnis Robot via Hierarchical Planning and Learning"
[2]: https://arxiv.org/html/2603.22201v2 "Make Tracking Easy: Neural Motion Retargeting for Humanoid Whole-body Control"
[3]: https://arxiv.org/html/2604.01158v1 "SMASH: Mastering Scalable Whole-Body Skills for Humanoid Ping-Pong with Egocentric Vision"
[4]: https://arxiv.org/abs/2505.01617?utm_source=chatgpt.com "High Speed Robotic Table Tennis Swinging Using Lightweight Hardware with Model Predictive Control"
[5]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html?utm_source=chatgpt.com "CubicSpline — SciPy v1.18.0 Manual"
[6]: https://arxiv.org/abs/2606.28805?utm_source=chatgpt.com "Physics Models for Sim-to-Real Transfer in Professional-Level Robot Table Tennis"
