# P5U Phase Unified Upper Tracker

状态：正式训练完成，教师资格未通过（2026-08-04）  
来源：`/home/bistu/下载/P5U统一上身动态跟踪策略.docx`  
来源 SHA-256：`72e1d56d38d99e897d91f89315331f253a2042df58e6f7e32b2c21190bed6a6b`

## 当前训练策略（已停止）

之前运行的是 Contract A 的早期版本：floating-base、16 条 train reference、冻结
`model_3396` 下肢支持、禁用 `model_900`，actor 输出 10D upper position residual，命令为
`q_ref + residual`。它只有关节预览和旧的 strike reward，没有连续 phase action、phase
插值、速度方向/有符号速度/pass-through/stop/reverse 联合奖励，也没有完整 reference
TCP preview。因此不能作为正式 P5U 策略继续训练。

## 正式链路

```text
canonical_goal_10d
  -> verified candidate reference
  -> current/future q,dq,TCP reference
  -> P5U actor
  -> continuous phase-adjusted reference + position residual
  -> safety filter
  -> PhysX actual
```

`canonical_goal_10d` 永远是唯一任务真值；不允许改目标、使用 actual relabel、输入
motion/reference/seed ID，`model_3396` 冻结，`model_900` 不进入正式运行链。

## Observation 合同

Actor 使用实际 q/dq、floating-base 状态、足底/contact、上一动作；当前 q_ref/dq_ref
及 q/dq 误差；+1/+3/+6/+12 的 q、dq、TCP position/normal/velocity；time-to-hit、
marked hit step、phase sin/cos；canonical target 与 actual/reference TCP 误差。新增
reference TCP 使用 MotionCommand candidate payload 的 wrist body 和同一 racket mount
FK，不复制 source motion。

## Action 合同消融

* Contract A：10D position residual，`q_cmd=q_ref+scale*residual`。
* Contract B：10D residual + 1 global continuous phase offset，权限 ±4 steps。
* Contract C：10D residual + global/shoulder/elbow/wrist phase，权限 ±4/±2/±2/±2 steps。

phase 作用于连续 reference 插值，不跳整数帧；相位使用低通、幅度/速率/组间一致性正则。
初始 actor mean 严格为零，A/B/C 都不能加载历史 actor。

## Reward 合同

全周期同时使用 q/dq/TCP reference tracking、phase consistency、稳定和安全项。
marked hit ±3 control steps 内单独计算 canonical position、normal、速度大小、速度方向、
有符号目标方向速度、timing kernel 和 pass-through；到点停车与反向运动分别惩罚。
联合 strike success 必须同时通过位置、法向、速度方向、速度大小和 timing。

## 训练顺序

1. payload/checkpoint/action/observation 静态合同与多 reference correctness；
2. 64–128 env、10–20 iteration smoke；
3. A/B/C action short ablation（同 seed、同 train/validation、300–500 iterations）；
4. 选 validation marked-hit canonical 指标最好的合同；
5. reward 因果消融；
6. 严格零初始化、4096 env、2000 iterations 正式训练；
7. paired PhysX 选择 validation 最佳 checkpoint，最后只评估一次 holdout。

任何 reward 上升而 canonical position/normal/velocity/timing 不改善的合同立即停止，
不得只增加训练轮数。

## 已实现的代码入口

* `A3UnifiedUpperReferenceTrackerAction`：A/B/C 共用，连续 phase 插值与低通；
* `A3FloatingUnifiedUpperReferenceTrackerEnvCfg`：A；
* `A3FloatingUnifiedUpperReferenceTrackerGlobalPhaseEnvCfg`：B；
* `A3FloatingUnifiedUpperReferenceTrackerGroupedPhaseEnvCfg`：C；
* `hope_observations.py`：phase、误差和 candidate reference TCP 观测；
* `hope_rewards.py`：canonical velocity 三项、timing、pass-through、stop/reverse、phase 正则；
* `cfg/task/HOPEA3FloatingUnifiedUpperReferenceTracker*.yaml`：A/B/C 训练入口。

已完成：A/B/C 1024×300 action 消融、R0/R1/R2/R3 1024×200 reward 消融、A+R2
4096×2000 严格零初始化正式训练，以及 validation checkpoint 选择和一次性 holdout replay。
结果见 `eval_outputs/p5u1_action_reward_ablation_v1.json` 与
`eval_outputs/p5u1_formal_selection_v1.json`。Contract A、Reward R2、model_1000
为 validation 综合最佳，但 validation/holdout composite canonical strike pass rate 均为
0，因此当前不得标记 `QUALIFIED_TEACHER`，也不得继续扩大数据或蒸馏高层策略。
checkpoint round-trip 证据见 `eval_outputs/p5u1_checkpoint_roundtrip_v1.json`；命中窗口
phase-lag 诊断见 `eval_outputs/p5u1_phase_lag_scan_model1000_v1.json`。后者只比较
reference 与 processed command，不把命令相位诊断误报为 PhysX actual 资格。
