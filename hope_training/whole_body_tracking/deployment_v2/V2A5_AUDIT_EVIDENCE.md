# V2-A5 HOPE Open-Source Canonical Contract

## 1 Executive conclusion

V2 第一版纯软件合同 prototype 已建立，且全部单元测试通过。它保持最终 HOPE deployment、model_21800、110D→31D contract 和既有 planner 不变；新增内容仅位于 `hope_training/whole_body_tracking/deployment_v2/`。

```text
HOPE_CANONICAL_SOURCE_ONLY=TRUE
MODEL21800_IDENTITY=PASS
OBS_DIM=7
ACTION_DIM=3
V2_SOFTWARE_CONTRACT_READY_TO_FREEZE=TRUE
```

这里的“可冻结”仅指 **HOPE_OPEN_SOURCE_SOFTWARE_CONTRACT v1**。它不是已训练策略、硬件 TCP 标定、物理 mocap/table 标定、sim2real 或真机验证。`STATUS=PROTOTYPE_NOT_TRAINED`。

## 2 Canonical source files

Canonical deployment evidence:

- `hope-deploy-baseline/a3_deploy/a3_deploy_example/models/model_21800/policy/exported/policy.onnx`：SHA256 `6bf1a2418f8538e23577a0153f2fe6a1e78dee91f41650a232259432a84a4dc8`；实际 metadata 包含 position、CORE velocity、PLANNER velocity per-clip boxes。
- `hope-deploy-baseline/.../include/a3_pingpong/pp_planner_input.hpp:24-35,236-305`：Schema-2 19D layout、validity、absolute wall deadline 与 identity parser。
- `hope-deploy-baseline/.../include/a3_pingpong/pp_policy.hpp:651-688`：从 ONNX position boxes 的 x/y 中心构造 `reach_offset_clip`；从 metadata 读取独立 CORE/PLANNER components。
- `hope-deploy-baseline/.../include/a3_pingpong/pp_policy.hpp:3034-3093`：frame 0/1、nearest-station side、tie→FH、explicit side semantics。
- `hope-deploy-baseline/.../include/a3_pingpong/pp_policy.hpp:3244-3290,3770-3780`：native support 是 CORE OR PLANNER，不是 bounding union。
- `hope-deploy-baseline/.../include/a3_pingpong/pp_reference_clock.hpp:51-52`：`+1→clip0`, `-1→clip1`。

Canonical planner evidence:

- `hope-model21800-isaac/hope_ws/src/msgs/msg/PredictedStrike.msg:1-9`。
- `hope-model21800-isaac/hope_ws/src/msgs/msg/RacketCommand.msg:1-12`。
- `hope-model21800-isaac/hope_ws/src/trajectory/src/strike_prediction_node.cpp:126-151,317-320`：预测 position/velocity/strike time 与 source TTS。
- `hope-model21800-isaac/hope_ws/src/solver/src/solver_node.cpp:202-240,300-320`：structured command mapping/header forwarding。
- `hope-model21800-isaac/hope_ws/src/solver/src/hit_plan_solver.cpp:30-68`：`plan.p_hit=strike.p_ball` 和 ideal racket velocity lineage。
- `hope-model21800-isaac/hope_ws/src/solver/src/racket_target_solver.cpp:122-137`：collision velocity formula。

Prototype 独立用最小 protobuf reader 读取 ONNX ModelProto `metadata_props`，不依赖缺失的 `onnx` Python package；然后同时校验 SHA 和 exact metadata。实现：`deployment_v2/hope_open_source_contract.py`。

## 3 Observation contract

```text
[
  hit_y_world,
  hit_z_world,
  incoming_vx_world,
  incoming_vy_world,
  incoming_vz_world,
  control_tts,
  swing_sign,
]
```

来源：`RacketCommand.position.y/z`、`ball_velocity_incoming.xyz`、aged `time_to_strike`、adapter nearest-station flight lock。SAC V2 是 strike planner，不重学 trajectory prediction；因此不加入固定-plane x、phase/bounce、solver nominal velocity 或完整 base/station state。

明确排除 V1 launch→hit-plane physical flight time、Isaac ground truth 和 future oracle。

## 4 Action contract

Action 是 normalized `float[3] ∈ [-1,1]^3`：

```text
[racket_vx_world, racket_vy_world, racket_vz_world]
physical = low(side) + 0.5*(normalized+1)*(high(side)-low(side))
```

实际 ONNX `hitter_pure_vel_planner_range_per_clip`：

| side | clip | vx | vy | vz |
|---|---:|---|---|---|---|
| forehand `+1` | 0 | `[1.57,2.55]` | `[0.10,0.52]` | `[0.41,1.35]` |
| backhand `-1` | 1 | `[1.55,2.52]` | `[-0.18,0.29]` | `[0.40,1.32]` |

这与预期 identity check 完全一致。训练 action 只使用 PLANNER box，不使用 CORE、±0.30 runtime margin 或 CORE+PLANNER bounding union。实现：`deployment_v2/hope_open_source_contract.py` 的 `map_normalized_velocity()` 与 `velocity_inside_planner_box()`。

## 5 Position software contract

```text
POSITION_CONTRACT=HOPE_OPEN_SOURCE_SOLVER_POSITION
```

Nominal target position 等于当前 HOPE open-source solver `/racket/command.position`，而 solver 当前复制 predicted ball centre：`hit_plan_solver.cpp:30-40`; `solver_node.cpp:306-314`。

Prototype 不加入 17 mm proxy、ball radius、paddle thickness 或 TCP offset。

This reproduces the current HOPE open-source software pipeline. It is not evidence of a validated real-hardware racket TCP calibration.

## 6 Frame software contract

```text
FRAME_CONTRACT=HOPE_OPEN_SOURCE_WORLD_TABLE_FRAME_CODE_0
```

输入是 world-labelled `RacketCommand`，Schema-2 固定 `[11]=0`；不接受 base-link action。该定义复现开源软件 frame，不证明物理 mocap/table calibration。

## 7 Timing contract

```text
age=max(0, source_now-header_stamp)
control_tts=command_tts-age
absolute_strike_wall=producer_wall+control_tts
```

非有限、不可比较 clock 或 `control_tts<=0` fail closed。producer sec/nsec 以最近纳秒编码 producer wall；builder 校验 absolute deadline 与 encoded producer+TTS 的误差不超过 `1e-9`。native runner 收包后继续按既有代码把 wall deadline 一次转换到 local monotonic countdown。

这是软件 timing contract，不声称跨主机硬件时钟同步已经验证。实现：`deployment_v2/schema2_adapter.py` 的 `age_command_timing()`、`reanchor_to_wall()`。

## 8 Side/station contract

从实际 ONNX position box 中心得到：

```text
reach_offset_clip[0]=(0.58,-0.44)
reach_offset_clip[1]=(0.58,-0.09)
```

新 flight：anchor 为 adapter held station，否则 current base xy；分别计算 `target_xy-reach_offset_clip[c]`，距离较小者胜；精确 tie 选 FH。后续同 flight revision 锁 side，不允许切换。candidate 通过 adapter gate 后才更新 held station。

```text
SIDE_CONTRACT=NEAREST_STATION_THEN_FLIGHT_LOCK
ADAPTER_STATION_MIRROR=PROTOTYPE_REQUIRES_NATIVE_PARITY_TEST
```

实现：`deployment_v2/hope_open_source_contract.py::select_nearest_station_side()` 与 `deployment_v2/schema2_adapter.py::AdapterStationMirror`。

## 9 Flight/revision lifecycle

`FlightRevisionManager` 维护 `next_flight_id, active_flight_id, revision_id, active_strike_time, consumed, expired, command_seq`。它依据 absolute strike-time shot-reuse tolerance、新 flight/consumed/expired 状态分界，不把每个 invalid→valid 当新 flight。

保证：所有 identity 正数；command sequence 严格递增；同-flight revision 严格递增；duplicate/reordered sequence 或 revision 被拒绝。

状态仍为 `POSSIBLE_WITH_RULE`，不是物理 `SAFE` 声明；下一阶段需要 predictor/ROS trace 验证 tolerance 与 flight boundary。

## 10 Schema-2 mapping

`build_schema2_packet()` 输出 `numpy.float64 shape (19,)`：

| Index | Mapping |
|---:|---|
| 0 | `2` |
| 1 | valid |
| 2 | locked swing sign |
| 3..5 | HOPE solver nominal position |
| 6..8 | V2 physical velocity in side PLANNER box |
| 9 | aged control TTS |
| 10 | re-anchored absolute wall deadline |
| 11 | `0` world/table software frame |
| 12..13 | producer wall sec/nsec |
| 14 | strictly increasing command sequence |
| 15 | adapter flight id |
| 16 | same-flight revision id |
| 17 | `0` |
| 18 | `0.0` |

Estimator zero values are permitted by native parser and absent from actor/control.

## 11 Native-support parity

`velocity_inside_native_component_support()` implements two separate membership tests:

```text
(velocity ∈ CORE±margin) OR (velocity ∈ PLANNER±margin)
```

它没有构造 bounding union。测试点 FH `(1.30,0.20,0.45)` 同时组合 core-only x 与 planner-only z，落在外部 bounding union 中但不在任何零 margin component，正确返回 `False`。

部署 reference parser `reference/a3_deploy_onnx_ref_pingpong/ros_command_source.py:63-108` 成功解析 prototype packet 的 valid/side/position/velocity/TTS 与 flight/revision。Coverage limit：它明确忽略 frame code、producer sec/nsec、command_seq、absolute strike wall、estimator count/span；完整 validity 仍以 C++ `pp_planner_input.hpp` 为 canonical。

## 12 Test results

命令（关闭 cache 与 bytecode，未启动 Isaac）：

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest hope_training/whole_body_tracking/deployment_v2/tests -q -p no:cacheprovider
```

结果：

```text
.............. [100%]
14 passed in 0.34s
```

14 个 pytest cases（含参数化/批量断言）覆盖全部要求：SHA identity；exact metadata；FH/BH low/high/center；每侧 10,000 random actions；不使用 margin；FH/BH/tie side；flight side lock；revision monotonic；duplicate/reorder rejection；zero estimator metadata；shape/dtype/indices；timing aging/expiry/wall invariant；NaN/Inf；CORE OR PLANNER non-union corner；reference parser head parity。

未运行 C++ build、ROS2、Isaac、optimizer、backward 或训练。

## 13 Remaining engineering validation

1. 用相同 command trace 对比 Python station mirror 与 native runner pending/held station；当前明确标为 parity-test required。
2. 用 ROS recorded trace 验证 strike-time tolerance、pre/post-bounce topic takeover、consumed/expired flight boundary。
3. 验证 source header clock 可比较性及 HDU/MDU wall synchronization；不改变当前 fail-closed 软件 contract。
4. 未来若进入真机资格阶段，另行验证硬件 TCP 与 mocap/table calibration；这些不属于本软件合同的事实声明。
5. 尚无 SAC V2 网络、checkpoint、Isaac environment integration 或 trained-policy result。

## 14 Freeze recommendation

建议冻结本目录作为 **Deployment-Aligned High-Level SAC V2 — HOPE Open-Source Contract v1** 的 prototype 软件边界，因为：canonical model identity/metadata 已实读；action 全域天然位于 per-side PLANNER box；side/timing/Schema-2/native component semantics 有确定实现；全部测试 PASS。

```text
V2_SOFTWARE_CONTRACT_READY_TO_FREEZE=TRUE
CONTRACT_SCOPE=SOFTWARE_ONLY
PROTOTYPE_NOT_TRAINED=TRUE
REAL_HARDWARE_CALIBRATION_VALIDATED=FALSE
REAL_ROBOT_VALIDATED=FALSE
```
