# HOPE 动捕数据清洗与处理修改计划

本文档是后续处理 `/workspace/DATA260703` 动捕数据的主计划。目标不是只把文件切成片段，而是把 Motive/OptiTrack 导出的原始乒乓数据稳定清洗成可供以下流程使用的统一样本：

```text
动捕清洗样本 -> 击球分析 -> 人体到 A3 重定向/IK -> 轨迹优化 -> TrackingFlat 模仿学习 -> HOPEPingPong 强化学习
```

本文档结合当前仓库、当前数据集、已有分析脚本和 HOPE/A3 训练接口。后续代码改动应优先遵守这里定义的数据契约、模块边界和验收标准。

## 1. 当前数据和项目上下文

### 1.1 数据源

当前数据目录：

```text
/workspace/DATA260703
```

已确认结构：

```text
/workspace/DATA260703
  Bvh/
    Point/
    Rige Body/
  Csv/
    Point/
    Rige Body/
```

其中 `Rige Body` 应理解为 `Rigid Body` 的拼写错误。

已生成的分析产物：

```text
analysis/mocap/DATA260703_analysis.md
analysis/mocap/DATA260703_racket_skeleton_matching.md
analysis/mocap/DATA260703_selected_clip_ranking.md
analysis/mocap/clips/
analysis/mocap/selected_clips/
```

已有探索脚本：

```text
analysis/mocap/analyze_motive_dataset.py
analysis/mocap/cut_bvh_candidates.py
analysis/mocap/match_rackets_to_skeletons.py
analysis/mocap/rank_selected_clips.py
```

这些脚本属于探索/候选片段筛选，不是最终清洗流水线。后续可以复用其中的 Motive 表头解析、刚体列定位、BVH 切片逻辑，但需要重构进正式模块。

### 1.2 数据集事实

基于当前分析：

```text
BVH files: 18
CSV files: 9
Total size: 7.32 GB
FPS: 360 Hz
Length unit: millimeters
Coordinate space: Motive Global
Skeletons per take: Skeleton 001, Skeleton 002
BVH skeleton: 1 root + 50 joints
Racket rigid bodies: TennisBats01, TennisBats02
Extra rigid body in Csv/Rige Body: Tennis
```

当前匹配结论：

```text
TennisBats01 -> Skeleton 001:RHand
TennisBats02 -> Skeleton 002:RHand
```

这个匹配非常稳定。球拍到对应右手的中位距离约 `0.18-0.22 m`，到另一名运动员的手一般超过 `2.7 m`。因此后续可默认右手持拍，但仍需在 quality flags 中记录该匹配置信度。

首批推荐用于清洗和重定向 smoke test 的片段：

```text
analysis/mocap/selected_clips/Table_Tennis_01_005_TennisBats01_90p11_92p11_Skeleton001.bvh
analysis/mocap/selected_clips/Table_Tennis_01_012_TennisBats01_115p69_117p69_Skeleton001.bvh
analysis/mocap/selected_clips/Table_Tennis_01_004_TennisBats02_51p86_53p86_Skeleton002.bvh
```

优先从第一条开始，因为它来自 `Csv/Rige Body/Table Tennis_01_005.csv`，该 CSV 同时包含 `TennisBats01`、`TennisBats02`、`Tennis`，最适合验证球、球拍、人体三者关系。

### 1.3 项目接口

当前 HOPE/A3 训练路径在：

```text
hope_training/whole_body_tracking/
```

训练用 motion 文件不是 Motive CSV/BVH，而是 `scripts/csv_to_npz.py` 输出的 `.npz`。训练 `.npz` 契约见：

```text
docs/interfaces/policy_io.md
```

训练 `.npz` 必须包含：

```text
fps
joint_pos
joint_vel
body_pos_w
body_quat_w
body_lin_vel_w
body_ang_vel_w
```

也就是说，本清洗流水线的输出分两层：

```text
层 1: 乒乓击球清洗样本 sample，用于检测击球、筛选片段、生成标签、指导重定向
层 2: A3 retargeted CSV / A3 motion npz，用于 TrackingFlat / HOPEPingPong 训练
```

本文档主要定义层 1，并规定它如何衔接层 2。

## 2. 总目标

最终从：

```text
Motive CSV/BVH + 球轨迹 + 球拍刚体/marker + 人体骨架/marker
```

清洗成：

```python
sample = {
    "episode_id": "...",
    "source": {...},
    "stroke_type": "forehand/backhand/serve/unknown",

    "time": np.ndarray,              # [T], seconds, relative or absolute
    "ball_pos": np.ndarray,          # [T, 3], meters, canonical frame
    "ball_vel": np.ndarray,          # [T, 3], m/s
    "racket_pos": np.ndarray,        # [T, 3], meters, canonical frame
    "racket_quat": np.ndarray,       # [T, 4], xyzw unless explicitly stated
    "racket_vel": np.ndarray,        # [T, 3], m/s
    "racket_omega": np.ndarray,      # [T, 3], rad/s

    "body": {...},                   # optional human skeleton/body signals
    "markers": {...},                # optional debug marker subset

    "hit_index": int,
    "hit_time": float,
    "hit_pos": np.ndarray,           # [3]
    "racket_pose_at_hit": np.ndarray,
    "racket_vel_at_hit": np.ndarray,

    "pre_hit_window": dict,
    "post_hit_window": dict,

    "ball_in_vel": np.ndarray,
    "ball_out_vel": np.ndarray,
    "landing_pos": np.ndarray | None,
    "success": bool | None,
    "quality_flags": dict,
}
```

第一版必须稳定产出以下字段：

```text
time
ball_pos
ball_vel
racket_pos
racket_quat
racket_vel
hit_index
hit_time
success
quality_flags
```

`body` 和完整 marker 字段可以第二阶段补齐。A3 重定向仍以 BVH 和后续 retargeting 工具为主，清洗样本用于挑选片段、定位击球、生成目标和验证质量。

## 3. 推荐目录和模块设计

建议新增正式清洗包：

```text
analysis/mocap_cleaning/
  __init__.py
  config.py
  schemas.py
  motive_loader.py
  bvh_utils.py
  time_sync.py
  coordinate.py
  rigid_body.py
  marker_reconstruct.py
  filtering.py
  derivative.py
  hit_detection.py
  segmentation.py
  labeling.py
  quality_check.py
  export_dataset.py
  plotting.py
  cli_clean_one.py
  cli_clean_batch.py
```

原因：

```text
analysis/mocap/             保留现有探索脚本和生成结果
analysis/mocap_cleaning/    放正式清洗逻辑
docs/                       放计划和接口说明
hope_training/              暂不直接塞入原始动捕清洗逻辑，避免污染训练入口
```

当清洗流水线稳定后，再考虑把通用模块移动到 `hope_training/whole_body_tracking/scripts/` 或独立 package。

## 4. 总体流水线

正式清洗 pipeline：

```text
load_raw_data
↓
validate_raw
↓
identify_entities
↓
convert_units
↓
build_time_axis / resample
↓
coordinate_transform
↓
repair_short_gaps
↓
remove_outliers
↓
smooth_signals
↓
compute_linear_and_angular_velocity
↓
detect_hit_index
↓
slice_episode
↓
extract_pre_post_features
↓
label_stroke_and_success
↓
compute_quality_flags
↓
export_sample_npz/json/hdf5
↓
plot_debug
```

第一版实施顺序应更保守：

```text
1. Motive CSV loader
2. Tennis/TennisBats rigid body extraction
3. mm -> m
4. hit candidate detection using racket + Tennis
5. sample npz export for one selected clip
6. debug plots
7. batch processing all selected clips
8. BVH clip alignment with detected hit
9. retargeting handoff
```

## 5. 数据契约

### 5.1 RawTrial

正式 loader 应先输出 `RawTrial`，不直接清洗。

```python
RawTrial = {
    "source_path": str,
    "take_name": str,
    "fps": float,
    "time": np.ndarray,              # [T], seconds
    "position_unit": "mm",
    "coordinate_space": "motive_global",
    "rigid_bodies": {
        "TennisBats01": {
            "pos": np.ndarray,       # [T, 3], raw unit
            "quat_xyzw": np.ndarray, # [T, 4]
        },
        "TennisBats02": {...},
        "Tennis": {...},
    },
    "bones": {
        "Skeleton 001:RHand": {"pos": ..., "quat_xyzw": ...},
        "Skeleton 001:Hip": {...},
    },
    "markers": {
        "...": np.ndarray,           # [T, 3], raw unit
    },
    "metadata": dict,
}
```

Motive CSV 当前表头是 8 行：

```text
0: metadata
1: blank
2: Type
3: Name
4: ID
5: Parent
6: Rotation/Position
7: X/Y/Z/W
```

已有探索中曾因为少读第 8 行导致球拍位置列识别失败。正式 loader 必须有测试覆盖这个表头结构。

### 5.2 CleanSample

清洗输出应使用统一单位和坐标：

```text
length: meters
time: seconds
linear velocity: m/s
angular velocity: rad/s
quaternion order: xyzw
canonical frame: hope_table_world first, robot_base later when base calibration is available
```

第一版 canonical frame 建议使用 `hope_table_world`，因为当前数据还没有确认 robot base 的静态/动态变换。等 base_link 或人体 hip 到机器人 root 的标定明确后，再输出额外的 `robot_base` 表达。

### 5.3 Episode ID

建议格式：

```text
DATA260703__Table_Tennis_01_005__TennisBats01__Skeleton001__hit_000
```

字段来源：

```text
dataset_id: DATA260703
take_name: Table Tennis_01_005
racket_id: TennisBats01
skeleton_id: Skeleton001
hit ordinal: hit_000
```

## 6. 坐标系计划

项目约定见：

```text
docs/interfaces/frames.md
```

HOPE table world:

```text
Origin: near-side left corner of table surface
X: toward opponent
Y: left
Z: up
Table surface height: 0.76 m above floor
```

当前 Motive CSV：

```text
Coordinate Space: Global
Length Units: Millimeters
```

正式计划：

```text
Phase A: motive_global_mm -> motive_global_m
Phase B: motive_global_m -> hope_table_world_m
Phase C: hope_table_world_m -> robot_base_m
```

Phase B 需要确定 `T_hope_table_motive`。可能来源：

```text
1. Motive 中的 PPT/table 刚体
2. Csv/Rige Body 中的 Tennis 刚体若不是球，可能是 table/other rigid body，需要验证
3. 人工标定表角和桌面平面
4. 外部标定文件
```

当前数据尚未确认 table rigid body。不能在代码里假设 Motive Global 已经等于 HOPE table world。正式清洗必须在 `quality_flags` 中记录：

```python
"coordinate_transform_available": bool
"coordinate_frame": "motive_global_m" | "hope_table_world_m" | "robot_base_m"
```

如果没有表坐标标定，第一版样本仍可输出，但应标记为：

```python
"usable_for_hit_detection": True
"usable_for_training_absolute_targets": False
```

## 7. 实体识别策略

### 7.1 球拍

当前数据优先使用刚体 pose：

```text
TennisBats01
TennisBats02
```

不需要第一版从 marker 重建球拍 pose。原因：

```text
Motive 已输出 TennisBats01/02 rigid body rotation + position
当前手拍距离匹配稳定
刚体 pose 足够用于击球检测和离线标签
```

但需要保留 marker 重建模块作为 fallback：

```text
marker_reconstruct.py
```

fallback 使用场景：

```text
刚体 pose 丢失
刚体旋转异常
需要重新定义 racket face normal
```

球拍四元数统一为 `xyzw`。Motive CSV 当前刚体旋转列为：

```text
Rotation X, Y, Z, W
```

这正好是 SciPy 默认 `xyzw`，但写入 sample 时必须显式记录：

```python
"quat_order": "xyzw"
```

### 7.2 持拍者

默认映射：

```text
TennisBats01 -> Skeleton 001:RHand
TennisBats02 -> Skeleton 002:RHand
```

正式逻辑不能只硬编码，应实现：

```text
for each racket:
  compute median distance to Skeleton001/002 LHand/RHand in candidate window
  choose nearest hand
  require margin against second-best hand
```

推荐阈值：

```text
best_hand_median_dist < 0.35 m
second_best_same_skeleton_hand - best > 0.10 m
other_skeleton_best - best > 1.0 m
```

质量字段：

```python
"racket_hand_match": {
    "racket": "TennisBats01",
    "skeleton": "Skeleton 001",
    "hand": "RHand",
    "median_distance_m": 0.216,
    "margin_to_other_skeleton_m": 3.0,
    "ok": True,
}
```

### 7.3 球

当前最大未确认点是 `Tennis` 是否代表球。必须先实现验证：

```text
输入: Csv/Rige Body/*.csv
检查: Tennis rigid body position range, speed range, z trajectory, bounce pattern
输出: ball_source = "rigid_body:Tennis" or "unknown"
```

初步判断规则：

```text
若 Tennis 的位置在桌面附近和球场空间内运动，速度可达数 m/s 到数十 m/s，并有抛物线/反弹特征，则作为 ball_pos。
若 Tennis 长时间静止或代表桌/网/其它刚体，则不能作为球。
```

如果 `Tennis` 不是球：

```text
1. 搜索 Marker 中是否存在球 marker
2. 通过高速度小 marker 轨迹聚类识别球
3. 若仍无法识别，则第一版 sample 的 ball_pos 缺失，不能做 hit_index 真检测，只能保留 racket-only candidate
```

`Point` CSV 没有 `Tennis` 刚体，仅有 `TennisBats01/02`。这些文件可以做人体/球拍样本，但不一定能做完整击球样本。

## 8. 缺失值和异常处理

### 8.1 缺失段

规则：

```text
gap <= 30 ms: 可插值
30 ms < gap <= 50 ms: 可插值但 warning
50 ms < gap <= 150 ms: 默认不插值，标记 bad segment
gap > 150 ms: episode 不用于训练
hit 前后 100 ms 内任何关键数据缺失: 丢弃或标记 unusable
```

关键数据：

```text
ball_pos
racket_pos
racket_quat
matched hand pos
hip/body center pos
```

质量字段：

```python
"has_long_missing_gap": bool
"missing_near_hit": bool
"missing_segments": list
```

### 8.2 异常速度阈值

初始阈值：

```text
human hand/bone: 15 m/s
racket rigid body: 25 m/s
ball: 50 m/s
hip/body center: 8 m/s
```

当前探索中出现：

```text
Csv/Rige Body/Table Tennis_01_007.csv TennisBats02 peak 29.20 m/s
```

该点已在候选筛选中跳过。正式清洗中不应简单删除整个 take，而应：

```text
1. 标记该速度跳变帧为 outlier
2. 检查是否为单帧跳变
3. 若短缺失可插值
4. 若发生在 hit 附近则 episode unusable
```

### 8.3 四元数连续性

四元数存在符号翻转等价性。正式处理必须确保相邻帧四元数同半球：

```text
if dot(q[i], q[i-1]) < 0: q[i] = -q[i]
```

否则角速度计算会出现虚假尖峰。

## 9. 滤波和微分

### 9.1 滤波原则

滤波必须晚于单位转换、短缺失修复和异常点处理，早于速度/角速度计算。

第一版建议：

```text
racket_pos: Savitzky-Golay, 15-25 ms
ball_pos: Savitzky-Golay, 9-15 ms
human bones: Savitzky-Golay, 25-35 ms
```

击球前后 `50 ms` 不应过度平滑球轨迹，因为碰撞会造成真实速度突变。可以采用：

```text
全局轻滤波 + 速度变化检测
或 hit detection 前用较轻滤波，hit detection 后再为训练/可视化生成平滑版本
```

### 9.2 速度

线速度使用中心差分：

```text
vel[i] = (pos[i+1] - pos[i-1]) / (time[i+1] - time[i-1])
```

边界用前向/后向差分。

### 9.3 角速度

用相邻四元数相对旋转：

```text
dR = R[i]^-1 * R[i+1]
omega[i] = dR.as_rotvec() / dt
```

输出：

```text
racket_omega: [T, 3], rad/s
```

角速度应做合理性检查：

```text
max_racket_omega < 100 rad/s 初始阈值
```

## 10. 击球检测

击球检测是正式清洗的核心。不能只用球拍速度峰值。

### 10.1 候选信号

必须组合：

```text
ball-racket distance
ball velocity change
racket speed
ball trajectory continuity
optional: ball-racket relative velocity along racket normal
```

第一版综合分：

```text
dist_score = exp(-dist / 0.08)
dv_score = normalized ||ball_vel[i+2] - ball_vel[i-2]||
racket_score = normalized ||racket_vel||
score = 0.5 * dist_score + 0.3 * dv_score + 0.2 * racket_score
valid = dist < 0.15 m
hit_index = argmax(score where valid)
```

若无球轨迹，则：

```text
hit_index = None
fallback_candidate_index = racket_speed_peak
usable_for_hit_training = False
```

### 10.2 击球质量阈值

初始规则：

```text
dist_at_hit < 0.12 m
racket_speed_at_hit > 1.0 m/s
ball_dv_at_hit > 1.0 m/s
missing_near_hit == False
```

这些阈值必须在 debug plots 中验证后再固定。

质量字段：

```python
"hit_detection": {
    "method": "ball_racket_composite",
    "hit_index": int,
    "dist_at_hit_m": float,
    "score_at_hit": float,
    "racket_speed_at_hit_mps": float,
    "ball_dv_at_hit_mps": float,
    "valid_hit": bool,
}
```

## 11. Episode 切分

第一版切分：

```text
pre_hit_time: 0.6 s
post_hit_time: 0.4 s
target_fps: 200 Hz or 100 Hz for clean sample
```

注意：原始数据是 `360 Hz`。清洗 sample 可以保留 360 Hz，也可以重采样。建议：

```text
debug sample: 360 Hz 保留原始信息
learning sample: 200 Hz 或 100 Hz
retargeting BVH: 保留原始或导出 30/50 Hz 版本
```

具体策略：

```text
clean_sample_npz: 200 Hz
debug_npz: 360 Hz
retarget_input_bvh: 原始 360 Hz clip，后续视 GMR/retargeter 要求降采样
A3 csv_to_npz input: 通常 30/50 Hz
```

必须输出：

```python
"pre_hit_window": {"start_index": ..., "end_index": ..., "duration_s": ...}
"post_hit_window": {"start_index": ..., "end_index": ..., "duration_s": ...}
"time_rel": time - hit_time
```

若片段长度不足：

```text
允许 padding，但必须输出 valid_mask
```

第一版可直接丢弃长度不足样本，减少复杂度。

## 12. 成功/失败和落点

成功判断依赖球轨迹和坐标系。如果没有 `hope_table_world`，成功标签不可靠。

第一版标签分级：

```text
success = True/False/None
success_label_reliable = bool
landing_pos = np.ndarray or None
```

当 `T_hope_table_motive` 可用时：

```text
detect landing = z 接近 table_z 且 ball_vel_z 从负变正
success = landing inside opponent table region
```

若没有落点但有击球后速度：

```text
weak_success = ball_out_vel 朝对方方向
success = None
quality_flags["weak_success_signal"] = True
```

不能把弱标签直接当可靠 success。

## 13. Stroke 类型

第一版默认：

```text
stroke_type = unknown
```

可以加规则标签，但必须标记置信度。

右手持拍规则需要坐标系明确。若在 HOPE table/robot frame 下：

```text
右手持拍:
  击球点/球拍在身体右侧并向内挥动 -> forehand
  击球点/球拍在身体左侧并向外/横向挥动 -> backhand
```

若仍在 Motive global 下，左右方向未标定，不能可靠区分 forehand/backhand。

字段：

```python
"stroke_type": "forehand" | "backhand" | "serve" | "unknown"
"stroke_label_source": "manual" | "rule" | "unknown"
"stroke_label_confidence": float
```

建议第一批 17 个 selected clips 先人工看图或可视化标注 3-5 个，再固化规则。

## 14. Debug 图和验收图

每个 sample 必须自动生成 debug 图：

```text
1. 3D ball + racket trajectory
2. ball-racket distance vs time_rel
3. ball speed vs time_rel
4. racket speed vs time_rel
5. hit score components vs time_rel
6. optional: x/y/z position traces
```

输出目录：

```text
analysis/mocap_cleaning_outputs/DATA260703/debug_plots/<episode_id>/
```

验收标准：

```text
ball-racket distance 在 t=0 附近最小
ball speed 或 ball velocity direction 在 t=0 附近发生变化
racket speed 在 t=0 附近较高
轨迹没有明显单位/轴向错误
```

若 debug plot 不满足上述条件，该 sample 不能进入训练。

## 15. 导出格式

### 15.1 单样本 NPZ

第一版优先导出 `.npz`：

```text
analysis/mocap_cleaning_outputs/DATA260703/samples/<episode_id>.npz
```

建议字段：

```text
episode_id
time
time_rel
valid_mask
ball_pos
ball_vel
racket_pos
racket_quat
racket_vel
racket_omega
hit_index
hit_time
hit_pos
racket_pose_at_hit
racket_vel_at_hit
ball_in_vel
ball_out_vel
landing_pos
success
stroke_type
quality_flags_json
source_json
```

`quality_flags_json` 和 `source_json` 用 JSON string 存储，便于 numpy 加载。

### 15.2 Manifest

批处理必须输出：

```text
analysis/mocap_cleaning_outputs/DATA260703/manifest.json
```

包括：

```python
{
  "dataset_id": "DATA260703",
  "created_at": "...",
  "config": {...},
  "samples": [
    {
      "episode_id": "...",
      "sample_path": "...",
      "debug_plot_dir": "...",
      "source_csv": "...",
      "source_bvh": "...",
      "quality_flags": {...},
      "usable_for_training": true
    }
  ]
}
```

### 15.3 HDF5

当样本数量增加后，再增加 HDF5：

```text
dataset.hdf5
  /episodes/<episode_id>/time
  /episodes/<episode_id>/ball_pos
  /episodes/<episode_id>/racket_pos
  ...
```

HDF5 不是第一版必须项。

## 16. 与重定向/IK 的衔接

清洗样本不会直接替代 BVH 重定向。推荐衔接方式：

```text
CleanSample 确定 hit_index、窗口、质量和标签
↓
按 sample 的窗口裁剪单人 BVH
↓
必要时降采样到 retargeter 要求 fps
↓
BVH -> A3 retargeted CSV
↓
hope_training/whole_body_tracking/scripts/csv_to_npz.py --robot agibot_a3
↓
TrackingFlat
↓
HOPEPingPong
```

A3 retargeted CSV 必须遵守：

```text
root position: 3 columns
root quaternion: 4 columns
A3 joint angles: 31 columns in docs/interfaces/joint_order.md order
```

注意：

```text
Motive BVH 是人体骨架，不是 A3 joint CSV
Motive CSV 是多行表头全局 pose，不是 csv_to_npz.py 输入
```

## 17. 与 HOPEPingPong 的衔接

`HOPEPingPong.yaml` 当前使用 `target_mode: reference_perturbed`，会围绕参考动作击球帧的 racket state 采样目标。因此清洗样本需要为后续任务提供：

```text
hit_index
racket_pos_at_hit
racket_quat_at_hit
racket_vel_at_hit
ball_in_vel
ball_out_vel
optional landing_pos/success
```

这些字段可用于：

```text
1. 选择真实挥拍参考动作
2. 校验 retarget 后 A3 racket state 是否接近人类数据
3. 调整 HOPEPingPong 的 strike_phase
4. 调整 racket target perturbation 范围
5. 筛掉击球不清晰或失败样本
```

`strike_phase` 可以由：

```text
strike_phase = hit_index / episode_length
```

导出到每个样本 metadata 中，后续训练配置可以按动作片段设置。

## 18. 质量 flags 标准

每个 sample 至少包含：

```python
quality_flags = {
    "coordinate_transform_available": False,
    "coordinate_sanity_ok": bool,
    "racket_hand_match_ok": bool,
    "ball_source_valid": bool,
    "has_long_missing_gap": bool,
    "missing_near_hit": bool,
    "racket_speed_reasonable": bool,
    "ball_speed_reasonable": bool,
    "hit_distance_ok": bool,
    "hit_velocity_change_ok": bool,
    "landing_detected": bool,
    "success_label_reliable": bool,
    "usable_for_hit_analysis": bool,
    "usable_for_retargeting": bool,
    "usable_for_training": bool,
}
```

建议判定：

```text
usable_for_hit_analysis =
  ball_source_valid
  and racket_hand_match_ok
  and not missing_near_hit

usable_for_retargeting =
  corresponding BVH exists
  and racket_hand_match_ok
  and human skeleton has no long gap near hit

usable_for_training =
  usable_for_hit_analysis
  and hit_distance_ok
  and hit_velocity_change_ok
  and coordinate_sanity_ok
```

如果没有 table/robot 坐标标定，`usable_for_training` 可以先表示“可用于离线样本训练/分析”，但不能表示“可直接作为绝对场景目标训练”。

## 19. 第一版实施计划

### Milestone 1: Motive CSV 正式 loader

目标：

```text
读取一个 Motive CSV，输出 RawTrial
```

范围：

```text
Csv/Rige Body/Table Tennis_01_005.csv
TennisBats01
TennisBats02
Tennis
Skeleton 001/002 RHand/LHand/Hip
```

验收：

```text
time 单调递增
fps = 360
单位标记为 mm
rigid body 列数正确
quat order = xyzw
能复现 TennisBats01 -> Skeleton001:RHand 的匹配
```

### Milestone 2: 确认 Tennis 是否为球

目标：

```text
判断 Csv/Rige Body 中的 Tennis rigid body 是否能作为 ball_pos
```

输出：

```text
analysis/mocap_cleaning_outputs/DATA260703/ball_source_report.md
```

验收：

```text
Tennis 轨迹范围、速度范围、z 高度、反弹/飞行形态有报告
结论明确: valid ball / invalid ball / uncertain
```

### Milestone 3: clean one episode

目标：

```text
对排名第 1 的片段生成 CleanSample npz + debug plots
```

输入：

```text
Csv/Rige Body/Table Tennis_01_005.csv
selected clip 90.11-92.11s
racket TennisBats01
skeleton Skeleton001
```

输出：

```text
sample npz
sample metadata json
debug plots
```

验收：

```text
hit_index 不为空
distance plot 在 hit 附近有低谷
racket speed 在 hit 附近高
若 Tennis 是球，则 ball speed 在 hit 附近有合理变化
quality_flags 可解释
```

### Milestone 4: batch selected clips

目标：

```text
批量处理 17 个 selected clips
```

验收：

```text
manifest.json 列出全部 sample
每个 sample 有 usable flags
生成汇总表: usable / rejected / rejection reason
```

### Milestone 5: retargeting handoff

目标：

```text
为可用 sample 输出对齐后的 BVH clip 和 retarget config
```

验收：

```text
每个 retarget candidate 有 source_bvh、hit_index、strike_phase、stroke_type
至少一个样本进入 A3 retargeted CSV
至少一个 A3 CSV 成功通过 csv_to_npz.py
```

## 20. 当前已完成与未完成

已完成探索：

```text
数据盘点
CSV/BVH 元信息统计
球拍速度粗筛
BVH 2s 候选片段裁剪
球拍到 skeleton 手部匹配
17 个 selected clips 排序
```

尚未完成正式清洗：

```text
正式 RawTrial loader
Tennis 是否为球的验证
统一单位输出
坐标系转换
缺失值修复
异常点修复
平滑滤波
角速度计算
真实 hit_index 检测
success/landing 判断
sample npz 导出
debug plot 批量生成
HDF5 数据集
A3 retargeted CSV 输出
```

## 21. 代码修改原则

后续实现时遵守：

```text
1. 不修改 /workspace/DATA260703 原始数据
2. 所有输出写入 analysis/mocap_cleaning_outputs/
3. 探索脚本保留，正式逻辑放入 analysis/mocap_cleaning/
4. 每个 CLI 都必须有 --dataset、--output-dir、--config
5. 每个 sample 都必须保存 source metadata
6. 每个 rejection 都必须有 reason
7. 不把 Motive CSV 直接喂给 csv_to_npz.py
8. 不在坐标系未确认时生成可靠 success 标签
9. 不把球拍 mocap 当作部署时策略输入，只用于离线清洗、标注和验证
10. 所有速度阈值必须进 config，不硬编码在核心逻辑里
```

## 22. 建议配置文件

建议新增：

```text
analysis/mocap_cleaning/configs/DATA260703.yaml
```

第一版内容：

```yaml
dataset_id: DATA260703
dataset_root: /workspace/DATA260703
position_unit: mm
quat_order: xyzw
source_frame: motive_global
target_frame: motive_global_m

fps:
  raw: 360
  clean_sample: 200

gap_policy:
  interpolate_max_s: 0.03
  warn_max_s: 0.05
  reject_max_s: 0.15
  hit_guard_s: 0.10

speed_thresholds:
  human_hand_mps: 15.0
  racket_mps: 25.0
  ball_mps: 50.0
  hip_mps: 8.0

filter:
  racket_window_ms: 25
  ball_window_ms: 15
  body_window_ms: 35
  polyorder: 3

hit_detection:
  max_distance_m: 0.15
  distance_ok_m: 0.12
  min_racket_speed_mps: 1.0
  min_ball_dv_mps: 1.0
  weights:
    distance: 0.5
    ball_dv: 0.3
    racket_speed: 0.2

episode:
  pre_hit_s: 0.6
  post_hit_s: 0.4

entities:
  rackets:
    TennisBats01:
      expected_skeleton: "Skeleton 001"
      expected_hand: RHand
    TennisBats02:
      expected_skeleton: "Skeleton 002"
      expected_hand: RHand
  ball_candidates:
    - "Tennis"
```

## 23. 最小可交付闭环

下一步最小实现应交付：

```text
analysis/mocap_cleaning/motive_loader.py
analysis/mocap_cleaning/derivative.py
analysis/mocap_cleaning/hit_detection.py
analysis/mocap_cleaning/export_dataset.py
analysis/mocap_cleaning/plotting.py
analysis/mocap_cleaning/cli_clean_one.py
analysis/mocap_cleaning/configs/DATA260703.yaml
```

命令形态：

```bash
python3 -m analysis.mocap_cleaning.cli_clean_one \
  --config analysis/mocap_cleaning/configs/DATA260703.yaml \
  --csv "/workspace/DATA260703/Csv/Rige Body/Table Tennis_01_005.csv" \
  --racket TennisBats01 \
  --skeleton "Skeleton 001" \
  --start 90.11 \
  --end 92.11 \
  --output-dir analysis/mocap_cleaning_outputs/DATA260703
```

预期输出：

```text
analysis/mocap_cleaning_outputs/DATA260703/samples/<episode_id>.npz
analysis/mocap_cleaning_outputs/DATA260703/metadata/<episode_id>.json
analysis/mocap_cleaning_outputs/DATA260703/debug_plots/<episode_id>/*.png
```

验收成功后，再做 batch。

