# DATA260703 正反手划分分析

本文只分析当前已合并数据集的正反手标签状态，不修改样本文件。

## 输入数据

- 数据包: `analysis/mocap_cleaning_outputs/DATA260703_combined/packed/DATA260703_combined_train.npz`
- 样本数: 792
- 每条样本长度: 201 帧
- 采样率: 200 Hz
- 当前坐标系: `motive_global_m`
- 当前 success/landing: 未启用，原因是 table frame 尚未完全标定

## 当前标签状态

当前 `stroke_type` 的分布为:

| label | count |
|---|---:|
| forehand | 726 |
| unknown | 66 |
| backhand | 0 |

这说明当前导出的正反手标签不能直接作为训练标签使用。问题不是数据中没有反手，而是之前的规则过粗，主要只看“球拍在身体右侧还是左侧”，导致大量右手持拍动作都被标成 `forehand`。

## 可用于正反手分析的字段

当前 packed dataset 已经包含以下字段，可用于不依赖球台坐标的正反手初筛:

- `racket_pos`: 球拍位置
- `racket_vel`: 球拍速度
- `body_center`: 人体中心
- `body_right_axis`: 人体右方向轴
- `hit_index`: 击球帧
- `episode_id`: 样本 ID
- `source_json`: 来源 CSV、球拍、骨架等元信息

关键思想是先把球拍运动投影到人体自身横向轴，而不是 Motive/球台全局轴:

```python
rel = racket_pos_at_hit - body_center_at_hit
lateral_offset = dot(rel, body_right_axis)
lateral_velocity = dot(racket_vel_at_hit, body_right_axis)
pre_to_hit_lateral_delta = lateral_offset_at_hit - lateral_offset_pre_hit
```

## 关键统计

### 来源分布

| source | count |
|---|---:|
| Csv/Point/Table Tennis_01_014.csv | 170 |
| Csv/Point/Table Tennis_01_013.csv | 152 |
| Csv/Rige Body/Table Tennis_01_009.csv | 114 |
| Csv/Rige Body/Table Tennis_01_006.csv | 82 |
| Csv/Point/Table Tennis_01_004.csv | 70 |
| Csv/Rige Body/Table Tennis_01_007.csv | 62 |
| Csv/Rige Body/Table Tennis_01_008.csv | 58 |
| Csv/Point/Table Tennis_01_012.csv | 49 |
| Csv/Rige Body/Table Tennis_01_005.csv | 35 |

### 球拍/骨架分布

| racket / skeleton | count |
|---|---:|
| TennisBats01 / Skeleton001 | 401 |
| TennisBats02 / Skeleton002 | 391 |

### 横向击球位置

`lateral_offset = dot(racket_pos_at_hit - body_center_at_hit, body_right_axis)`

整体分布:

| percentile | value m |
|---:|---:|
| 0 | -0.025 |
| 5 | 0.055 |
| 25 | 0.155 |
| 50 | 0.284 |
| 75 | 0.406 |
| 95 | 0.557 |
| 100 | 0.758 |

按球拍拆分:

| racket | median lateral_offset m | interpretation |
|---|---:|---|
| TennisBats01 | 0.394 | 击球点明显在持拍手外侧，更像正手 |
| TennisBats02 | 0.163 | 击球点更靠身体中线，更像反手 |

只看横向位置不足以严格区分正反手，因为右手反手也可能仍然出现在身体右前方或中线附近。

### 横向击球速度

`lateral_velocity = dot(racket_vel_at_hit, body_right_axis)`

按球拍拆分:

| racket | median lateral_velocity m/s | pattern |
|---|---:|---|
| TennisBats01 | -1.450 | 击球瞬间向身体内侧/左向运动，符合右手正手 |
| TennisBats02 | 0.438 | 击球瞬间向身体外侧/右向运动，符合右手反手 |

这是目前最强的正反手区分信号。

### 击球前到击球时的横向位移

`pre_to_hit_lateral_delta = lateral_at_hit - lateral_80ms_before_hit`

按球拍拆分:

| racket | median delta m | pattern |
|---|---:|---|
| TennisBats01 | -0.085 | 从右侧向内挥，符合正手 |
| TennisBats02 | 0.028 | 从中线/左侧向外挥，符合反手 |

该信号和横向速度基本一致。已知标签条件下，速度符号规则和位移符号规则在 665 条样本上同向，仅 1 条明显冲突。

## 候选规则分析

### 规则 A: 只看击球横向速度

```python
if lateral_velocity < -0.2:
    stroke_type = "forehand"
elif lateral_velocity > 0.2:
    stroke_type = "backhand"
else:
    stroke_type = "unknown"
```

结果:

| label | count |
|---|---:|
| forehand | 421 |
| backhand | 322 |
| unknown | 49 |

按球拍:

| racket | forehand | backhand | unknown |
|---|---:|---:|---:|
| TennisBats01 | 384 | 5 | 12 |
| TennisBats02 | 37 | 317 | 37 |

### 规则 B: 只看击球前到击球时横向位移

```python
if pre_to_hit_lateral_delta < -0.02:
    stroke_type = "forehand"
elif pre_to_hit_lateral_delta > 0.02:
    stroke_type = "backhand"
else:
    stroke_type = "unknown"
```

结果:

| label | count |
|---|---:|
| forehand | 414 |
| backhand | 270 |
| unknown | 108 |

按球拍:

| racket | forehand | backhand | unknown |
|---|---:|---:|---:|
| TennisBats01 | 374 | 10 | 17 |
| TennisBats02 | 40 | 260 | 91 |

该规则更保守，反手数量少一些。

### 规则 C: 组合规则

建议第一版使用组合规则:

```python
score_forehand = 0
score_backhand = 0

if lateral_velocity < -0.5:
    score_forehand += 2
elif lateral_velocity < -0.2:
    score_forehand += 1
elif lateral_velocity > 0.5:
    score_backhand += 2
elif lateral_velocity > 0.2:
    score_backhand += 1

if pre_to_hit_lateral_delta < -0.04:
    score_forehand += 2
elif pre_to_hit_lateral_delta < -0.015:
    score_forehand += 1
elif pre_to_hit_lateral_delta > 0.04:
    score_backhand += 2
elif pre_to_hit_lateral_delta > 0.015:
    score_backhand += 1

if lateral_offset > 0.28:
    score_forehand += 1
elif lateral_offset < 0.18:
    score_backhand += 1

if score_forehand >= score_backhand + 2:
    stroke_type = "forehand"
elif score_backhand >= score_forehand + 2:
    stroke_type = "backhand"
else:
    stroke_type = "unknown"
```

组合规则结果:

| label | count |
|---|---:|
| forehand | 420 |
| backhand | 325 |
| unknown | 47 |

按球拍:

| racket | forehand | backhand | unknown |
|---|---:|---:|---:|
| TennisBats01 | 384 | 8 | 9 |
| TennisBats02 | 36 | 317 | 38 |

按来源:

| source | forehand | backhand | unknown |
|---|---:|---:|---:|
| Csv/Point/Table Tennis_01_004.csv | 62 | 3 | 5 |
| Csv/Point/Table Tennis_01_012.csv | 21 | 24 | 4 |
| Csv/Point/Table Tennis_01_013.csv | 74 | 69 | 9 |
| Csv/Point/Table Tennis_01_014.csv | 80 | 86 | 4 |
| Csv/Rige Body/Table Tennis_01_005.csv | 23 | 10 | 2 |
| Csv/Rige Body/Table Tennis_01_006.csv | 41 | 34 | 7 |
| Csv/Rige Body/Table Tennis_01_007.csv | 31 | 24 | 7 |
| Csv/Rige Body/Table Tennis_01_008.csv | 28 | 26 | 4 |
| Csv/Rige Body/Table Tennis_01_009.csv | 60 | 49 | 5 |

## 结论

1. 当前数据里不是没有反手，现有标签规则漏掉了反手。
2. `TennisBats01/Skeleton001` 大多数是正手模式。
3. `TennisBats02/Skeleton002` 大多数是反手模式。
4. 仅靠“球拍在身体左侧/右侧”不够，必须引入击球瞬间横向速度和击球前到击球时的横向位移。
5. 正反手划分不一定要等完整 table frame 标定，因为人体局部坐标已经足够做第一版动作类型标签。
6. table frame 仍然重要，但主要影响落点、success、球台区域判断、机器人目标坐标，不是正反手初筛的硬前置。

## 建议修改计划

### 第一步: 先新增 stroke relabel 工具

新增一个只读输入、另存输出的脚本，不覆盖原始 packed dataset:

```text
analysis/mocap_cleaning_tools/relabel_stroke_type.py
```

输入:

```text
DATA260703_combined_train.npz
```

输出:

```text
DATA260703_combined_train_stroke_relabel.npz
DATA260703_stroke_relabel_report.md
```

当前已经生成的重标注文件位于:

```text
analysis/mocap_cleaning_outputs/DATA260703_combined/stroke_relabel/DATA260703_combined_train_stroke_relabel.npz
```

新增字段:

```python
stroke_type_rule_v2
stroke_confidence_rule_v2
stroke_features_rule_v2
```

其中 `stroke_features_rule_v2` 至少包含:

```python
lateral_offset_m
lateral_velocity_mps
pre_to_hit_lateral_delta_m
rule_score_forehand
rule_score_backhand
```

### 第二步: 人工抽检

优先抽检以下样本:

- `TennisBats01` 被规则标为 `backhand` 的 8 条
- `TennisBats02` 被规则标为 `forehand` 的 36 条
- 47 条 `unknown`

这 91 条是最可能含有误标或边界动作的样本。抽检通过后，再把 `stroke_type_rule_v2` 提升为正式 `stroke_type`。

### 第三步: 后续正式清洗

等 table frame 标定完成后，重新输出完整数据集时:

- 保留人体局部正反手规则作为主规则
- 用球台方向和站位做 sanity check
- 正式写入 `stroke_label_source = "body_local_swing_rule_v2"`
- 对低置信度样本保留 `unknown`，不要硬分
