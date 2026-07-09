# Packed CleanSample Dataset

Dataset: `DATA260708_train`
Dataset file: `data/analysis/mocap_cleaning_outputs/DATA260708/packed/DATA260708_train.npz`
Format: `npz`

| Metric | Value |
|---|---:|
| Source samples | 1103 |
| Packed samples | 1099 |
| Skipped samples | 4 |
| Frames per sample | 201 |
| FPS | 199.99999999999983 |

## Stroke Distribution

- `backhand`: 12
- `forehand`: 900
- `unknown`: 187

## Source CSV Distribution

- `CSV/T001_001.csv`: 13
- `CSV/T001_002.csv`: 8
- `CSV/T001_003.csv`: 11
- `CSV/T002_001.csv`: 6
- `CSV/T002_006.csv`: 16
- `CSV/T002_008.csv`: 8
- `CSV/T002_010.csv`: 15
- `CSV/T002_012.csv`: 12
- `CSV/T002_013.csv`: 16
- `CSV/T002_015.csv`: 16
- `CSV/T002_019.csv`: 16
- `CSV/T002_021.csv`: 16
- `CSV/T002_022.csv`: 16
- `CSV/T002_023.csv`: 16
- `CSV/T002_024.csv`: 16
- `CSV/T002_027.csv`: 16
- `CSV/T002_028.csv`: 16
- `CSV/T03_002.csv`: 16
- `CSV/T03_003.csv`: 16
- `CSV/T03_004.csv`: 9
- `CSV/T03_005.csv`: 16
- `CSV/T03_007.csv`: 9
- `CSV/T03_008.csv`: 15
- `CSV/T03_011.csv`: 11
- `CSV/T03_012.csv`: 16
- `CSV/T03_013.csv`: 16
- `CSV/T03_014.csv`: 16
- `CSV/T03_015.csv`: 16
- `CSV/T03_016.csv`: 12
- `CSV/T03_019.csv`: 16
- `CSV/T03_022.csv`: 15
- `CSV/T03_025.csv`: 16
- `CSV/T03_027.csv`: 11
- `CSV/T03_028.csv`: 7
- `CSV/T03_030.csv`: 14
- `CSV/T03_031.csv`: 13
- `CSV/T03_032.csv`: 16
- `CSV/T03_034.csv`: 11
- `CSV/T03_037.csv`: 14
- `CSV/T03_038.csv`: 9
- `CSV/T03_039.csv`: 7
- `CSV/T03_044.csv`: 16
- `CSV/T03_045.csv`: 9
- `CSV/T03_050.csv`: 16
- `CSV/T03_052.csv`: 13
- `CSV/T03_053.csv`: 16
- `CSV/T03_059.csv`: 8
- `CSV/T03_065.csv`: 16
- `CSV/T03_067.csv`: 9
- `CSV/T03_068.csv`: 15
- `CSV/T03_069.csv`: 9
- `CSV/T03_071.csv`: 7
- `CSV/T03_072.csv`: 9
- `CSV/T03_075.csv`: 16
- `CSV/T03_076.csv`: 15
- `CSV/T03_077.csv`: 10
- `CSV/T03_078.csv`: 12
- `CSV/T03_079.csv`: 15
- `CSV/T03_080.csv`: 8
- `CSV/T03_083.csv`: 16
- `CSV/T03_084.csv`: 5
- `CSV/T04_001.csv`: 12
- `CSV/T04_005.csv`: 8
- `CSV/T04_006.csv`: 4
- `CSV/T04_007.csv`: 6
- `CSV/T04_009.csv`: 15
- `CSV/T04_010.csv`: 7
- `CSV/T04_014.csv`: 11
- `CSV/T04_018.csv`: 16
- `CSV/T04_021.csv`: 15
- `CSV/T04_023.csv`: 15
- `CSV/T04_024.csv`: 6
- `CSV/T04_026.csv`: 9
- `CSV/T_001.csv`: 15
- `CSV/T_002.csv`: 15
- `CSV/T_003.csv`: 16
- `CSV/T_006.csv`: 16
- `CSV/T_009.csv`: 16
- `CSV/T_010.csv`: 15
- `CSV/T_011.csv`: 16
- `CSV/T_013.csv`: 15
- `CSV/T_014.csv`: 15
- `CSV/T_018.csv`: 8
- `CSV/T_019.csv`: 12
- `CSV/T_020.csv`: 16
- `CSV/T_021.csv`: 10

## Fields

- `time`: `(1099, 201)`
- `time_rel`: `(1099, 201)`
- `valid_mask`: `(1099, 201)`
- `ball_pos`: `(1099, 201, 3)`
- `ball_vel`: `(1099, 201, 3)`
- `racket_pos`: `(1099, 201, 3)`
- `racket_quat`: `(1099, 201, 4)`
- `racket_vel`: `(1099, 201, 3)`
- `racket_omega`: `(1099, 201, 3)`
- `body_center`: `(1099, 201, 3)`
- `body_right_axis`: `(1099, 201, 3)`
- `hit_pos`: `(1099, 3)`
- `racket_pose_at_hit`: `(1099, 7)`
- `racket_vel_at_hit`: `(1099, 3)`
- `ball_in_vel`: `(1099, 3)`
- `ball_out_vel`: `(1099, 3)`
- `landing_pos`: `(1099, 3)`
- `dist`: `(1099, 201)`
- `ball_dv`: `(1099, 201)`
- `score`: `(1099, 201)`
- `hit_index`: `(1099,)`
- `hit_time`: `(1099,)`
- `success`: `(1099,)`
- `episode_id`: `(1099,)`
- `stroke_type`: `(1099,)`
- `quality_flags_json`: `(1099,)`
- `source_json`: `(1099,)`
