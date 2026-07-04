# Packed CleanSample Dataset

Dataset: `DATA260703_rigidbody_max_train`
Dataset file: `data/analysis/mocap_cleaning_outputs/DATA260703_max/packed/DATA260703_rigidbody_max_train.npz`
Format: `npz`

| Metric | Value |
|---|---:|
| Source samples | 355 |
| Packed samples | 351 |
| Skipped samples | 4 |
| Frames per sample | 201 |
| FPS | 199.99999999996874 |

## Stroke Distribution

- `forehand`: 331
- `unknown`: 20

## Source CSV Distribution

- `Csv/Rige Body/Table Tennis_01_005.csv`: 35
- `Csv/Rige Body/Table Tennis_01_006.csv`: 82
- `Csv/Rige Body/Table Tennis_01_007.csv`: 62
- `Csv/Rige Body/Table Tennis_01_008.csv`: 58
- `Csv/Rige Body/Table Tennis_01_009.csv`: 114

## Fields

- `time`: `(351, 201)`
- `time_rel`: `(351, 201)`
- `valid_mask`: `(351, 201)`
- `ball_pos`: `(351, 201, 3)`
- `ball_vel`: `(351, 201, 3)`
- `racket_pos`: `(351, 201, 3)`
- `racket_quat`: `(351, 201, 4)`
- `racket_vel`: `(351, 201, 3)`
- `racket_omega`: `(351, 201, 3)`
- `body_center`: `(351, 201, 3)`
- `body_right_axis`: `(351, 201, 3)`
- `hit_pos`: `(351, 3)`
- `racket_pose_at_hit`: `(351, 7)`
- `racket_vel_at_hit`: `(351, 3)`
- `ball_in_vel`: `(351, 3)`
- `ball_out_vel`: `(351, 3)`
- `landing_pos`: `(351, 3)`
- `dist`: `(351, 201)`
- `ball_dv`: `(351, 201)`
- `score`: `(351, 201)`
- `hit_index`: `(351,)`
- `hit_time`: `(351,)`
- `success`: `(351,)`
- `episode_id`: `(351,)`
- `stroke_type`: `(351,)`
- `quality_flags_json`: `(351,)`
- `source_json`: `(351,)`
