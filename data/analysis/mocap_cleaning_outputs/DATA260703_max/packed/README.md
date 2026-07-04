# DATA260703 Rige Body Packed Dataset

This packed dataset contains only the Rige Body max set.

It does not include `Csv/Point/Table Tennis_01_004.csv` or any Point-derived samples.

## Files

- `DATA260703_rigidbody_max_train.npz`: compressed training dataset.
- `pack_report.md`: packing summary and field shapes.
- `validation_report.md`: validation summary.

## Read Example

```python
import numpy as np

data = np.load(
    "data/analysis/mocap_cleaning_outputs/DATA260703_max/packed/DATA260703_rigidbody_max_train.npz",
    allow_pickle=False,
)

ball_pos = data["ball_pos"]          # [351, 201, 3]
racket_pos = data["racket_pos"]      # [351, 201, 3]
racket_quat = data["racket_quat"]    # [351, 201, 4], xyzw
hit_index = data["hit_index"]        # [351], always 120
stroke_type = data["stroke_type"]    # [351]
```

## Dataset Contract

- Samples: 351
- FPS: 200
- Frames per sample: 201
- Window: hit-centered, `[-0.6s, +0.4s]`
- `hit_index`: 120
- Position unit: meter
- Coordinate frame: `motive_global_m`
- Quaternion order: `xyzw`
- `success`: `-1` for all samples because table landing calibration is not available.

## Main Fields

- `ball_pos`: `[N, T, 3]`
- `ball_vel`: `[N, T, 3]`
- `racket_pos`: `[N, T, 3]`
- `racket_quat`: `[N, T, 4]`
- `racket_vel`: `[N, T, 3]`
- `racket_omega`: `[N, T, 3]`
- `body_center`: `[N, T, 3]`
- `body_right_axis`: `[N, T, 3]`
- `hit_pos`: `[N, 3]`
- `racket_pose_at_hit`: `[N, 7]`
- `ball_in_vel`: `[N, 3]`
- `ball_out_vel`: `[N, 3]`
- `stroke_type`: `[N]`
- `quality_flags_json`: `[N]`
- `source_json`: `[N]`

