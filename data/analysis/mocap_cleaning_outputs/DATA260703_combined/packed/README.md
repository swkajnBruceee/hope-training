# DATA260703 Combined Packed Dataset

This dataset merges only quality-approved samples from:

- Rige Body max set: 351 samples
- Point stitched audit-ok set: 441 samples

It excludes:

- Point single-marker experimental set
- Point stitched samples rejected by audit
- Rige Body samples flagged `usable_for_training=False`

## Files

- `DATA260703_combined_train.npz`: compressed training dataset.
- `pack_report.md`: packing summary and field shapes.
- `validation_report.md`: validation summary.

## Read Example

```python
import numpy as np

data = np.load(
    "data/analysis/mocap_cleaning_outputs/DATA260703_combined/packed/DATA260703_combined_train.npz",
    allow_pickle=False,
)

ball_pos = data["ball_pos"]          # [792, 201, 3]
racket_pos = data["racket_pos"]      # [792, 201, 3]
racket_quat = data["racket_quat"]    # [792, 201, 4], xyzw
hit_index = data["hit_index"]        # [792], always 120
stroke_type = data["stroke_type"]    # [792]
```

## Dataset Contract

- Samples: 792
- FPS: 200
- Frames per sample: 201
- Window: hit-centered, `[-0.6s, +0.4s]`
- `hit_index`: 120
- Position unit: meter
- Coordinate frame: `motive_global_m`
- Quaternion order: `xyzw`
- `success`: `-1` for all samples because table landing calibration is not available.

## Source Distribution

- `Csv/Point/Table Tennis_01_004.csv`: 70
- `Csv/Point/Table Tennis_01_012.csv`: 49
- `Csv/Point/Table Tennis_01_013.csv`: 152
- `Csv/Point/Table Tennis_01_014.csv`: 170
- `Csv/Rige Body/Table Tennis_01_005.csv`: 35
- `Csv/Rige Body/Table Tennis_01_006.csv`: 82
- `Csv/Rige Body/Table Tennis_01_007.csv`: 62
- `Csv/Rige Body/Table Tennis_01_008.csv`: 58
- `Csv/Rige Body/Table Tennis_01_009.csv`: 114

## Current Limitations

- `success` is unknown for all samples.
- Coordinates are still in Motive global meters, not robot base or table frame.
- `stroke_type` is rule-based and still needs improved backhand detection.

