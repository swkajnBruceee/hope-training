# DATA260703 CleanSample Manifest

Created at: `2026-07-04T06:19:53.634414+00:00`

| Episode | Frames | Hit Index | Hit t_rel (s) | Usable | Sample |
|---|---:|---:|---:|---|---|
| `Table_Tennis_01_005_TennisBats01_90p11_92p11_Skeleton001` | 201 | 120 | 0.914 | True | `data/analysis/mocap_cleaning_outputs/DATA260703/samples/Table_Tennis_01_005_TennisBats01_90p11_92p11_Skeleton001.npz` |
| `Table_Tennis_01_006_TennisBats01_12p03_14p03_Skeleton001` | 201 | 120 | 0.978 | True | `data/analysis/mocap_cleaning_outputs/DATA260703/samples/Table_Tennis_01_006_TennisBats01_12p03_14p03_Skeleton001.npz` |
| `Table_Tennis_01_006_TennisBats02_79p22_81p22_Skeleton002` | 201 | 120 | 0.961 | True | `data/analysis/mocap_cleaning_outputs/DATA260703/samples/Table_Tennis_01_006_TennisBats02_79p22_81p22_Skeleton002.npz` |
| `Table_Tennis_01_007_TennisBats01_63p44_65p44_Skeleton001` | 201 | 120 | 0.953 | True | `data/analysis/mocap_cleaning_outputs/DATA260703/samples/Table_Tennis_01_007_TennisBats01_63p44_65p44_Skeleton001.npz` |
| `Table_Tennis_01_008_TennisBats01_63p22_65p22_Skeleton001` | 201 | 120 | 0.917 | True | `data/analysis/mocap_cleaning_outputs/DATA260703/samples/Table_Tennis_01_008_TennisBats01_63p22_65p22_Skeleton001.npz` |
| `Table_Tennis_01_009_TennisBats01_138p89_140p89_Skeleton001` | 201 | 120 | 0.914 | True | `data/analysis/mocap_cleaning_outputs/DATA260703/samples/Table_Tennis_01_009_TennisBats01_138p89_140p89_Skeleton001.npz` |
| `Table_Tennis_01_009_TennisBats02_106p64_108p64_Skeleton002` | 201 | 120 | 1.292 | True | `data/analysis/mocap_cleaning_outputs/DATA260703/samples/Table_Tennis_01_009_TennisBats02_106p64_108p64_Skeleton002.npz` |

## Notes

- `racket_quat` comes from Motive rigid body rotation in xyzw order.
- `racket_omega` is computed from frame-to-frame quaternion deltas in rad/s.
- CleanSample time axes are hit-centered and resampled to the configured target FPS.
- `success=-1` means unknown because table/world landing labels are not available yet.

## Label Status

- Reliable success labels: 0
- Unreliable/unknown success labels: 7
- `missing_table_calibration`: 7

## Stroke Status

- `forehand`: 6
- `unknown`: 1
- `body_lateral_offset_rule`: 6
- `low_lateral_separation`: 1
