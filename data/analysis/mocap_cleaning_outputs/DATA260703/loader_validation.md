# DATA260703 Loader Validation

- Config: `data/analysis/mocap_cleaning/configs/DATA260703.yaml`
- CSV: `data/DATA260703/Csv/Rige Body/Table Tennis_01_005.csv`
- Take: `Table Tennis_01_005`
- FPS: 360.0
- Frames: 33431
- Time monotonic: True
- Position unit: `millimeters`
- Coordinate space: `global`
- Quaternion order: `xyzw`

## Entity Counts

| Type | Count |
|---|---:|
| Bone | 102 |
| Rigid Body | 3 |
| Marker | 887 |

## Loaded Rigid Bodies

| Name | Position Shape | Quaternion Shape |
|---|---|---|
| TennisBats01 | `[33431, 3]` | `[33431, 4]` |
| TennisBats02 | `[33431, 3]` | `[33431, 4]` |
| Tennis | `[33431, 3]` | `[33431, 4]` |

## Loaded Bones

| Name | Position Shape | Quaternion Shape |
|---|---|---|
| Skeleton 001:RHand | `[33431, 3]` | `[33431, 4]` |
| Skeleton 001:Hip | `[33431, 3]` | `[33431, 4]` |
| Skeleton 001:LHand | `[33431, 3]` | `[33431, 4]` |

## Racket-Hand Match

- Racket: `TennisBats01`
- Hand: `Skeleton 001:RHand`
- Median distance: 0.209 m
- Min distance: 0.195 m
- Max distance: 0.229 m
- OK: True
