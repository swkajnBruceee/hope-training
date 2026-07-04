# Racket To Skeleton Matching

Distances are computed from each racket rigid body to skeleton hand bones inside the candidate window.
Lower median distance is the better match.

| CSV | Racket | Best Hand | Median Dist (m) | Min Dist (m) | Window (s) |
|---|---|---|---:|---:|---|
| `Csv/Point/Table Tennis_01_004.csv` | TennisBats01 | Skeleton 001:RHand | 0.206 | 0.200 | 101.97-102.97 |
| `Csv/Point/Table Tennis_01_004.csv` | TennisBats02 | Skeleton 002:RHand | 0.201 | 0.186 | 52.36-53.36 |
| `Csv/Point/Table Tennis_01_012.csv` | TennisBats01 | Skeleton 001:RHand | 0.207 | 0.198 | 116.19-117.19 |
| `Csv/Point/Table Tennis_01_012.csv` | TennisBats02 | Skeleton 002:RHand | 0.219 | 0.195 | 139.22-140.22 |
| `Csv/Point/Table Tennis_01_013.csv` | TennisBats01 | Skeleton 001:RHand | 0.209 | 0.206 | 37.08-38.08 |
| `Csv/Point/Table Tennis_01_013.csv` | TennisBats02 | Skeleton 002:RHand | 0.185 | 0.180 | 118.92-119.92 |
| `Csv/Point/Table Tennis_01_014.csv` | TennisBats01 | Skeleton 001:RHand | 0.205 | 0.198 | 23.97-24.97 |
| `Csv/Point/Table Tennis_01_014.csv` | TennisBats02 | Skeleton 002:RHand | 0.223 | 0.202 | 160.06-161.06 |
| `Csv/Rige Body/Table Tennis_01_005.csv` | TennisBats01 | Skeleton 001:RHand | 0.221 | 0.206 | 90.61-91.61 |
| `Csv/Rige Body/Table Tennis_01_005.csv` | TennisBats02 | Skeleton 002:RHand | 0.198 | 0.193 | 71.75-72.75 |
| `Csv/Rige Body/Table Tennis_01_006.csv` | TennisBats01 | Skeleton 001:RHand | 0.209 | 0.205 | 12.53-13.53 |
| `Csv/Rige Body/Table Tennis_01_006.csv` | TennisBats02 | Skeleton 002:RHand | 0.203 | 0.195 | 79.72-80.72 |
| `Csv/Rige Body/Table Tennis_01_007.csv` | TennisBats01 | Skeleton 001:RHand | 0.208 | 0.201 | 63.94-64.94 |
| `Csv/Rige Body/Table Tennis_01_008.csv` | TennisBats01 | Skeleton 001:RHand | 0.216 | 0.210 | 63.72-64.72 |
| `Csv/Rige Body/Table Tennis_01_008.csv` | TennisBats02 | Skeleton 002:RHand | 0.211 | 0.194 | 71.25-72.25 |
| `Csv/Rige Body/Table Tennis_01_009.csv` | TennisBats01 | Skeleton 001:RHand | 0.211 | 0.205 | 139.39-140.39 |
| `Csv/Rige Body/Table Tennis_01_009.csv` | TennisBats02 | Skeleton 002:RHand | 0.216 | 0.195 | 107.14-108.14 |

## Full Ranking

### `Csv/Point/Table Tennis_01_004.csv` `101.97-102.97s`
- `TennisBats01`: Skeleton 001:RHand=0.206m, Skeleton 001:LHand=0.595m, Skeleton 002:LHand=3.448m, Skeleton 002:RHand=3.735m

### `Csv/Point/Table Tennis_01_004.csv` `52.36-53.36s`
- `TennisBats02`: Skeleton 002:RHand=0.201m, Skeleton 002:LHand=0.588m, Skeleton 001:RHand=3.667m, Skeleton 001:LHand=3.725m

### `Csv/Point/Table Tennis_01_012.csv` `116.19-117.19s`
- `TennisBats01`: Skeleton 001:RHand=0.207m, Skeleton 001:LHand=0.682m, Skeleton 002:LHand=3.349m, Skeleton 002:RHand=3.351m

### `Csv/Point/Table Tennis_01_012.csv` `139.22-140.22s`
- `TennisBats02`: Skeleton 002:RHand=0.219m, Skeleton 002:LHand=0.517m, Skeleton 001:RHand=3.096m, Skeleton 001:LHand=3.252m

### `Csv/Point/Table Tennis_01_013.csv` `37.08-38.08s`
- `TennisBats01`: Skeleton 001:RHand=0.209m, Skeleton 001:LHand=0.679m, Skeleton 002:RHand=3.199m, Skeleton 002:LHand=3.341m

### `Csv/Point/Table Tennis_01_013.csv` `118.92-119.92s`
- `TennisBats02`: Skeleton 002:RHand=0.185m, Skeleton 002:LHand=0.346m, Skeleton 001:LHand=2.942m, Skeleton 001:RHand=2.949m

### `Csv/Point/Table Tennis_01_014.csv` `23.97-24.97s`
- `TennisBats01`: Skeleton 001:RHand=0.205m, Skeleton 001:LHand=0.652m, Skeleton 002:RHand=3.021m, Skeleton 002:LHand=3.033m

### `Csv/Point/Table Tennis_01_014.csv` `160.06-161.06s`
- `TennisBats02`: Skeleton 002:RHand=0.223m, Skeleton 002:LHand=0.577m, Skeleton 001:RHand=3.082m, Skeleton 001:LHand=3.205m

### `Csv/Rige Body/Table Tennis_01_005.csv` `90.61-91.61s`
- `TennisBats01`: Skeleton 001:RHand=0.221m, Skeleton 001:LHand=0.856m, Skeleton 002:RHand=3.621m, Skeleton 002:LHand=3.694m

### `Csv/Rige Body/Table Tennis_01_005.csv` `71.75-72.75s`
- `TennisBats02`: Skeleton 002:RHand=0.198m, Skeleton 002:LHand=0.355m, Skeleton 001:LHand=2.765m, Skeleton 001:RHand=2.871m

### `Csv/Rige Body/Table Tennis_01_006.csv` `12.53-13.53s`
- `TennisBats01`: Skeleton 001:RHand=0.209m, Skeleton 001:LHand=0.643m, Skeleton 002:RHand=3.095m, Skeleton 002:LHand=3.365m

### `Csv/Rige Body/Table Tennis_01_006.csv` `79.72-80.72s`
- `TennisBats02`: Skeleton 002:RHand=0.203m, Skeleton 002:LHand=0.355m, Skeleton 001:LHand=2.896m, Skeleton 001:RHand=2.997m

### `Csv/Rige Body/Table Tennis_01_007.csv` `63.94-64.94s`
- `TennisBats01`: Skeleton 001:RHand=0.208m, Skeleton 001:LHand=0.653m, Skeleton 002:RHand=3.152m, Skeleton 002:LHand=3.213m

### `Csv/Rige Body/Table Tennis_01_008.csv` `63.72-64.72s`
- `TennisBats01`: Skeleton 001:RHand=0.216m, Skeleton 001:LHand=0.643m, Skeleton 002:RHand=3.115m, Skeleton 002:LHand=3.195m

### `Csv/Rige Body/Table Tennis_01_008.csv` `71.25-72.25s`
- `TennisBats02`: Skeleton 002:RHand=0.211m, Skeleton 002:LHand=0.425m, Skeleton 001:LHand=2.826m, Skeleton 001:RHand=2.888m

### `Csv/Rige Body/Table Tennis_01_009.csv` `139.39-140.39s`
- `TennisBats01`: Skeleton 001:RHand=0.211m, Skeleton 001:LHand=0.828m, Skeleton 002:LHand=3.637m, Skeleton 002:RHand=3.732m

### `Csv/Rige Body/Table Tennis_01_009.csv` `107.14-108.14s`
- `TennisBats02`: Skeleton 002:RHand=0.216m, Skeleton 002:LHand=0.581m, Skeleton 001:RHand=3.085m, Skeleton 001:LHand=3.163m
