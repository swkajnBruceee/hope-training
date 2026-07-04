# DATA260703 Ball Source Validation

Dataset: `data/DATA260703`

Decision legend: `valid` means usable as `ball_pos`; `invalid` means do not use as ball; `uncertain` means inspect plots/raw data before use.

| CSV | Candidate | Decision | Valid Ratio | Height Range (m) | Median Speed | Robust P95 Speed | Max Speed | Jump Ratio | Near Racket Events |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `Csv/Rige Body/Table Tennis_01_005.csv` | Tennis | **uncertain** | 1.000 | 6.994 | 2.141 | 6.262 | 523.812 | 0.066% | TennisBats01:929, TennisBats02:1032 |
| `Csv/Rige Body/Table Tennis_01_006.csv` | Tennis | **uncertain** | 1.000 | 5.301 | 3.420 | 6.496 | 376.507 | 0.033% | TennisBats01:1406, TennisBats02:1588 |
| `Csv/Rige Body/Table Tennis_01_007.csv` | Tennis | **uncertain** | 0.986 | 7.105 | 3.111 | 6.706 | 539.289 | 0.036% | TennisBats01:1046, TennisBats02:1398 |
| `Csv/Rige Body/Table Tennis_01_008.csv` | Tennis | **uncertain** | 1.000 | 5.349 | 2.087 | 6.468 | 435.911 | 0.078% | TennisBats01:1429, TennisBats02:1743 |
| `Csv/Rige Body/Table Tennis_01_009.csv` | Tennis | **uncertain** | 0.999 | 8.614 | 3.850 | 6.796 | 708.363 | 0.035% | TennisBats01:2118, TennisBats02:2543 |

## Selected Clip Window Metrics

| Clip | Candidate | Racket | Window (s) | Min Dist (m) | Near<0.20m | Median Speed | P95 Speed | Max Speed | Z Range (m) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `Table_Tennis_01_005_TennisBats01_90p11_92p11_Skeleton001.bvh` | Tennis | TennisBats01 | 90.11-92.11 | 0.116 | 23/720 | 3.208 | 5.728 | 17.916 | 4.505 |
| `Table_Tennis_01_005_TennisBats02_71p25_73p25_Skeleton002.bvh` | Tennis | TennisBats02 | 71.25-73.25 | 0.041 | 54/721 | 1.825 | 5.386 | 142.220 | 3.050 |
| `Table_Tennis_01_006_TennisBats01_12p03_14p03_Skeleton001.bvh` | Tennis | TennisBats01 | 12.03-14.03 | 0.044 | 27/720 | 4.960 | 7.403 | 29.515 | 2.978 |
| `Table_Tennis_01_006_TennisBats02_79p22_81p22_Skeleton002.bvh` | Tennis | TennisBats02 | 79.22-81.22 | 0.051 | 53/720 | 2.100 | 5.550 | 72.925 | 2.543 |
| `Table_Tennis_01_007_TennisBats01_63p44_65p44_Skeleton001.bvh` | Tennis | TennisBats01 | 63.44-65.44 | 0.033 | 25/720 | 5.315 | 7.973 | 21.631 | 3.062 |
| `Table_Tennis_01_008_TennisBats01_63p22_65p22_Skeleton001.bvh` | Tennis | TennisBats01 | 63.22-65.22 | 0.071 | 27/720 | 5.243 | 7.152 | 25.874 | 2.952 |
| `Table_Tennis_01_008_TennisBats02_70p75_72p75_Skeleton002.bvh` | Tennis | TennisBats02 | 70.75-72.75 | 0.071 | 46/721 | 2.351 | 6.279 | 114.030 | 3.120 |
| `Table_Tennis_01_009_TennisBats01_138p89_140p89_Skeleton001.bvh` | Tennis | TennisBats01 | 138.89-140.89 | 0.040 | 28/720 | 4.010 | 6.406 | 18.625 | 3.445 |
| `Table_Tennis_01_009_TennisBats02_106p64_108p64_Skeleton002.bvh` | Tennis | TennisBats02 | 106.64-108.64 | 0.051 | 30/720 | 5.148 | 7.865 | 24.496 | 2.995 |

## Reasons

### `Csv/Rige Body/Table Tennis_01_005.csv`
- `Tennis`: uncertain - has isolated implausible speed (523.812 m/s); dynamic enough to inspect, but not enough evidence for valid ball

### `Csv/Rige Body/Table Tennis_01_006.csv`
- `Tennis`: uncertain - has isolated implausible speed (376.507 m/s); dynamic enough to inspect, but not enough evidence for valid ball

### `Csv/Rige Body/Table Tennis_01_007.csv`
- `Tennis`: uncertain - has isolated implausible speed (539.289 m/s); dynamic enough to inspect, but not enough evidence for valid ball

### `Csv/Rige Body/Table Tennis_01_008.csv`
- `Tennis`: uncertain - has isolated implausible speed (435.911 m/s); dynamic enough to inspect, but not enough evidence for valid ball

### `Csv/Rige Body/Table Tennis_01_009.csv`
- `Tennis`: uncertain - has isolated implausible speed (708.363 m/s); dynamic enough to inspect, but not enough evidence for valid ball
