# DATA260703 Cleaned Tennis Windows

Selected manifest: `analysis/mocap/selected_clips/manifest.json`

| Clip | Racket | Raw Max Speed | Clean Max Speed | Clean P95 Speed | Outliers | Filled Gaps | Long Gaps | Valid Ratio | Min Dist | Usable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Table_Tennis_01_005_TennisBats01_90p11_92p11_Skeleton001.bvh` | TennisBats01 | 17.92 | 17.92 | 5.73 | 0 | 0 | 0 | 1.000 | 0.116 | True |
| `Table_Tennis_01_005_TennisBats02_71p25_73p25_Skeleton002.bvh` | TennisBats02 | 142.22 | 94.81 | 5.42 | 2 | 1 | 0 | 1.000 | 0.041 | False |
| `Table_Tennis_01_006_TennisBats01_12p03_14p03_Skeleton001.bvh` | TennisBats01 | 29.52 | 29.52 | 7.40 | 0 | 0 | 0 | 1.000 | 0.044 | True |
| `Table_Tennis_01_006_TennisBats02_79p22_81p22_Skeleton002.bvh` | TennisBats02 | 72.93 | 48.61 | 5.58 | 2 | 1 | 0 | 1.000 | 0.051 | True |
| `Table_Tennis_01_007_TennisBats01_63p44_65p44_Skeleton001.bvh` | TennisBats01 | 21.63 | 21.63 | 7.97 | 0 | 0 | 0 | 1.000 | 0.033 | True |
| `Table_Tennis_01_008_TennisBats01_63p22_65p22_Skeleton001.bvh` | TennisBats01 | 25.87 | 25.87 | 7.15 | 0 | 0 | 0 | 1.000 | 0.071 | True |
| `Table_Tennis_01_008_TennisBats02_70p75_72p75_Skeleton002.bvh` | TennisBats02 | 114.03 | 76.03 | 6.33 | 2 | 1 | 0 | 1.000 | 0.071 | False |
| `Table_Tennis_01_009_TennisBats01_138p89_140p89_Skeleton001.bvh` | TennisBats01 | 18.62 | 18.62 | 6.41 | 0 | 0 | 0 | 1.000 | 0.040 | True |
| `Table_Tennis_01_009_TennisBats02_106p64_108p64_Skeleton002.bvh` | TennisBats02 | 24.50 | 24.50 | 7.87 | 0 | 0 | 0 | 1.000 | 0.051 | True |

## Rejection / Warning Reasons

- `Table_Tennis_01_005_TennisBats01_90p11_92p11_Skeleton001.bvh`: trajectory cleaned successfully
- `Table_Tennis_01_005_TennisBats02_71p25_73p25_Skeleton002.bvh`: cleaned trajectory still exceeds 50.0 m/s
- `Table_Tennis_01_006_TennisBats01_12p03_14p03_Skeleton001.bvh`: trajectory cleaned successfully
- `Table_Tennis_01_006_TennisBats02_79p22_81p22_Skeleton002.bvh`: trajectory cleaned successfully
- `Table_Tennis_01_007_TennisBats01_63p44_65p44_Skeleton001.bvh`: trajectory cleaned successfully
- `Table_Tennis_01_008_TennisBats01_63p22_65p22_Skeleton001.bvh`: trajectory cleaned successfully
- `Table_Tennis_01_008_TennisBats02_70p75_72p75_Skeleton002.bvh`: cleaned trajectory still exceeds 50.0 m/s
- `Table_Tennis_01_009_TennisBats01_138p89_140p89_Skeleton001.bvh`: trajectory cleaned successfully
- `Table_Tennis_01_009_TennisBats02_106p64_108p64_Skeleton002.bvh`: trajectory cleaned successfully
