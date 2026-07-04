# Mocap Dataset Analysis

Dataset: `data/DATA260703`

## Overview

- BVH files: 18
- CSV files: 9
- Total size: 7.32 GB

## BVH Files

| File | Frames | FPS | Duration (s) | Roots | Joints | Size (MB) |
|---|---:|---:|---:|---:|---:|---:|
| `Bvh/Point/Table Tennis_01_004_Skeleton 001.bvh` | 37706 | 360.0 | 104.7 | 1 | 50 | 75.1 |
| `Bvh/Point/Table Tennis_01_004_Skeleton 002.bvh` | 37706 | 360.0 | 104.7 | 1 | 50 | 75.5 |
| `Bvh/Point/Table Tennis_01_012_Skeleton 001.bvh` | 73334 | 360.0 | 203.7 | 1 | 50 | 146.5 |
| `Bvh/Point/Table Tennis_01_012_Skeleton 002.bvh` | 73334 | 360.0 | 203.7 | 1 | 50 | 146.6 |
| `Bvh/Point/Table Tennis_01_013_Skeleton 001.bvh` | 59935 | 360.0 | 166.5 | 1 | 50 | 119.8 |
| `Bvh/Point/Table Tennis_01_013_Skeleton 002.bvh` | 59935 | 360.0 | 166.5 | 1 | 50 | 120.1 |
| `Bvh/Point/Table Tennis_01_014_Skeleton 001.bvh` | 79271 | 360.0 | 220.2 | 1 | 50 | 158.3 |
| `Bvh/Point/Table Tennis_01_014_Skeleton 002.bvh` | 79271 | 360.0 | 220.2 | 1 | 50 | 158.6 |
| `Bvh/Rige Body/Table Tennis_01_005_Skeleton 001.bvh` | 33431 | 360.0 | 92.9 | 1 | 50 | 66.7 |
| `Bvh/Rige Body/Table Tennis_01_005_Skeleton 002.bvh` | 33431 | 360.0 | 92.9 | 1 | 50 | 66.9 |
| `Bvh/Rige Body/Table Tennis_01_006_Skeleton 001.bvh` | 36158 | 360.0 | 100.4 | 1 | 50 | 72.2 |
| `Bvh/Rige Body/Table Tennis_01_006_Skeleton 002.bvh` | 36158 | 360.0 | 100.4 | 1 | 50 | 72.4 |
| `Bvh/Rige Body/Table Tennis_01_007_Skeleton 001.bvh` | 33738 | 360.0 | 93.7 | 1 | 50 | 67.4 |
| `Bvh/Rige Body/Table Tennis_01_007_Skeleton 002.bvh` | 33738 | 360.0 | 93.7 | 1 | 50 | 67.5 |
| `Bvh/Rige Body/Table Tennis_01_008_Skeleton 001.bvh` | 46270 | 360.0 | 128.5 | 1 | 50 | 92.4 |
| `Bvh/Rige Body/Table Tennis_01_008_Skeleton 002.bvh` | 46270 | 360.0 | 128.5 | 1 | 50 | 92.5 |
| `Bvh/Rige Body/Table Tennis_01_009_Skeleton 001.bvh` | 50982 | 360.0 | 141.6 | 1 | 50 | 101.9 |
| `Bvh/Rige Body/Table Tennis_01_009_Skeleton 002.bvh` | 50982 | 360.0 | 141.6 | 1 | 50 | 102.0 |

## CSV Files

| File | Frames | FPS | Duration (s) | Rigid Bodies | Markers | Size (MB) |
|---|---:|---:|---:|---|---:|---:|
| `Csv/Point/Table Tennis_01_004.csv` | 37706 | 360.0 | 104.7 | TennisBats01, TennisBats02 | 156 | 420.3 |
| `Csv/Point/Table Tennis_01_012.csv` | 73334 | 360.0 | 203.7 | TennisBats01, TennisBats02 | 537 | 870.8 |
| `Csv/Point/Table Tennis_01_013.csv` | 59935 | 360.0 | 166.5 | TennisBats01, TennisBats02 | 257 | 694.3 |
| `Csv/Point/Table Tennis_01_014.csv` | 79271 | 360.0 | 220.2 | TennisBats01, TennisBats02 | 299 | 924.6 |
| `Csv/Rige Body/Table Tennis_01_005.csv` | 33431 | 360.0 | 92.9 | TennisBats01, TennisBats02, Tennis | 887 | 456.7 |
| `Csv/Rige Body/Table Tennis_01_006.csv` | 36158 | 360.0 | 100.4 | TennisBats01, TennisBats02, Tennis | 390 | 439.0 |
| `Csv/Rige Body/Table Tennis_01_007.csv` | 33738 | 360.0 | 93.7 | TennisBats01, TennisBats02, Tennis | 663 | 433.5 |
| `Csv/Rige Body/Table Tennis_01_008.csv` | 46270 | 360.0 | 128.5 | TennisBats01, TennisBats02, Tennis | 635 | 596.3 |
| `Csv/Rige Body/Table Tennis_01_009.csv` | 50982 | 360.0 | 141.6 | TennisBats01, TennisBats02, Tennis | 788 | 680.6 |

## Racket Speed Peaks

The windows below are coarse 1.5 s candidates centered on sampled racket speed peaks.

| File | Racket | Peak Speed (m/s) | Time (s) | Suggested Window (s) |
|---|---|---:|---:|---|
| `Csv/Point/Table Tennis_01_004.csv` | TennisBats01 | 5.14 | 102.47 | 101.72-103.22 |
| `Csv/Point/Table Tennis_01_004.csv` | TennisBats02 | 5.48 | 52.86 | 52.11-53.61 |
| `Csv/Point/Table Tennis_01_012.csv` | TennisBats01 | 5.16 | 116.69 | 115.94-117.44 |
| `Csv/Point/Table Tennis_01_012.csv` | TennisBats02 | 4.00 | 139.72 | 138.97-140.47 |
| `Csv/Point/Table Tennis_01_013.csv` | TennisBats01 | 4.60 | 37.58 | 36.83-38.33 |
| `Csv/Point/Table Tennis_01_013.csv` | TennisBats02 | 3.59 | 119.42 | 118.67-120.17 |
| `Csv/Point/Table Tennis_01_014.csv` | TennisBats01 | 4.14 | 24.47 | 23.72-25.22 |
| `Csv/Point/Table Tennis_01_014.csv` | TennisBats02 | 3.80 | 160.56 | 159.81-161.31 |
| `Csv/Rige Body/Table Tennis_01_005.csv` | TennisBats01 | 5.52 | 91.11 | 90.36-91.86 |
| `Csv/Rige Body/Table Tennis_01_005.csv` | TennisBats02 | 3.55 | 72.25 | 71.50-73.00 |
| `Csv/Rige Body/Table Tennis_01_006.csv` | TennisBats01 | 4.62 | 13.03 | 12.28-13.78 |
| `Csv/Rige Body/Table Tennis_01_006.csv` | TennisBats02 | 3.89 | 80.22 | 79.47-80.97 |
| `Csv/Rige Body/Table Tennis_01_007.csv` | TennisBats01 | 4.72 | 64.44 | 63.69-65.19 |
| `Csv/Rige Body/Table Tennis_01_007.csv` | TennisBats02 | 29.20 | 2.72 | 1.97-3.47 |
| `Csv/Rige Body/Table Tennis_01_008.csv` | TennisBats01 | 4.67 | 64.22 | 63.47-64.97 |
| `Csv/Rige Body/Table Tennis_01_008.csv` | TennisBats02 | 3.69 | 71.75 | 71.00-72.50 |
| `Csv/Rige Body/Table Tennis_01_009.csv` | TennisBats01 | 5.13 | 139.89 | 139.14-140.64 |
| `Csv/Rige Body/Table Tennis_01_009.csv` | TennisBats02 | 4.21 | 107.64 | 106.89-108.39 |

## Practical Read

- Use BVH as the main retargeting input; it is already split by skeleton.
- Use CSV rigid bodies to locate swing/contact candidates and to validate racket motion offline.
- Before training, cut one-person BVH clips around selected windows, then retarget to A3 joint space.
