# DATA260708 Competition Core First Analysis

Dataset: `datasets_ready/DATA260708_competition_core.npz`

## Overview

| Metric | Value |
|---|---:|
| Samples | 1034 |
| Success | 950 |
| Failure | 84 |
| Success rate | 0.919 |

## Success By Racket

| racket | total | success | failure | success rate |
|---|---:|---:|---:|---:|
| gao01 | 520 | 475 | 45 | 0.913 |
| liang01 | 514 | 475 | 39 | 0.924 |

## Success By Stroke Type

| stroke | total | success | failure | success rate |
|---|---:|---:|---:|---:|
| backhand | 330 | 309 | 21 | 0.936 |
| forehand | 497 | 439 | 58 | 0.883 |
| unknown | 207 | 202 | 5 | 0.976 |

Note: `unknown` is a conservative rule-label bucket, not a stroke class. Its high success rate should not be interpreted as an action advantage without manual review.

## Position And Speed Summary

- `hit_x_m`: p10=-0.0793, p50=2.2449, p90=2.7972
- `hit_y_m`: p10=-1.2335, p50=-0.7272, p90=-0.2750
- `hit_z_m`: p10=0.2096, p50=0.2899, p90=0.3713
- `landing_x_m`: p10=0.4935, p50=1.3965, p90=2.2560
- `landing_y_m`: p10=-1.0607, p50=-0.7164, p90=-0.3914
- `hit_distance_ball_to_racket_center_m`: p10=0.0295, p50=0.0403, p90=0.0621
- `racket_speed_at_hit_mps`: p10=1.3885, p50=1.9513, p90=2.9571
- `ball_in_speed_mps`: p10=1.9031, p50=2.7938, p90=3.6860
- `ball_out_speed_mps`: p10=4.1239, p50=4.8786, p90=5.8425

## Landing Zones

| x bin | y bin | total | success | failure |
|---|---|---:|---:|---:|
| 1.83-2.28 | -0.76--0.38 | 211 | 185 | 26 |
| 0.46-0.91 | -1.14--0.76 | 201 | 184 | 17 |
| 0.46-0.91 | -0.76--0.38 | 109 | 104 | 5 |
| 1.83-2.28 | -1.14--0.76 | 81 | 74 | 7 |
| 2.28-2.74 | -0.76--0.38 | 53 | 50 | 3 |
| 0.00-0.46 | -1.14--0.76 | 51 | 47 | 4 |
| 1.83-2.28 | -0.38-0.00 | 44 | 44 | 0 |
| 1.37-1.83 | -0.76--0.38 | 41 | 38 | 3 |
| 0.91-1.37 | -0.76--0.38 | 36 | 32 | 4 |
| 0.91-1.37 | -1.14--0.76 | 32 | 28 | 4 |
| 0.00-0.46 | -0.76--0.38 | 22 | 21 | 1 |
| 1.37-1.83 | -1.14--0.76 | 22 | 20 | 2 |
| 0.46-0.91 | -1.52--1.14 | 21 | 20 | 1 |
| 0.46-0.91 | -0.38-0.00 | 18 | 18 | 0 |
| 2.28-2.74 | -1.14--0.76 | 18 | 18 | 0 |
| 1.37-1.83 | -0.38-0.00 | 15 | 14 | 1 |
| 2.28-2.74 | -0.38-0.00 | 14 | 13 | 1 |
| 1.83-2.28 | -1.52--1.14 | 12 | 12 | 0 |
| 0.00-0.46 | -1.52--1.14 | 8 | 8 | 0 |
| 0.91-1.37 | -1.52--1.14 | 6 | 6 | 0 |
| 0.91-1.37 | -0.38-0.00 | 5 | 4 | 1 |
| 1.37-1.83 | -1.52--1.14 | 5 | 3 | 2 |
| 2.28-2.74 | -1.52--1.14 | 4 | 4 | 0 |
| 0.00-0.46 | -0.38-0.00 | 3 | 3 | 0 |

## Output Tables

- `success_by_racket`: `data/analysis/reports/DATA260708_competition_core_analysis/success_by_racket.csv`
- `success_by_stroke`: `data/analysis/reports/DATA260708_competition_core_analysis/success_by_stroke.csv`
- `success_by_racket_speed`: `data/analysis/reports/DATA260708_competition_core_analysis/success_by_racket_speed.csv`
- `success_by_ball_out_speed`: `data/analysis/reports/DATA260708_competition_core_analysis/success_by_ball_out_speed.csv`
- `success_by_hit_distance`: `data/analysis/reports/DATA260708_competition_core_analysis/success_by_hit_distance.csv`
- `landing_zones`: `data/analysis/reports/DATA260708_competition_core_analysis/landing_zones.csv`
