# CleanSample Audit Report

Input manifest: `analysis/mocap_cleaning_outputs/DATA260703_point_stitched/manifest.json`

| Metric | Value |
|---|---:|
| Samples | 455 |
| Audit OK | 441 |
| Audit rejected | 14 |
| Duplicate groups | 0 |
| Ball speed p95 of sample max | 13.766 |
| Ball acceleration p95 of sample p99 | 940.535 |

## Rejection Reasons

- `ball_acc_p99_ge_1000`: 10
- `flagged_not_usable`: 2
- `nonfinite_ball_pos`: 2
- `nonfinite_ball_vel`: 2
- `racket_omega_ge_40`: 2

## Worst Samples

- `Table_Tennis_01_014_TennisBats01_74p20_76p20_Skeleton001`: audit_ok=False, hit_dist=0.067, max_ball_speed=4.47, p99_acc=341.5, max_omega=67.8, reasons=racket_omega_ge_40,flagged_not_usable
- `Table_Tennis_01_004_TennisBats01_101p43_103p43_Skeleton001`: audit_ok=False, hit_dist=0.060, max_ball_speed=14.77, p99_acc=597.6, max_omega=43.2, reasons=racket_omega_ge_40,flagged_not_usable
- `Table_Tennis_01_012_TennisBats02_0p00_1p46_Skeleton002`: audit_ok=False, hit_dist=0.049, max_ball_speed=6.30, p99_acc=467.8, max_omega=7.8, reasons=nonfinite_ball_pos,nonfinite_ball_vel
- `Table_Tennis_01_013_TennisBats02_0p00_1p59_Skeleton002`: audit_ok=False, hit_dist=0.032, max_ball_speed=5.64, p99_acc=331.4, max_omega=27.5, reasons=nonfinite_ball_pos,nonfinite_ball_vel
- `Table_Tennis_01_013_TennisBats02_125p91_127p91_Skeleton002`: audit_ok=False, hit_dist=0.061, max_ball_speed=14.32, p99_acc=1045.5, max_omega=12.3, reasons=ball_acc_p99_ge_1000
- `Table_Tennis_01_013_TennisBats02_109p55_111p55_Skeleton002`: audit_ok=False, hit_dist=0.059, max_ball_speed=13.82, p99_acc=1607.0, max_omega=11.4, reasons=ball_acc_p99_ge_1000
- `Table_Tennis_01_013_TennisBats02_29p15_31p15_Skeleton002`: audit_ok=False, hit_dist=0.049, max_ball_speed=13.98, p99_acc=1151.7, max_omega=5.4, reasons=ball_acc_p99_ge_1000
- `Table_Tennis_01_014_TennisBats01_42p51_44p51_Skeleton001`: audit_ok=False, hit_dist=0.044, max_ball_speed=13.48, p99_acc=1037.2, max_omega=11.4, reasons=ball_acc_p99_ge_1000
- `Table_Tennis_01_013_TennisBats02_108p33_110p33_Skeleton002`: audit_ok=False, hit_dist=0.044, max_ball_speed=12.53, p99_acc=1027.7, max_omega=5.9, reasons=ball_acc_p99_ge_1000
- `Table_Tennis_01_013_TennisBats02_41p11_43p11_Skeleton002`: audit_ok=False, hit_dist=0.042, max_ball_speed=14.48, p99_acc=1783.2, max_omega=7.2, reasons=ball_acc_p99_ge_1000
- `Table_Tennis_01_014_TennisBats01_46p04_48p04_Skeleton001`: audit_ok=False, hit_dist=0.031, max_ball_speed=13.79, p99_acc=1288.4, max_omega=11.4, reasons=ball_acc_p99_ge_1000
- `Table_Tennis_01_014_TennisBats01_48p38_50p38_Skeleton001`: audit_ok=False, hit_dist=0.031, max_ball_speed=12.78, p99_acc=1128.6, max_omega=5.8, reasons=ball_acc_p99_ge_1000
- `Table_Tennis_01_013_TennisBats01_71p88_73p88_Skeleton001`: audit_ok=False, hit_dist=0.031, max_ball_speed=13.17, p99_acc=1125.2, max_omega=10.8, reasons=ball_acc_p99_ge_1000
- `Table_Tennis_01_013_TennisBats01_94p74_96p74_Skeleton001`: audit_ok=False, hit_dist=0.031, max_ball_speed=11.03, p99_acc=1140.9, max_omega=11.5, reasons=ball_acc_p99_ge_1000
- `Table_Tennis_01_004_TennisBats02_38p36_40p36_Skeleton002`: audit_ok=True, hit_dist=0.117, max_ball_speed=4.56, p99_acc=374.5, max_omega=11.0, reasons=none
- `Table_Tennis_01_012_TennisBats01_139p79_141p79_Skeleton001`: audit_ok=True, hit_dist=0.099, max_ball_speed=6.19, p99_acc=466.7, max_omega=7.1, reasons=none
- `Table_Tennis_01_013_TennisBats01_66p52_68p52_Skeleton001`: audit_ok=True, hit_dist=0.092, max_ball_speed=4.62, p99_acc=340.4, max_omega=7.8, reasons=none
- `Table_Tennis_01_004_TennisBats01_10p94_12p94_Skeleton001`: audit_ok=True, hit_dist=0.088, max_ball_speed=8.51, p99_acc=535.3, max_omega=9.4, reasons=none
- `Table_Tennis_01_014_TennisBats02_61p78_63p78_Skeleton002`: audit_ok=True, hit_dist=0.087, max_ball_speed=4.90, p99_acc=411.5, max_omega=7.9, reasons=none
- `Table_Tennis_01_004_TennisBats01_81p24_83p24_Skeleton001`: audit_ok=True, hit_dist=0.085, max_ball_speed=7.20, p99_acc=397.4, max_omega=9.3, reasons=none
