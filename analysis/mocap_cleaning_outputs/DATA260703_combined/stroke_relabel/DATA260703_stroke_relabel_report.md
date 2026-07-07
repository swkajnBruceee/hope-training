# DATA260703 Stroke Relabel Report

This report is generated from body-local swing features. Player/racket id is not used as a classifier input.

- samples: 792
- input old labels: `{'forehand': 726, 'unknown': 66}`
- output labels: `{'forehand': 419, 'backhand': 298, 'unknown': 75}`

### Output Label Counts

| value | count |
|---|---:|
| forehand | 419 |
| backhand | 298 |
| unknown | 75 |

### Output By Racket

| group | forehand | backhand | unknown | total |
|---|---:|---:|---:|---:|
| TennisBats01 | 383 | 5 | 13 | 401 |
| TennisBats02 | 36 | 293 | 62 | 391 |

### Output By Skeleton

| group | forehand | backhand | unknown | total |
|---|---:|---:|---:|---:|
| Skeleton001 | 383 | 5 | 13 | 401 |
| Skeleton002 | 36 | 293 | 62 | 391 |

### Output By Source CSV

| group | forehand | backhand | unknown | total |
|---|---:|---:|---:|---:|
| Csv/Point/Table Tennis_01_004.csv | 62 | 0 | 8 | 70 |
| Csv/Point/Table Tennis_01_012.csv | 21 | 21 | 7 | 49 |
| Csv/Point/Table Tennis_01_013.csv | 73 | 65 | 14 | 152 |
| Csv/Point/Table Tennis_01_014.csv | 80 | 79 | 11 | 170 |
| Csv/Rige Body/Table Tennis_01_005.csv | 23 | 9 | 3 | 35 |
| Csv/Rige Body/Table Tennis_01_006.csv | 40 | 34 | 8 | 82 |
| Csv/Rige Body/Table Tennis_01_007.csv | 32 | 20 | 10 | 62 |
| Csv/Rige Body/Table Tennis_01_008.csv | 28 | 22 | 8 | 58 |
| Csv/Rige Body/Table Tennis_01_009.csv | 60 | 48 | 6 | 114 |

## Feature Summary

| label | n | median lateral offset m | median lateral velocity m/s | median pre-to-hit delta m | median confidence |
|---|---:|---:|---:|---:|---:|
| forehand | 419 | 0.3980 | -1.5058 | -0.0961 | 1.000 |
| backhand | 298 | 0.1451 | 0.5494 | 0.0290 | 1.000 |
| unknown | 75 | 0.1696 | 0.1397 | -0.0031 | 0.000 |

## Boundary Samples

Boundary sample count (`unknown` or confidence < 0.70): 75

| idx | episode_id | label | conf | racket | source | lat_off | lat_vel | delta | reason |
|---:|---|---|---:|---|---|---:|---:|---:|---|
| 24 | Table_Tennis_01_005_TennisBats02_1p94_3p94_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_005.csv | 0.1880 | 0.0717 | 0.0020 | weak_motion_evidence |
| 28 | Table_Tennis_01_005_TennisBats02_50p52_52p52_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_005.csv | 0.1113 | 0.1661 | -0.0113 | weak_motion_evidence |
| 31 | Table_Tennis_01_005_TennisBats02_54p47_56p47_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_005.csv | 0.1946 | 0.2349 | 0.0037 | weak_motion_evidence |
| 56 | Table_Tennis_01_006_TennisBats01_47p65_49p65_Skeleton001 | unknown | 0.000 | TennisBats01 | Csv/Rige Body/Table Tennis_01_006.csv | 0.3704 | -0.2374 | -0.0094 | weak_motion_evidence |
| 66 | Table_Tennis_01_006_TennisBats01_70p92_72p92_Skeleton001 | unknown | 0.000 | TennisBats01 | Csv/Rige Body/Table Tennis_01_006.csv | 0.6225 | 0.0047 | 0.0087 | weak_motion_evidence |
| 78 | Table_Tennis_01_006_TennisBats02_0p69_2p69_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_006.csv | 0.2885 | 0.2315 | 0.0088 | weak_motion_evidence |
| 79 | Table_Tennis_01_006_TennisBats02_11p26_13p26_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_006.csv | 0.1135 | 0.2000 | -0.0099 | weak_motion_evidence |
| 90 | Table_Tennis_01_006_TennisBats02_37p71_39p71_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_006.csv | 0.2746 | 0.0414 | -0.0067 | weak_motion_evidence |
| 91 | Table_Tennis_01_006_TennisBats02_38p94_40p94_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_006.csv | 0.0673 | -0.1179 | -0.0190 | weak_motion_evidence |
| 95 | Table_Tennis_01_006_TennisBats02_42p66_44p66_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_006.csv | 0.1879 | 0.1839 | 0.0071 | weak_motion_evidence |
| 101 | Table_Tennis_01_006_TennisBats02_56p24_58p24_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_006.csv | 0.1234 | 0.0937 | -0.0059 | weak_motion_evidence |
| 148 | Table_Tennis_01_007_TennisBats02_11p42_13p42_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_007.csv | 0.1185 | 0.2131 | 0.0026 | weak_motion_evidence |
| 151 | Table_Tennis_01_007_TennisBats02_22p98_24p98_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_007.csv | 0.1245 | 0.0321 | -0.0196 | weak_motion_evidence |
| 153 | Table_Tennis_01_007_TennisBats02_26p59_28p59_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_007.csv | 0.1948 | 0.1835 | -0.0155 | weak_motion_evidence |
| 156 | Table_Tennis_01_007_TennisBats02_31p63_33p63_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_007.csv | 0.1754 | 0.1056 | 0.0112 | weak_motion_evidence |
| 157 | Table_Tennis_01_007_TennisBats02_38p83_40p83_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_007.csv | 0.1313 | 0.2505 | 0.0096 | weak_motion_evidence |
| 163 | Table_Tennis_01_007_TennisBats02_53p98_55p98_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_007.csv | 0.1696 | 0.2702 | 0.0101 | weak_motion_evidence |
| 164 | Table_Tennis_01_007_TennisBats02_55p19_57p19_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_007.csv | 0.0828 | -0.0522 | -0.0156 | weak_motion_evidence |
| 166 | Table_Tennis_01_007_TennisBats02_57p70_59p70_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_007.csv | 0.1045 | 0.2845 | 0.0054 | weak_motion_evidence |
| 171 | Table_Tennis_01_007_TennisBats02_63p92_65p92_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_007.csv | 0.0530 | -0.1192 | -0.0139 | weak_motion_evidence |
| 178 | Table_Tennis_01_007_TennisBats02_89p86_91p86_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_007.csv | 0.1033 | 0.0662 | -0.0084 | weak_motion_evidence |
| 188 | Table_Tennis_01_008_TennisBats01_28p76_30p76_Skeleton001 | unknown | 0.000 | TennisBats01 | Csv/Rige Body/Table Tennis_01_008.csv | 0.6871 | -0.1963 | 0.0105 | weak_motion_evidence |
| 210 | Table_Tennis_01_008_TennisBats02_108p65_110p65_Skeleton002 | unknown | 0.350 | TennisBats02 | Csv/Rige Body/Table Tennis_01_008.csv | -0.0176 | 0.6097 | -0.0167 | ambiguous_body_local_swing |
| 217 | Table_Tennis_01_008_TennisBats02_17p46_19p46_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_008.csv | 0.2857 | -0.1970 | -0.0053 | weak_motion_evidence |
| 218 | Table_Tennis_01_008_TennisBats02_28p03_30p03_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_008.csv | 0.0591 | 0.0512 | -0.0137 | weak_motion_evidence |
| 220 | Table_Tennis_01_008_TennisBats02_36p40_38p40_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_008.csv | 0.1766 | 0.2401 | 0.0107 | weak_motion_evidence |
| 227 | Table_Tennis_01_008_TennisBats02_64p94_66p94_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_008.csv | 0.2453 | 0.2228 | -0.0106 | weak_motion_evidence |
| 230 | Table_Tennis_01_008_TennisBats02_7p43_9p43_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_008.csv | 0.1523 | 0.2699 | 0.0096 | weak_motion_evidence |
| 236 | Table_Tennis_01_008_TennisBats02_9p93_11p93_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_008.csv | 0.1929 | 0.1147 | -0.0076 | weak_motion_evidence |
| 247 | Table_Tennis_01_009_TennisBats01_113p96_115p96_Skeleton001 | unknown | 0.000 | TennisBats01 | Csv/Rige Body/Table Tennis_01_009.csv | 0.5914 | -0.1721 | 0.0069 | weak_motion_evidence |
| 260 | Table_Tennis_01_009_TennisBats01_135p91_137p91_Skeleton001 | unknown | 0.000 | TennisBats01 | Csv/Rige Body/Table Tennis_01_009.csv | 0.1870 | -0.0712 | 0.0059 | weak_motion_evidence |
| 300 | Table_Tennis_01_009_TennisBats02_101p77_103p77_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_009.csv | 0.1578 | 0.1916 | -0.0014 | weak_motion_evidence |
| 314 | Table_Tennis_01_009_TennisBats02_131p14_133p14_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_009.csv | 0.2527 | 0.2145 | -0.0052 | weak_motion_evidence |
| 340 | Table_Tennis_01_009_TennisBats02_69p11_71p11_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_009.csv | 0.1760 | 0.0429 | -0.0020 | weak_motion_evidence |
| 344 | Table_Tennis_01_009_TennisBats02_7p30_9p30_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Rige Body/Table Tennis_01_009.csv | 0.1072 | -0.0050 | -0.0134 | weak_motion_evidence |
| 356 | Table_Tennis_01_004_TennisBats01_21p34_23p34_Skeleton001 | unknown | 0.000 | TennisBats01 | Csv/Point/Table Tennis_01_004.csv | 0.0924 | 0.2229 | 0.0104 | weak_motion_evidence |
| 361 | Table_Tennis_01_004_TennisBats01_36p22_38p22_Skeleton001 | unknown | 0.000 | TennisBats01 | Csv/Point/Table Tennis_01_004.csv | 0.1551 | -0.0444 | -0.0031 | weak_motion_evidence |
| 367 | Table_Tennis_01_004_TennisBats01_52p86_54p86_Skeleton001 | unknown | 0.000 | TennisBats01 | Csv/Point/Table Tennis_01_004.csv | 0.0479 | 0.0397 | 0.0124 | weak_motion_evidence |
| 390 | Table_Tennis_01_004_TennisBats02_17p46_19p46_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_004.csv | 0.1286 | 0.1112 | -0.0248 | weak_motion_evidence |
| 392 | Table_Tennis_01_004_TennisBats02_21p93_23p93_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_004.csv | 0.1178 | 0.1397 | 0.0204 | weak_motion_evidence |
| 393 | Table_Tennis_01_004_TennisBats02_27p80_29p80_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_004.csv | 0.1864 | 0.0301 | 0.0007 | weak_motion_evidence |
| 396 | Table_Tennis_01_004_TennisBats02_36p78_38p78_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_004.csv | 0.0725 | 0.0525 | -0.0068 | weak_motion_evidence |
| 404 | Table_Tennis_01_004_TennisBats02_63p33_65p33_Skeleton002 | unknown | 0.250 | TennisBats02 | Csv/Point/Table Tennis_01_004.csv | 0.0657 | -0.0598 | -0.0487 | ambiguous_body_local_swing |
| 442 | Table_Tennis_01_012_TennisBats02_127p63_129p63_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_012.csv | 0.0105 | 0.2627 | 0.0069 | weak_motion_evidence |
| 448 | Table_Tennis_01_012_TennisBats02_17p87_19p87_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_012.csv | 0.1679 | -0.1423 | -0.0167 | weak_motion_evidence |
| 450 | Table_Tennis_01_012_TennisBats02_192p34_194p34_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_012.csv | 0.1652 | 0.1956 | -0.0021 | weak_motion_evidence |
| 451 | Table_Tennis_01_012_TennisBats02_19p12_21p12_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_012.csv | 0.0853 | 0.0762 | 0.0110 | weak_motion_evidence |
| 453 | Table_Tennis_01_012_TennisBats02_24p88_26p88_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_012.csv | 0.2180 | 0.0834 | -0.0152 | weak_motion_evidence |
| 461 | Table_Tennis_01_012_TennisBats02_52p26_54p26_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_012.csv | 0.2206 | 0.0109 | -0.0095 | weak_motion_evidence |
| 462 | Table_Tennis_01_012_TennisBats02_53p47_55p47_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_012.csv | 0.0572 | 0.2671 | 0.0017 | weak_motion_evidence |
| 493 | Table_Tennis_01_013_TennisBats01_155p92_157p92_Skeleton001 | unknown | 0.000 | TennisBats01 | Csv/Point/Table Tennis_01_013.csv | 0.4770 | -0.1350 | -0.0150 | weak_motion_evidence |
| 520 | Table_Tennis_01_013_TennisBats01_52p52_54p52_Skeleton001 | unknown | 0.000 | TennisBats01 | Csv/Point/Table Tennis_01_013.csv | 0.2321 | 0.0215 | -0.0052 | weak_motion_evidence |
| 541 | Table_Tennis_01_013_TennisBats01_92p17_94p17_Skeleton001 | unknown | 0.000 | TennisBats01 | Csv/Point/Table Tennis_01_013.csv | 0.1301 | 0.0209 | 0.0008 | weak_motion_evidence |
| 548 | Table_Tennis_01_013_TennisBats02_119p81_121p81_Skeleton002 | unknown | 0.200 | TennisBats02 | Csv/Point/Table Tennis_01_013.csv | 0.2183 | 0.3585 | -0.0182 | ambiguous_body_local_swing |
| 550 | Table_Tennis_01_013_TennisBats02_123p54_125p54_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_013.csv | 0.0834 | 0.0745 | 0.0032 | weak_motion_evidence |
| 558 | Table_Tennis_01_013_TennisBats02_141p68_143p68_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_013.csv | 0.2239 | 0.0559 | -0.0090 | weak_motion_evidence |
| 582 | Table_Tennis_01_013_TennisBats02_31p81_33p81_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_013.csv | 0.1424 | 0.2102 | 0.0100 | weak_motion_evidence |
| 591 | Table_Tennis_01_013_TennisBats02_43p62_45p62_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_013.csv | 0.1300 | 0.2278 | 0.0066 | weak_motion_evidence |
| 598 | Table_Tennis_01_013_TennisBats02_63p22_65p22_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_013.csv | 0.3023 | 0.0340 | -0.0215 | weak_motion_evidence |
| 600 | Table_Tennis_01_013_TennisBats02_65p82_67p82_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_013.csv | 0.2383 | 0.1604 | -0.0073 | weak_motion_evidence |
| 602 | Table_Tennis_01_013_TennisBats02_72p44_74p44_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_013.csv | 0.1847 | 0.1514 | -0.0085 | weak_motion_evidence |
| 604 | Table_Tennis_01_013_TennisBats02_74p92_76p92_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_013.csv | 0.1970 | 0.2521 | 0.0048 | weak_motion_evidence |
| 616 | Table_Tennis_01_013_TennisBats02_91p53_93p53_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_013.csv | 0.2362 | 0.2934 | -0.0030 | weak_motion_evidence |
| 618 | Table_Tennis_01_013_TennisBats02_94p04_96p04_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_013.csv | 0.1488 | 0.1649 | -0.0236 | weak_motion_evidence |
| 631 | Table_Tennis_01_014_TennisBats01_111p49_113p49_Skeleton001 | unknown | 0.000 | TennisBats01 | Csv/Point/Table Tennis_01_014.csv | 0.5669 | -0.0865 | -0.0174 | weak_motion_evidence |
| 636 | Table_Tennis_01_014_TennisBats01_133p36_135p36_Skeleton001 | unknown | 0.000 | TennisBats01 | Csv/Point/Table Tennis_01_014.csv | 0.5997 | 0.0545 | -0.0131 | weak_motion_evidence |
| 720 | Table_Tennis_01_014_TennisBats02_136p48_138p48_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_014.csv | 0.1760 | 0.2335 | 0.0070 | weak_motion_evidence |
| 723 | Table_Tennis_01_014_TennisBats02_150p94_152p94_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_014.csv | 0.1930 | 0.2235 | 0.0083 | weak_motion_evidence |
| 733 | Table_Tennis_01_014_TennisBats02_173p26_175p26_Skeleton002 | unknown | 0.050 | TennisBats02 | Csv/Point/Table Tennis_01_014.csv | -0.0178 | 0.2743 | -0.0319 | ambiguous_body_local_swing |
| 747 | Table_Tennis_01_014_TennisBats02_205p42_207p42_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_014.csv | 0.1144 | 0.2732 | 0.0074 | weak_motion_evidence |
| 764 | Table_Tennis_01_014_TennisBats02_51p33_53p33_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_014.csv | 0.1823 | 0.2615 | 0.0104 | weak_motion_evidence |
| 770 | Table_Tennis_01_014_TennisBats02_59p02_61p02_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_014.csv | 0.0804 | 0.1425 | -0.0042 | weak_motion_evidence |
| 773 | Table_Tennis_01_014_TennisBats02_61p78_63p78_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_014.csv | 0.2745 | 0.2909 | -0.0032 | weak_motion_evidence |
| 776 | Table_Tennis_01_014_TennisBats02_69p29_71p29_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_014.csv | 0.0893 | 0.2598 | 0.0071 | weak_motion_evidence |
| 784 | Table_Tennis_01_014_TennisBats02_7p84_9p84_Skeleton002 | unknown | 0.000 | TennisBats02 | Csv/Point/Table Tennis_01_014.csv | 0.2231 | 0.1803 | -0.0018 | weak_motion_evidence |

## Rule Notes

- Positive/negative labels are inferred from motion relative to `body_right_axis`, not from Motive/table axes.
- Racket id and skeleton id are used only in this report for auditing.
- Low-confidence samples are intentionally kept as `unknown` to avoid contaminating training labels.
- This relabeling can be run before table-frame calibration; table-frame calibration is still required for landing and success labels.
