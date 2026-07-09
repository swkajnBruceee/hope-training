# DATA260703 Stroke Relabel Report

This report is generated from body-local swing features. Player/racket id is not used as a classifier input.

- samples: 1053
- input old labels: `{'forehand': 877, 'unknown': 170, 'backhand': 6}`
- output labels: `{'forehand': 507, 'backhand': 335, 'unknown': 211}`

### Output Label Counts

| value | count |
|---|---:|
| forehand | 507 |
| backhand | 335 |
| unknown | 211 |

### Output By Racket

| group | forehand | backhand | unknown | total |
|---|---:|---:|---:|---:|
| gao01 | 265 | 189 | 75 | 529 |
| liang01 | 242 | 146 | 136 | 524 |

### Output By Skeleton

| group | forehand | backhand | unknown | total |
|---|---:|---:|---:|---:|
| unknown | 507 | 335 | 211 | 1053 |

### Output By Source CSV

| group | forehand | backhand | unknown | total |
|---|---:|---:|---:|---:|
| CSV/T001_001.csv | 7 | 4 | 2 | 13 |
| CSV/T001_002.csv | 8 | 0 | 0 | 8 |
| CSV/T001_003.csv | 10 | 0 | 1 | 11 |
| CSV/T002_001.csv | 5 | 0 | 1 | 6 |
| CSV/T002_006.csv | 12 | 1 | 3 | 16 |
| CSV/T002_008.csv | 4 | 2 | 2 | 8 |
| CSV/T002_010.csv | 9 | 1 | 5 | 15 |
| CSV/T002_012.csv | 10 | 1 | 1 | 12 |
| CSV/T002_013.csv | 11 | 0 | 5 | 16 |
| CSV/T002_015.csv | 8 | 4 | 4 | 16 |
| CSV/T002_019.csv | 11 | 1 | 4 | 16 |
| CSV/T002_021.csv | 12 | 1 | 3 | 16 |
| CSV/T002_022.csv | 0 | 16 | 0 | 16 |
| CSV/T002_023.csv | 0 | 14 | 2 | 16 |
| CSV/T002_024.csv | 0 | 14 | 2 | 16 |
| CSV/T002_027.csv | 0 | 13 | 3 | 16 |
| CSV/T002_028.csv | 0 | 13 | 3 | 16 |
| CSV/T03_002.csv | 13 | 2 | 1 | 16 |
| CSV/T03_003.csv | 14 | 0 | 2 | 16 |
| CSV/T03_004.csv | 7 | 0 | 2 | 9 |
| CSV/T03_005.csv | 14 | 0 | 2 | 16 |
| CSV/T03_007.csv | 8 | 0 | 1 | 9 |
| CSV/T03_008.csv | 12 | 2 | 1 | 15 |
| CSV/T03_011.csv | 10 | 0 | 1 | 11 |
| CSV/T03_012.csv | 14 | 0 | 2 | 16 |
| CSV/T03_013.csv | 15 | 1 | 0 | 16 |
| CSV/T03_014.csv | 10 | 2 | 4 | 16 |
| CSV/T03_015.csv | 11 | 0 | 5 | 16 |
| CSV/T03_016.csv | 9 | 0 | 3 | 12 |
| CSV/T03_019.csv | 12 | 1 | 3 | 16 |
| CSV/T03_022.csv | 10 | 1 | 4 | 15 |
| CSV/T03_025.csv | 13 | 0 | 3 | 16 |
| CSV/T03_027.csv | 2 | 6 | 3 | 11 |
| CSV/T03_028.csv | 1 | 4 | 2 | 7 |
| CSV/T03_030.csv | 0 | 13 | 1 | 14 |
| CSV/T03_031.csv | 0 | 12 | 1 | 13 |
| CSV/T03_032.csv | 0 | 16 | 0 | 16 |
| CSV/T03_034.csv | 0 | 11 | 0 | 11 |
| CSV/T03_037.csv | 2 | 10 | 2 | 14 |
| CSV/T03_038.csv | 1 | 6 | 2 | 9 |
| CSV/T03_039.csv | 1 | 5 | 1 | 7 |
| CSV/T03_044.csv | 16 | 0 | 0 | 16 |
| CSV/T03_045.csv | 5 | 0 | 4 | 9 |
| CSV/T03_050.csv | 15 | 0 | 1 | 16 |
| CSV/T03_052.csv | 13 | 0 | 0 | 13 |
| CSV/T03_053.csv | 14 | 0 | 2 | 16 |
| CSV/T03_059.csv | 8 | 0 | 0 | 8 |
| CSV/T03_065.csv | 15 | 0 | 1 | 16 |
| CSV/T03_067.csv | 8 | 0 | 1 | 9 |
| CSV/T03_068.csv | 14 | 0 | 1 | 15 |
| CSV/T03_069.csv | 8 | 0 | 1 | 9 |
| CSV/T03_071.csv | 3 | 0 | 4 | 7 |
| CSV/T03_072.csv | 1 | 5 | 3 | 9 |
| CSV/T03_075.csv | 3 | 11 | 2 | 16 |
| CSV/T03_076.csv | 4 | 10 | 1 | 15 |
| CSV/T03_077.csv | 1 | 6 | 3 | 10 |
| CSV/T03_078.csv | 4 | 6 | 2 | 12 |
| CSV/T03_079.csv | 2 | 13 | 0 | 15 |
| CSV/T03_080.csv | 1 | 6 | 1 | 8 |
| CSV/T03_083.csv | 3 | 10 | 3 | 16 |
| CSV/T03_084.csv | 2 | 3 | 0 | 5 |
| CSV/T04_001.csv | 2 | 1 | 9 | 12 |
| CSV/T04_005.csv | 4 | 0 | 4 | 8 |
| CSV/T04_006.csv | 2 | 0 | 2 | 4 |
| CSV/T04_007.csv | 1 | 0 | 5 | 6 |
| CSV/T04_009.csv | 7 | 0 | 8 | 15 |
| CSV/T04_010.csv | 0 | 3 | 4 | 7 |
| CSV/T04_014.csv | 4 | 0 | 7 | 11 |
| CSV/T04_018.csv | 7 | 0 | 9 | 16 |
| CSV/T04_021.csv | 6 | 0 | 9 | 15 |
| CSV/T04_023.csv | 6 | 0 | 9 | 15 |
| CSV/T04_024.csv | 0 | 0 | 6 | 6 |
| CSV/T04_026.csv | 1 | 0 | 8 | 9 |
| CSV/T_006.csv | 0 | 14 | 2 | 16 |
| CSV/T_009.csv | 1 | 14 | 1 | 16 |
| CSV/T_010.csv | 1 | 13 | 1 | 15 |
| CSV/T_011.csv | 1 | 11 | 4 | 16 |
| CSV/T_013.csv | 1 | 14 | 0 | 15 |
| CSV/T_014.csv | 0 | 15 | 0 | 15 |
| CSV/T_018.csv | 6 | 0 | 2 | 8 |
| CSV/T_019.csv | 10 | 2 | 0 | 12 |
| CSV/T_020.csv | 12 | 1 | 3 | 16 |
| CSV/T_021.csv | 9 | 0 | 1 | 10 |

## Feature Summary

| label | n | median lateral offset m | median lateral velocity m/s | median pre-to-hit delta m | median confidence |
|---|---:|---:|---:|---:|---:|
| forehand | 507 | 0.4681 | -0.7252 | -0.0557 | 1.000 |
| backhand | 335 | 0.1843 | 0.7512 | 0.0493 | 1.000 |
| unknown | 211 | 0.5421 | 0.0390 | -0.0033 | 0.000 |

## Boundary Samples

Boundary sample count (`unknown` or confidence < 0.70): 211

| idx | episode_id | label | conf | racket | source | lat_off | lat_vel | delta | reason |
|---:|---|---|---:|---|---|---:|---:|---:|---|
| 5 | T001_001_gao01_7p13_9p13 | unknown | 0.000 | gao01 | CSV/T001_001.csv | 0.8018 | 0.0231 | 0.0004 | weak_motion_evidence |
| 7 | T001_001_liang01_1p31_3p31 | unknown | 0.000 | liang01 | CSV/T001_001.csv | 0.7344 | -0.0352 | -0.0174 | weak_motion_evidence |
| 21 | T001_003_gao01_1p53_3p53 | unknown | 0.000 | gao01 | CSV/T001_003.csv | 0.6233 | -0.0494 | -0.0098 | weak_motion_evidence |
| 33 | T002_001_gao01_6p17_8p17 | unknown | 0.000 | gao01 | CSV/T002_001.csv | 0.7014 | 0.1427 | 0.0084 | weak_motion_evidence |
| 47 | T002_006_liang01_14p00_16p00 | unknown | 0.000 | liang01 | CSV/T002_006.csv | 0.6581 | 0.0278 | -0.0141 | weak_motion_evidence |
| 50 | T002_006_liang01_23p92_25p92 | unknown | 0.000 | liang01 | CSV/T002_006.csv | 0.4770 | 0.2765 | 0.0071 | weak_motion_evidence |
| 51 | T002_006_liang01_29p53_31p53 | unknown | 0.000 | liang01 | CSV/T002_006.csv | 0.5967 | -0.0479 | -0.0222 | weak_motion_evidence |
| 54 | T002_008_gao01_13p03_15p03 | unknown | 0.000 | gao01 | CSV/T002_008.csv | 0.6781 | 0.0193 | -0.0042 | weak_motion_evidence |
| 57 | T002_008_gao01_21p20_23p20 | unknown | 0.050 | gao01 | CSV/T002_008.csv | 0.7742 | 0.5558 | -0.0201 | ambiguous_body_local_swing |
| 66 | T002_010_gao01_4p62_6p62 | unknown | 0.000 | gao01 | CSV/T002_010.csv | 0.6683 | -0.0475 | 0.0025 | weak_motion_evidence |
| 67 | T002_010_gao01_5p97_7p97 | unknown | 0.000 | gao01 | CSV/T002_010.csv | 0.7562 | 0.0910 | 0.0012 | weak_motion_evidence |
| 68 | T002_010_gao01_7p36_9p36 | unknown | 0.000 | gao01 | CSV/T002_010.csv | 0.4777 | -0.0897 | -0.0145 | weak_motion_evidence |
| 69 | T002_010_liang01_11p35_13p35 | unknown | 0.250 | liang01 | CSV/T002_010.csv | 0.0032 | -0.0713 | -0.0494 | ambiguous_body_local_swing |
| 75 | T002_010_liang01_8p04_10p04 | unknown | 0.000 | liang01 | CSV/T002_010.csv | 0.4324 | -0.0506 | 0.0028 | weak_motion_evidence |
| 77 | T002_012_gao01_1p33_3p33 | unknown | 0.000 | gao01 | CSV/T002_012.csv | 0.5227 | 0.0794 | 0.0128 | weak_motion_evidence |
| 94 | T002_013_gao01_5p70_7p70 | unknown | 0.000 | gao01 | CSV/T002_013.csv | 0.5112 | -0.0770 | 0.0065 | weak_motion_evidence |
| 95 | T002_013_gao01_7p33_9p33 | unknown | 0.000 | gao01 | CSV/T002_013.csv | 0.3103 | -0.0269 | -0.0125 | weak_motion_evidence |
| 96 | T002_013_gao01_8p77_10p77 | unknown | 0.000 | gao01 | CSV/T002_013.csv | 0.5658 | -0.1335 | -0.0144 | weak_motion_evidence |
| 100 | T002_013_liang01_14p44_16p44 | unknown | 0.350 | liang01 | CSV/T002_013.csv | 0.6582 | 0.2212 | -0.0267 | ambiguous_body_local_swing |
| 104 | T002_013_liang01_8p03_10p03 | unknown | 0.000 | liang01 | CSV/T002_013.csv | 0.5985 | 0.1787 | 0.0073 | weak_motion_evidence |
| 109 | T002_015_gao01_1p13_3p13 | unknown | 0.000 | gao01 | CSV/T002_015.csv | 0.5190 | -0.1034 | -0.0134 | weak_motion_evidence |
| 114 | T002_015_liang01_10p98_12p98 | unknown | 0.000 | liang01 | CSV/T002_015.csv | 0.7239 | 0.1122 | 0.0209 | weak_motion_evidence |
| 117 | T002_015_liang01_1p83_3p83 | unknown | 0.000 | liang01 | CSV/T002_015.csv | 0.5758 | -0.2337 | -0.0076 | weak_motion_evidence |
| 118 | T002_015_liang01_28p48_30p48 | unknown | 0.250 | liang01 | CSV/T002_015.csv | 0.6758 | 0.4058 | 0.0067 | ambiguous_body_local_swing |
| 123 | T002_019_gao01_14p14_16p14 | unknown | 0.000 | gao01 | CSV/T002_019.csv | 0.7219 | 0.2783 | -0.0140 | weak_motion_evidence |
| 124 | T002_019_gao01_2p82_4p82 | unknown | 0.000 | gao01 | CSV/T002_019.csv | 0.7236 | 0.0809 | -0.0036 | weak_motion_evidence |
| 127 | T002_019_gao01_7p64_9p64 | unknown | 0.000 | gao01 | CSV/T002_019.csv | 0.1255 | 0.2636 | -0.0081 | weak_motion_evidence |
| 129 | T002_019_liang01_10p13_12p13 | unknown | 0.000 | liang01 | CSV/T002_019.csv | 0.5024 | -0.1019 | -0.0156 | weak_motion_evidence |
| 147 | T002_021_liang01_15p51_17p51 | unknown | 0.250 | liang01 | CSV/T002_021.csv | 0.5838 | 0.5725 | 0.0111 | ambiguous_body_local_swing |
| 151 | T002_021_liang01_7p39_9p39 | unknown | 0.250 | liang01 | CSV/T002_021.csv | 0.7126 | 0.3734 | 0.0035 | ambiguous_body_local_swing |
| 152 | T002_021_liang01_8p85_10p85 | unknown | 0.250 | liang01 | CSV/T002_021.csv | 0.5948 | 0.3953 | -0.0113 | ambiguous_body_local_swing |
| 180 | T002_023_liang01_1p80_3p80 | unknown | 0.000 | liang01 | CSV/T002_023.csv | 0.0126 | 0.2191 | 0.0070 | weak_motion_evidence |
| 181 | T002_023_liang01_25p94_27p94 | unknown | 0.000 | liang01 | CSV/T002_023.csv | -0.0947 | 0.2226 | -0.0056 | weak_motion_evidence |
| 186 | T002_024_gao01_11p96_13p96 | unknown | 0.000 | gao01 | CSV/T002_024.csv | 0.3640 | 0.0720 | 0.0117 | weak_motion_evidence |
| 198 | T002_024_liang01_3p72_5p72 | unknown | 0.000 | liang01 | CSV/T002_024.csv | 0.1591 | -0.2705 | 0.0042 | weak_motion_evidence |
| 204 | T002_027_gao01_1p61_3p61 | unknown | 0.000 | gao01 | CSV/T002_027.csv | 0.5078 | 0.1189 | 0.0227 | weak_motion_evidence |
| 212 | T002_027_liang01_2p37_4p37 | unknown | 0.000 | liang01 | CSV/T002_027.csv | 0.0401 | 0.2180 | -0.0017 | weak_motion_evidence |
| 213 | T002_027_liang01_3p74_5p74 | unknown | 0.000 | liang01 | CSV/T002_027.csv | 0.0143 | 0.2532 | 0.0059 | weak_motion_evidence |
| 226 | T002_028_liang01_11p06_13p06 | unknown | 0.000 | liang01 | CSV/T002_028.csv | 0.0269 | 0.1619 | 0.0001 | weak_motion_evidence |
| 230 | T002_028_liang01_4p56_6p56 | unknown | 0.000 | liang01 | CSV/T002_028.csv | -0.0059 | 0.2352 | 0.0050 | weak_motion_evidence |
| 231 | T002_028_liang01_8p49_10p49 | unknown | 0.000 | liang01 | CSV/T002_028.csv | -0.0231 | -0.1055 | -0.0174 | weak_motion_evidence |
| 238 | T03_002_gao01_5p38_7p38 | unknown | 0.000 | gao01 | CSV/T03_002.csv | 0.3621 | 0.1877 | 0.0107 | weak_motion_evidence |
| 250 | T03_003_gao01_1p49_3p49 | unknown | 0.000 | gao01 | CSV/T03_003.csv | 0.7990 | 0.0983 | 0.0078 | weak_motion_evidence |
| 255 | T03_003_gao01_8p39_10p39 | unknown | 0.000 | gao01 | CSV/T03_003.csv | 0.6035 | -0.0078 | 0.0123 | weak_motion_evidence |
| 265 | T03_004_gao01_1p72_3p72 | unknown | 0.000 | gao01 | CSV/T03_004.csv | 0.3168 | -0.1333 | -0.0014 | weak_motion_evidence |
| 267 | T03_004_gao01_4p46_6p46 | unknown | 0.000 | gao01 | CSV/T03_004.csv | 0.8579 | -0.1302 | 0.0012 | weak_motion_evidence |
| 274 | T03_005_gao01_10p76_12p76 | unknown | 0.000 | gao01 | CSV/T03_005.csv | 0.3858 | -0.1129 | -0.0180 | weak_motion_evidence |
| 278 | T03_005_gao01_5p36_7p36 | unknown | 0.000 | gao01 | CSV/T03_005.csv | 0.0725 | -0.0549 | -0.0189 | weak_motion_evidence |
| 291 | T03_007_gao01_2p25_4p25 | unknown | 0.000 | gao01 | CSV/T03_007.csv | 0.7468 | -0.0929 | -0.0064 | weak_motion_evidence |
| 300 | T03_008_gao01_1p36_3p36 | unknown | 0.000 | gao01 | CSV/T03_008.csv | 0.4894 | -0.1090 | 0.0069 | weak_motion_evidence |
| 320 | T03_011_liang01_0p87_2p87 | unknown | 0.000 | liang01 | CSV/T03_011.csv | 0.6058 | 0.0333 | 0.0038 | weak_motion_evidence |
| 332 | T03_012_gao01_9p38_11p38 | unknown | 0.000 | gao01 | CSV/T03_012.csv | 0.7593 | 0.0006 | -0.0224 | weak_motion_evidence |
| 333 | T03_012_liang01_0p84_2p84 | unknown | 0.000 | liang01 | CSV/T03_012.csv | 0.4968 | -0.1413 | -0.0166 | weak_motion_evidence |
| 359 | T03_014_gao01_18p36_20p36 | unknown | 0.350 | gao01 | CSV/T03_014.csv | 0.6449 | 0.2784 | -0.0252 | ambiguous_body_local_swing |
| 360 | T03_014_gao01_21p00_23p00 | unknown | 0.250 | gao01 | CSV/T03_014.csv | 0.5876 | 0.0676 | 0.0427 | ambiguous_body_local_swing |
| 369 | T03_014_liang01_20p28_22p28 | unknown | 0.000 | liang01 | CSV/T03_014.csv | 0.7451 | 0.0756 | -0.0054 | weak_motion_evidence |
| 370 | T03_014_liang01_24p23_26p23 | unknown | 0.000 | liang01 | CSV/T03_014.csv | 0.6785 | -0.0917 | -0.0060 | weak_motion_evidence |
| 373 | T03_015_gao01_10p83_12p83 | unknown | 0.000 | gao01 | CSV/T03_015.csv | 0.0911 | 0.2125 | -0.0101 | weak_motion_evidence |
| 375 | T03_015_gao01_14p73_16p73 | unknown | 0.000 | gao01 | CSV/T03_015.csv | 0.5421 | -0.2250 | -0.0078 | weak_motion_evidence |
| 378 | T03_015_gao01_1p17_3p17 | unknown | 0.000 | gao01 | CSV/T03_015.csv | 0.5550 | -0.0537 | -0.0120 | weak_motion_evidence |
| 379 | T03_015_gao01_4p10_6p10 | unknown | 0.000 | gao01 | CSV/T03_015.csv | 0.5344 | -0.2241 | 0.0161 | weak_motion_evidence |
| 384 | T03_015_liang01_15p34_17p34 | unknown | 0.000 | liang01 | CSV/T03_015.csv | 0.6118 | -0.1953 | -0.0098 | weak_motion_evidence |
| 391 | T03_016_gao01_3p97_5p97 | unknown | 0.000 | gao01 | CSV/T03_016.csv | 0.3299 | -0.0977 | -0.0014 | weak_motion_evidence |
| 392 | T03_016_gao01_5p34_7p34 | unknown | 0.000 | gao01 | CSV/T03_016.csv | 0.5829 | -0.1346 | -0.0186 | weak_motion_evidence |
| 396 | T03_016_liang01_1p99_3p99 | unknown | 0.000 | liang01 | CSV/T03_016.csv | 0.6273 | 0.0529 | 0.0101 | weak_motion_evidence |
| 403 | T03_019_gao01_12p11_14p11 | unknown | 0.000 | gao01 | CSV/T03_019.csv | 0.8079 | 0.1992 | 0.0109 | weak_motion_evidence |
| 405 | T03_019_gao01_3p34_5p34 | unknown | 0.000 | gao01 | CSV/T03_019.csv | 0.5725 | 0.1436 | 0.0059 | weak_motion_evidence |
| 408 | T03_019_gao01_9p52_11p52 | unknown | 0.000 | gao01 | CSV/T03_019.csv | 0.4602 | -0.1277 | -0.0140 | weak_motion_evidence |
| 421 | T03_022_gao01_26p38_28p38 | unknown | 0.000 | gao01 | CSV/T03_022.csv | 0.7001 | -0.1499 | -0.0094 | weak_motion_evidence |
| 423 | T03_022_gao01_7p38_9p38 | unknown | 0.250 | gao01 | CSV/T03_022.csv | 0.5093 | 0.2050 | 0.0186 | ambiguous_body_local_swing |
| 425 | T03_022_liang01_12p73_14p73 | unknown | 0.000 | liang01 | CSV/T03_022.csv | 0.6019 | -0.2559 | -0.0047 | weak_motion_evidence |
| 426 | T03_022_liang01_1p91_3p91 | unknown | 0.000 | liang01 | CSV/T03_022.csv | 0.5887 | -0.1479 | -0.0051 | weak_motion_evidence |
| 437 | T03_025_gao01_22p19_24p19 | unknown | 0.000 | gao01 | CSV/T03_025.csv | 0.6168 | 0.2452 | -0.0012 | weak_motion_evidence |
| 438 | T03_025_gao01_7p66_9p66 | unknown | 0.000 | gao01 | CSV/T03_025.csv | 0.3759 | -0.0116 | 0.0116 | weak_motion_evidence |
| 447 | T03_025_liang01_9p62_11p62 | unknown | 0.350 | liang01 | CSV/T03_025.csv | 0.6421 | -0.4554 | 0.0240 | ambiguous_body_local_swing |
| 449 | T03_027_gao01_1p30_3p30 | unknown | 0.250 | gao01 | CSV/T03_027.csv | 0.4370 | -0.0244 | 0.0306 | ambiguous_body_local_swing |
| 455 | T03_027_liang01_1p95_3p95 | unknown | 0.000 | liang01 | CSV/T03_027.csv | 0.5553 | 0.0514 | 0.0218 | weak_motion_evidence |
| 457 | T03_027_liang01_5p14_7p14 | unknown | 0.250 | liang01 | CSV/T03_027.csv | 0.0360 | 0.0693 | -0.0370 | ambiguous_body_local_swing |
| 464 | T03_028_liang01_2p02_4p02 | unknown | 0.200 | liang01 | CSV/T03_028.csv | 0.2924 | 0.6608 | -0.0260 | ambiguous_body_local_swing |
| 465 | T03_028_liang01_3p43_5p43 | unknown | 0.000 | liang01 | CSV/T03_028.csv | -0.0197 | 0.1904 | -0.0028 | weak_motion_evidence |
| 468 | T03_030_gao01_3p81_5p81 | unknown | 0.000 | gao01 | CSV/T03_030.csv | 0.2841 | 0.1332 | 0.0143 | weak_motion_evidence |
| 481 | T03_031_gao01_1p33_3p33 | unknown | 0.000 | gao01 | CSV/T03_031.csv | 0.3530 | -0.0187 | 0.0226 | weak_motion_evidence |
| 523 | T03_037_gao01_3p45_5p45 | unknown | 0.250 | gao01 | CSV/T03_037.csv | 0.3629 | 0.1987 | 0.0208 | ambiguous_body_local_swing |
| 529 | T03_037_liang01_2p76_4p76 | unknown | 0.250 | liang01 | CSV/T03_037.csv | -0.0585 | -0.0837 | -0.0307 | ambiguous_body_local_swing |
| 541 | T03_038_liang01_4p53_6p53 | unknown | 0.000 | liang01 | CSV/T03_038.csv | 0.0060 | 0.1711 | -0.0124 | weak_motion_evidence |
| 542 | T03_038_liang01_5p83_7p83 | unknown | 0.250 | liang01 | CSV/T03_038.csv | -0.0791 | 0.0175 | -0.0379 | ambiguous_body_local_swing |
| 549 | T03_039_liang01_4p83_6p83 | unknown | 0.250 | liang01 | CSV/T03_039.csv | -0.0674 | -0.1077 | -0.0352 | ambiguous_body_local_swing |
| 566 | T03_045_gao01_1p32_3p32 | unknown | 0.000 | gao01 | CSV/T03_045.csv | 0.6041 | 0.0569 | 0.0005 | weak_motion_evidence |
| 568 | T03_045_gao01_4p83_6p83 | unknown | 0.000 | gao01 | CSV/T03_045.csv | 0.5913 | -0.0677 | -0.0191 | weak_motion_evidence |
| 571 | T03_045_liang01_1p92_3p92 | unknown | 0.250 | liang01 | CSV/T03_045.csv | 0.5425 | 0.2759 | 0.0174 | ambiguous_body_local_swing |
| 572 | T03_045_liang01_3p94_5p94 | unknown | 0.250 | liang01 | CSV/T03_045.csv | 0.3524 | 0.2911 | 0.0128 | ambiguous_body_local_swing |
| 589 | T03_050_liang01_6p37_8p37 | unknown | 0.000 | liang01 | CSV/T03_050.csv | 0.6537 | 0.1996 | -0.0026 | weak_motion_evidence |
| 611 | T03_053_gao01_8p80_10p80 | unknown | 0.000 | gao01 | CSV/T03_053.csv | 0.6940 | -0.1208 | 0.0003 | weak_motion_evidence |
| 617 | T03_053_liang01_5p11_7p11 | unknown | 0.000 | liang01 | CSV/T03_053.csv | 0.4193 | -0.0978 | -0.0085 | weak_motion_evidence |
| 639 | T03_065_liang01_2p18_4p18 | unknown | 0.000 | liang01 | CSV/T03_065.csv | 0.6381 | -0.1250 | -0.0081 | weak_motion_evidence |
| 649 | T03_067_liang01_0p93_2p93 | unknown | 0.000 | liang01 | CSV/T03_067.csv | 0.5778 | -0.1373 | -0.0129 | weak_motion_evidence |
| 664 | T03_068_liang01_5p29_7p29 | unknown | 0.000 | liang01 | CSV/T03_068.csv | 0.5251 | -0.2797 | -0.0085 | weak_motion_evidence |
| 668 | T03_069_gao01_1p29_3p29 | unknown | 0.250 | gao01 | CSV/T03_069.csv | 0.6465 | 0.1421 | 0.0256 | ambiguous_body_local_swing |
| 680 | T03_071_gao01_3p87_5p87 | unknown | 0.000 | gao01 | CSV/T03_071.csv | 0.0300 | 0.2696 | 0.0010 | weak_motion_evidence |
| 681 | T03_071_liang01_0p68_2p68 | unknown | 0.000 | liang01 | CSV/T03_071.csv | 0.5422 | 0.0154 | 0.0089 | weak_motion_evidence |
| 682 | T03_071_liang01_1p96_3p96 | unknown | 0.250 | liang01 | CSV/T03_071.csv | 0.6924 | 0.1553 | 0.0121 | ambiguous_body_local_swing |
| 683 | T03_071_liang01_3p22_5p22 | unknown | 0.000 | liang01 | CSV/T03_071.csv | 0.6809 | 0.1947 | 0.0028 | weak_motion_evidence |
| 684 | T03_072_gao01_1p11_3p11 | unknown | 0.000 | gao01 | CSV/T03_072.csv | 0.5809 | -0.2151 | -0.0115 | weak_motion_evidence |
| 689 | T03_072_liang01_1p71_3p71 | unknown | 0.200 | liang01 | CSV/T03_072.csv | 0.2102 | 0.5440 | -0.0189 | ambiguous_body_local_swing |
| 691 | T03_072_liang01_4p79_6p79 | unknown | 0.250 | liang01 | CSV/T03_072.csv | 0.4474 | 0.9629 | -0.0165 | ambiguous_body_local_swing |
| 704 | T03_075_liang01_17p37_19p37 | unknown | 0.000 | liang01 | CSV/T03_075.csv | 0.5315 | 0.0447 | -0.0005 | weak_motion_evidence |
| 706 | T03_075_liang01_5p22_7p22 | unknown | 0.000 | liang01 | CSV/T03_075.csv | 0.0576 | 0.2373 | -0.0083 | weak_motion_evidence |
| 710 | T03_076_gao01_11p41_13p41 | unknown | 0.350 | gao01 | CSV/T03_076.csv | 0.4127 | 0.2138 | -0.0518 | ambiguous_body_local_swing |
| 730 | T03_077_liang01_2p36_4p36 | unknown | 0.000 | liang01 | CSV/T03_077.csv | 0.0406 | 0.0514 | -0.0192 | weak_motion_evidence |
| 731 | T03_077_liang01_3p65_5p65 | unknown | 0.000 | liang01 | CSV/T03_077.csv | 0.0764 | -0.0191 | -0.0224 | weak_motion_evidence |
| 733 | T03_077_liang01_6p28_8p28 | unknown | 0.000 | liang01 | CSV/T03_077.csv | 0.5668 | -0.0636 | 0.0026 | weak_motion_evidence |
| 739 | T03_078_gao01_7p28_9p28 | unknown | 0.000 | gao01 | CSV/T03_078.csv | 0.6331 | -0.1381 | -0.0235 | weak_motion_evidence |
| 744 | T03_078_liang01_5p21_7p21 | unknown | 0.000 | liang01 | CSV/T03_078.csv | 0.0461 | -0.0508 | -0.0000 | weak_motion_evidence |
| 768 | T03_080_liang01_4p62_6p62 | unknown | 0.000 | liang01 | CSV/T03_080.csv | 0.0025 | 0.2438 | -0.0065 | weak_motion_evidence |
| 774 | T03_083_gao01_5p53_7p53 | unknown | 0.250 | gao01 | CSV/T03_083.csv | 0.4919 | 0.3080 | -0.0093 | ambiguous_body_local_swing |
| 777 | T03_083_liang01_0p84_2p84 | unknown | 0.000 | liang01 | CSV/T03_083.csv | 0.5084 | 0.1134 | 0.0202 | weak_motion_evidence |
| 781 | T03_083_liang01_2p37_4p37 | unknown | 0.000 | liang01 | CSV/T03_083.csv | 0.5202 | -0.1720 | -0.0040 | weak_motion_evidence |
| 792 | T04_001_gao01_3p45_5p45 | unknown | 0.250 | gao01 | CSV/T04_001.csv | 0.5580 | 0.3068 | 0.0033 | ambiguous_body_local_swing |
| 793 | T04_001_gao01_5p68_7p68 | unknown | 0.000 | gao01 | CSV/T04_001.csv | 0.5984 | 0.2110 | -0.0097 | weak_motion_evidence |
| 795 | T04_001_liang01_0p31_2p31 | unknown | 0.000 | liang01 | CSV/T04_001.csv | nan | nan | nan | nonfinite_stroke_features |
| 796 | T04_001_liang01_11p39_13p39 | unknown | 0.000 | liang01 | CSV/T04_001.csv | nan | nan | nan | nonfinite_stroke_features |
| 797 | T04_001_liang01_1p54_3p54 | unknown | 0.000 | liang01 | CSV/T04_001.csv | nan | nan | nan | nonfinite_stroke_features |
| 798 | T04_001_liang01_2p86_4p86 | unknown | 0.000 | liang01 | CSV/T04_001.csv | nan | nan | nan | nonfinite_stroke_features |
| 799 | T04_001_liang01_5p13_7p13 | unknown | 0.000 | liang01 | CSV/T04_001.csv | nan | nan | nan | nonfinite_stroke_features |
| 800 | T04_001_liang01_7p24_9p24 | unknown | 0.000 | liang01 | CSV/T04_001.csv | nan | nan | nan | nonfinite_stroke_features |
| 801 | T04_001_liang01_9p35_11p35 | unknown | 0.000 | liang01 | CSV/T04_001.csv | nan | nan | nan | nonfinite_stroke_features |
| 806 | T04_005_liang01_0p77_2p77 | unknown | 0.000 | liang01 | CSV/T04_005.csv | nan | nan | nan | nonfinite_stroke_features |
| 807 | T04_005_liang01_3p14_5p14 | unknown | 0.000 | liang01 | CSV/T04_005.csv | nan | nan | nan | nonfinite_stroke_features |
| 808 | T04_005_liang01_5p38_7p38 | unknown | 0.000 | liang01 | CSV/T04_005.csv | nan | nan | nan | nonfinite_stroke_features |
| 809 | T04_005_liang01_8p72_10p72 | unknown | 0.000 | liang01 | CSV/T04_005.csv | nan | nan | nan | nonfinite_stroke_features |
| 812 | T04_006_liang01_1p54_3p54 | unknown | 0.000 | liang01 | CSV/T04_006.csv | nan | nan | nan | nonfinite_stroke_features |
| 813 | T04_006_liang01_3p71_5p71 | unknown | 0.000 | liang01 | CSV/T04_006.csv | nan | nan | nan | nonfinite_stroke_features |
| 814 | T04_007_gao01_4p56_6p56 | unknown | 0.000 | gao01 | CSV/T04_007.csv | 0.5188 | -0.0655 | -0.0032 | weak_motion_evidence |
| 816 | T04_007_liang01_1p67_3p67 | unknown | 0.000 | liang01 | CSV/T04_007.csv | nan | nan | nan | nonfinite_stroke_features |
| 817 | T04_007_liang01_3p99_5p99 | unknown | 0.000 | liang01 | CSV/T04_007.csv | nan | nan | nan | nonfinite_stroke_features |
| 818 | T04_007_liang01_6p28_8p28 | unknown | 0.000 | liang01 | CSV/T04_007.csv | nan | nan | nan | nonfinite_stroke_features |
| 819 | T04_007_liang01_7p51_9p51 | unknown | 0.000 | liang01 | CSV/T04_007.csv | nan | nan | nan | nonfinite_stroke_features |
| 825 | T04_009_gao01_32p99_34p99 | unknown | 0.000 | gao01 | CSV/T04_009.csv | 0.5311 | -0.0420 | -0.0165 | weak_motion_evidence |
| 828 | T04_009_liang01_12p51_14p51 | unknown | 0.000 | liang01 | CSV/T04_009.csv | nan | nan | nan | nonfinite_stroke_features |
| 829 | T04_009_liang01_22p15_24p15 | unknown | 0.000 | liang01 | CSV/T04_009.csv | nan | nan | nan | nonfinite_stroke_features |
| 830 | T04_009_liang01_24p60_26p60 | unknown | 0.000 | liang01 | CSV/T04_009.csv | nan | nan | nan | nonfinite_stroke_features |
| 831 | T04_009_liang01_27p11_29p11 | unknown | 0.000 | liang01 | CSV/T04_009.csv | nan | nan | nan | nonfinite_stroke_features |
| 832 | T04_009_liang01_4p09_6p09 | unknown | 0.000 | liang01 | CSV/T04_009.csv | nan | nan | nan | nonfinite_stroke_features |
| 833 | T04_009_liang01_7p67_9p67 | unknown | 0.000 | liang01 | CSV/T04_009.csv | nan | nan | nan | nonfinite_stroke_features |
| 834 | T04_009_liang01_8p94_10p94 | unknown | 0.000 | liang01 | CSV/T04_009.csv | nan | nan | nan | nonfinite_stroke_features |
| 838 | T04_010_liang01_1p71_3p71 | unknown | 0.000 | liang01 | CSV/T04_010.csv | nan | nan | nan | nonfinite_stroke_features |
| 839 | T04_010_liang01_3p07_5p07 | unknown | 0.000 | liang01 | CSV/T04_010.csv | nan | nan | nan | nonfinite_stroke_features |
| 840 | T04_010_liang01_4p30_6p30 | unknown | 0.000 | liang01 | CSV/T04_010.csv | nan | nan | nan | nonfinite_stroke_features |
| 841 | T04_010_liang01_6p70_8p70 | unknown | 0.000 | liang01 | CSV/T04_010.csv | nan | nan | nan | nonfinite_stroke_features |
| 843 | T04_014_gao01_10p90_12p90 | unknown | 0.000 | gao01 | CSV/T04_014.csv | 0.7012 | 0.2008 | 0.0031 | weak_motion_evidence |
| 844 | T04_014_gao01_2p41_4p41 | unknown | 0.000 | gao01 | CSV/T04_014.csv | 0.5425 | 0.1289 | 0.0208 | weak_motion_evidence |
| 848 | T04_014_liang01_10p12_12p12 | unknown | 0.000 | liang01 | CSV/T04_014.csv | nan | nan | nan | nonfinite_stroke_features |
| 849 | T04_014_liang01_1p79_3p79 | unknown | 0.000 | liang01 | CSV/T04_014.csv | nan | nan | nan | nonfinite_stroke_features |
| 850 | T04_014_liang01_4p14_6p14 | unknown | 0.000 | liang01 | CSV/T04_014.csv | nan | nan | nan | nonfinite_stroke_features |
| 851 | T04_014_liang01_6p46_8p46 | unknown | 0.000 | liang01 | CSV/T04_014.csv | nan | nan | nan | nonfinite_stroke_features |
| 852 | T04_014_liang01_8p90_10p90 | unknown | 0.000 | liang01 | CSV/T04_014.csv | nan | nan | nan | nonfinite_stroke_features |
| 860 | T04_018_gao01_8p07_10p07 | unknown | 0.000 | gao01 | CSV/T04_018.csv | 0.6604 | -0.1007 | -0.0033 | weak_motion_evidence |
| 861 | T04_018_liang01_0p71_2p71 | unknown | 0.000 | liang01 | CSV/T04_018.csv | nan | nan | nan | nonfinite_stroke_features |
| 862 | T04_018_liang01_10p80_12p80 | unknown | 0.000 | liang01 | CSV/T04_018.csv | nan | nan | nan | nonfinite_stroke_features |
| 863 | T04_018_liang01_13p01_15p01 | unknown | 0.000 | liang01 | CSV/T04_018.csv | nan | nan | nan | nonfinite_stroke_features |
| 864 | T04_018_liang01_15p24_17p24 | unknown | 0.000 | liang01 | CSV/T04_018.csv | nan | nan | nan | nonfinite_stroke_features |
| 865 | T04_018_liang01_16p57_18p57 | unknown | 0.000 | liang01 | CSV/T04_018.csv | nan | nan | nan | nonfinite_stroke_features |
| 866 | T04_018_liang01_2p91_4p91 | unknown | 0.000 | liang01 | CSV/T04_018.csv | nan | nan | nan | nonfinite_stroke_features |
| 867 | T04_018_liang01_5p16_7p16 | unknown | 0.000 | liang01 | CSV/T04_018.csv | nan | nan | nan | nonfinite_stroke_features |
| 868 | T04_018_liang01_8p62_10p62 | unknown | 0.000 | liang01 | CSV/T04_018.csv | nan | nan | nan | nonfinite_stroke_features |
| 872 | T04_021_gao01_14p88_16p88 | unknown | 0.000 | gao01 | CSV/T04_021.csv | 0.7410 | -0.0543 | -0.0213 | weak_motion_evidence |
| 876 | T04_021_liang01_0p32_2p32 | unknown | 0.000 | liang01 | CSV/T04_021.csv | nan | nan | nan | nonfinite_stroke_features |
| 877 | T04_021_liang01_10p76_12p76 | unknown | 0.000 | liang01 | CSV/T04_021.csv | nan | nan | nan | nonfinite_stroke_features |
| 878 | T04_021_liang01_13p03_15p03 | unknown | 0.000 | liang01 | CSV/T04_021.csv | nan | nan | nan | nonfinite_stroke_features |
| 879 | T04_021_liang01_16p63_18p63 | unknown | 0.000 | liang01 | CSV/T04_021.csv | nan | nan | nan | nonfinite_stroke_features |
| 880 | T04_021_liang01_1p53_3p53 | unknown | 0.000 | liang01 | CSV/T04_021.csv | nan | nan | nan | nonfinite_stroke_features |
| 881 | T04_021_liang01_2p81_4p81 | unknown | 0.000 | liang01 | CSV/T04_021.csv | nan | nan | nan | nonfinite_stroke_features |
| 882 | T04_021_liang01_6p11_8p11 | unknown | 0.000 | liang01 | CSV/T04_021.csv | nan | nan | nan | nonfinite_stroke_features |
| 883 | T04_021_liang01_9p48_11p48 | unknown | 0.000 | liang01 | CSV/T04_021.csv | nan | nan | nan | nonfinite_stroke_features |
| 886 | T04_023_gao01_12p48_14p48 | unknown | 0.000 | gao01 | CSV/T04_023.csv | 0.5305 | -0.1114 | -0.0059 | weak_motion_evidence |
| 887 | T04_023_gao01_1p54_3p54 | unknown | 0.000 | gao01 | CSV/T04_023.csv | 0.3762 | -0.2710 | 0.0070 | weak_motion_evidence |
| 892 | T04_023_liang01_0p89_2p89 | unknown | 0.000 | liang01 | CSV/T04_023.csv | nan | nan | nan | nonfinite_stroke_features |
| 893 | T04_023_liang01_10p41_12p41 | unknown | 0.000 | liang01 | CSV/T04_023.csv | nan | nan | nan | nonfinite_stroke_features |
| 894 | T04_023_liang01_11p74_13p74 | unknown | 0.000 | liang01 | CSV/T04_023.csv | nan | nan | nan | nonfinite_stroke_features |
| 895 | T04_023_liang01_2p14_4p14 | unknown | 0.000 | liang01 | CSV/T04_023.csv | nan | nan | nan | nonfinite_stroke_features |
| 896 | T04_023_liang01_4p50_6p50 | unknown | 0.000 | liang01 | CSV/T04_023.csv | nan | nan | nan | nonfinite_stroke_features |
| 897 | T04_023_liang01_6p87_8p87 | unknown | 0.000 | liang01 | CSV/T04_023.csv | nan | nan | nan | nonfinite_stroke_features |
| 898 | T04_023_liang01_8p12_10p12 | unknown | 0.000 | liang01 | CSV/T04_023.csv | nan | nan | nan | nonfinite_stroke_features |
| 899 | T04_024_gao01_0p92_2p92 | unknown | 0.000 | gao01 | CSV/T04_024.csv | 0.5108 | -0.0246 | 0.0105 | weak_motion_evidence |
| 900 | T04_024_gao01_3p18_5p18 | unknown | 0.000 | gao01 | CSV/T04_024.csv | 0.5197 | 0.0602 | 0.0128 | weak_motion_evidence |
| 901 | T04_024_gao01_5p41_7p41 | unknown | 0.000 | gao01 | CSV/T04_024.csv | 0.6122 | 0.0706 | 0.0096 | weak_motion_evidence |
| 902 | T04_024_liang01_1p49_3p49 | unknown | 0.000 | liang01 | CSV/T04_024.csv | nan | nan | nan | nonfinite_stroke_features |
| 903 | T04_024_liang01_3p71_5p71 | unknown | 0.000 | liang01 | CSV/T04_024.csv | nan | nan | nan | nonfinite_stroke_features |
| 904 | T04_024_liang01_5p95_7p95 | unknown | 0.000 | liang01 | CSV/T04_024.csv | nan | nan | nan | nonfinite_stroke_features |
| 905 | T04_026_gao01_10p39_12p39 | unknown | 0.050 | gao01 | CSV/T04_026.csv | 0.7882 | 0.5104 | -0.0229 | ambiguous_body_local_swing |
| 907 | T04_026_gao01_3p34_5p34 | unknown | 0.000 | gao01 | CSV/T04_026.csv | 0.6590 | -0.1251 | -0.0225 | weak_motion_evidence |
| 908 | T04_026_gao01_5p62_7p62 | unknown | 0.000 | gao01 | CSV/T04_026.csv | 0.5836 | -0.0359 | -0.0059 | weak_motion_evidence |
| 909 | T04_026_gao01_7p96_9p96 | unknown | 0.000 | gao01 | CSV/T04_026.csv | 0.6358 | -0.1610 | 0.0008 | weak_motion_evidence |
| 910 | T04_026_liang01_1p63_3p63 | unknown | 0.000 | liang01 | CSV/T04_026.csv | nan | nan | nan | nonfinite_stroke_features |
| 911 | T04_026_liang01_3p92_5p92 | unknown | 0.000 | liang01 | CSV/T04_026.csv | nan | nan | nan | nonfinite_stroke_features |
| 912 | T04_026_liang01_7p35_9p35 | unknown | 0.000 | liang01 | CSV/T04_026.csv | nan | nan | nan | nonfinite_stroke_features |
| 913 | T04_026_liang01_9p71_11p71 | unknown | 0.000 | liang01 | CSV/T04_026.csv | nan | nan | nan | nonfinite_stroke_features |
| 926 | T_006_liang01_30p08_32p08 | unknown | 0.000 | liang01 | CSV/T_006.csv | 0.0054 | 0.2339 | 0.0073 | weak_motion_evidence |
| 928 | T_006_liang01_4p29_6p29 | unknown | 0.000 | liang01 | CSV/T_006.csv | 0.0300 | 0.2509 | 0.0086 | weak_motion_evidence |
| 934 | T_009_gao01_22p36_24p36 | unknown | 0.000 | gao01 | CSV/T_009.csv | 0.5808 | -0.0883 | 0.0240 | weak_motion_evidence |

## Rule Notes

- Positive/negative labels are inferred from motion relative to `body_right_axis`, not from Motive/table axes.
- Racket id and skeleton id are used only in this report for auditing.
- Low-confidence samples are intentionally kept as `unknown` to avoid contaminating training labels.
- This relabeling can be run before table-frame calibration; table-frame calibration is still required for landing and success labels.
