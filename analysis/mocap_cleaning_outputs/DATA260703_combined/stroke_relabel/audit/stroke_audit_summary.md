# DATA260703 Stroke Label Audit

抽检目标：检查 v2 正反手标签是否按逐拍动作特征划分，而不是把某个人/球拍固定为正手或反手。

- total samples: 792
- label counts: `{'forehand': 419, 'backhand': 298, 'unknown': 75}`
- unknown samples: `75`
- low confidence known samples: `12`

## By Racket

| racket | forehand | backhand | unknown | total |
|---|---:|---:|---:|---:|
| TennisBats01 | 383 | 5 | 13 | 401 |
| TennisBats02 | 36 | 293 | 62 | 391 |

## By Source

| source | forehand | backhand | unknown | total |
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

## Audit Buckets

| bucket | selected | meaning |
|---|---:|---|
| unknown | 24 | 规则没有足够证据硬分，优先检查是否应保持 unknown |
| bats01_backhand | 5 | TennisBats01 中被判为反手，验证是否是真反手或异常动作 |
| low_conf_known | 12 | 已判正/反手但置信度偏低 |
| bats02_forehand | 24 | TennisBats02 中被判为正手，验证不是误把反手片段判正手 |

## Diagnostic SVGs

每张 SVG 包含 5 条曲线：人体局部横向位置、人体局部横向速度、球拍-球距离、球速、球拍速度。竖虚线是击球时刻。

### unknown

- [0210_unknown.svg](plots/0210_unknown.svg) `Table_Tennis_01_008_TennisBats02_108p65_110p65_Skeleton002`
- [0548_unknown.svg](plots/0548_unknown.svg) `Table_Tennis_01_013_TennisBats02_119p81_121p81_Skeleton002`
- [0616_unknown.svg](plots/0616_unknown.svg) `Table_Tennis_01_013_TennisBats02_91p53_93p53_Skeleton002`
- [0773_unknown.svg](plots/0773_unknown.svg) `Table_Tennis_01_014_TennisBats02_61p78_63p78_Skeleton002`
- [0166_unknown.svg](plots/0166_unknown.svg) `Table_Tennis_01_007_TennisBats02_57p70_59p70_Skeleton002`
- [0733_unknown.svg](plots/0733_unknown.svg) `Table_Tennis_01_014_TennisBats02_173p26_175p26_Skeleton002`
- [0747_unknown.svg](plots/0747_unknown.svg) `Table_Tennis_01_014_TennisBats02_205p42_207p42_Skeleton002`
- [0163_unknown.svg](plots/0163_unknown.svg) `Table_Tennis_01_007_TennisBats02_53p98_55p98_Skeleton002`
- [0230_unknown.svg](plots/0230_unknown.svg) `Table_Tennis_01_008_TennisBats02_7p43_9p43_Skeleton002`
- [0462_unknown.svg](plots/0462_unknown.svg) `Table_Tennis_01_012_TennisBats02_53p47_55p47_Skeleton002`
- [0442_unknown.svg](plots/0442_unknown.svg) `Table_Tennis_01_012_TennisBats02_127p63_129p63_Skeleton002`
- [0764_unknown.svg](plots/0764_unknown.svg) `Table_Tennis_01_014_TennisBats02_51p33_53p33_Skeleton002`
- [0776_unknown.svg](plots/0776_unknown.svg) `Table_Tennis_01_014_TennisBats02_69p29_71p29_Skeleton002`
- [0604_unknown.svg](plots/0604_unknown.svg) `Table_Tennis_01_013_TennisBats02_74p92_76p92_Skeleton002`
- [0157_unknown.svg](plots/0157_unknown.svg) `Table_Tennis_01_007_TennisBats02_38p83_40p83_Skeleton002`
- [0220_unknown.svg](plots/0220_unknown.svg) `Table_Tennis_01_008_TennisBats02_36p40_38p40_Skeleton002`
- [0056_unknown.svg](plots/0056_unknown.svg) `Table_Tennis_01_006_TennisBats01_47p65_49p65_Skeleton001`
- [0031_unknown.svg](plots/0031_unknown.svg) `Table_Tennis_01_005_TennisBats02_54p47_56p47_Skeleton002`
- [0720_unknown.svg](plots/0720_unknown.svg) `Table_Tennis_01_014_TennisBats02_136p48_138p48_Skeleton002`
- [0078_unknown.svg](plots/0078_unknown.svg) `Table_Tennis_01_006_TennisBats02_0p69_2p69_Skeleton002`
- [0591_unknown.svg](plots/0591_unknown.svg) `Table_Tennis_01_013_TennisBats02_43p62_45p62_Skeleton002`
- [0723_unknown.svg](plots/0723_unknown.svg) `Table_Tennis_01_014_TennisBats02_150p94_152p94_Skeleton002`
- [0356_unknown.svg](plots/0356_unknown.svg) `Table_Tennis_01_004_TennisBats01_21p34_23p34_Skeleton001`
- [0227_unknown.svg](plots/0227_unknown.svg) `Table_Tennis_01_008_TennisBats02_64p94_66p94_Skeleton002`

### bats01_backhand

- [0044_backhand.svg](plots/0044_backhand.svg) `Table_Tennis_01_006_TennisBats01_25p69_27p69_Skeleton001`
- [0204_backhand.svg](plots/0204_backhand.svg) `Table_Tennis_01_008_TennisBats01_84p01_86p01_Skeleton001`
- [0259_backhand.svg](plots/0259_backhand.svg) `Table_Tennis_01_009_TennisBats01_134p62_136p62_Skeleton001`
- [0267_backhand.svg](plots/0267_backhand.svg) `Table_Tennis_01_009_TennisBats01_19p53_21p53_Skeleton001`
- [0288_backhand.svg](plots/0288_backhand.svg) `Table_Tennis_01_009_TennisBats01_69p91_71p91_Skeleton001`

### low_conf_known

- [0089_backhand.svg](plots/0089_backhand.svg) `Table_Tennis_01_006_TennisBats02_36p41_38p41_Skeleton002`
- [0111_backhand.svg](plots/0111_backhand.svg) `Table_Tennis_01_006_TennisBats02_7p34_9p34_Skeleton002`
- [0167_backhand.svg](plots/0167_backhand.svg) `Table_Tennis_01_007_TennisBats02_58p91_60p91_Skeleton002`
- [0312_backhand.svg](plots/0312_backhand.svg) `Table_Tennis_01_009_TennisBats02_127p39_129p39_Skeleton002`
- [0320_backhand.svg](plots/0320_backhand.svg) `Table_Tennis_01_009_TennisBats02_13p43_15p43_Skeleton002`
- [0329_backhand.svg](plots/0329_backhand.svg) `Table_Tennis_01_009_TennisBats02_29p57_31p57_Skeleton002`
- [0459_backhand.svg](plots/0459_backhand.svg) `Table_Tennis_01_012_TennisBats02_49p80_51p80_Skeleton002`
- [0559_backhand.svg](plots/0559_backhand.svg) `Table_Tennis_01_013_TennisBats02_142p89_144p89_Skeleton002`
- [0568_backhand.svg](plots/0568_backhand.svg) `Table_Tennis_01_013_TennisBats02_157p85_159p85_Skeleton002`
- [0573_backhand.svg](plots/0573_backhand.svg) `Table_Tennis_01_013_TennisBats02_164p00_166p00_Skeleton002`
- [0606_backhand.svg](plots/0606_backhand.svg) `Table_Tennis_01_013_TennisBats02_77p43_79p43_Skeleton002`
- [0143_forehand.svg](plots/0143_forehand.svg) `Table_Tennis_01_007_TennisBats01_77p88_79p88_Skeleton001`

### bats02_forehand

- [0405_forehand.svg](plots/0405_forehand.svg) `Table_Tennis_01_004_TennisBats02_65p26_67p26_Skeleton002`
- [0159_forehand.svg](plots/0159_forehand.svg) `Table_Tennis_01_007_TennisBats02_41p59_43p59_Skeleton002`
- [0302_forehand.svg](plots/0302_forehand.svg) `Table_Tennis_01_009_TennisBats02_105p58_107p58_Skeleton002`
- [0346_forehand.svg](plots/0346_forehand.svg) `Table_Tennis_01_009_TennisBats02_82p78_84p78_Skeleton002`
- [0350_forehand.svg](plots/0350_forehand.svg) `Table_Tennis_01_009_TennisBats02_93p32_95p32_Skeleton002`
- [0386_forehand.svg](plots/0386_forehand.svg) `Table_Tennis_01_004_TennisBats02_0p60_2p60_Skeleton002`
- [0387_forehand.svg](plots/0387_forehand.svg) `Table_Tennis_01_004_TennisBats02_100p80_102p80_Skeleton002`
- [0388_forehand.svg](plots/0388_forehand.svg) `Table_Tennis_01_004_TennisBats02_10p28_12p28_Skeleton002`
- [0389_forehand.svg](plots/0389_forehand.svg) `Table_Tennis_01_004_TennisBats02_15p98_17p98_Skeleton002`
- [0391_forehand.svg](plots/0391_forehand.svg) `Table_Tennis_01_004_TennisBats02_20p50_22p50_Skeleton002`
- [0394_forehand.svg](plots/0394_forehand.svg) `Table_Tennis_01_004_TennisBats02_2p04_4p04_Skeleton002`
- [0395_forehand.svg](plots/0395_forehand.svg) `Table_Tennis_01_004_TennisBats02_35p36_37p36_Skeleton002`
- [0397_forehand.svg](plots/0397_forehand.svg) `Table_Tennis_01_004_TennisBats02_38p36_40p36_Skeleton002`
- [0398_forehand.svg](plots/0398_forehand.svg) `Table_Tennis_01_004_TennisBats02_3p39_5p39_Skeleton002`
- [0399_forehand.svg](plots/0399_forehand.svg) `Table_Tennis_01_004_TennisBats02_47p32_49p32_Skeleton002`
- [0400_forehand.svg](plots/0400_forehand.svg) `Table_Tennis_01_004_TennisBats02_48p62_50p62_Skeleton002`
- [0401_forehand.svg](plots/0401_forehand.svg) `Table_Tennis_01_004_TennisBats02_50p92_52p92_Skeleton002`
- [0402_forehand.svg](plots/0402_forehand.svg) `Table_Tennis_01_004_TennisBats02_52p15_54p15_Skeleton002`
- [0403_forehand.svg](plots/0403_forehand.svg) `Table_Tennis_01_004_TennisBats02_61p92_63p92_Skeleton002`
- [0406_forehand.svg](plots/0406_forehand.svg) `Table_Tennis_01_004_TennisBats02_67p69_69p69_Skeleton002`
- [0407_forehand.svg](plots/0407_forehand.svg) `Table_Tennis_01_004_TennisBats02_69p21_71p21_Skeleton002`
- [0408_forehand.svg](plots/0408_forehand.svg) `Table_Tennis_01_004_TennisBats02_70p49_72p49_Skeleton002`
- [0409_forehand.svg](plots/0409_forehand.svg) `Table_Tennis_01_004_TennisBats02_71p74_73p74_Skeleton002`
- [0410_forehand.svg](plots/0410_forehand.svg) `Table_Tennis_01_004_TennisBats02_72p98_74p98_Skeleton002`

## Current Read

- `TennisBats02` 内部同时存在 `forehand` 和 `backhand`，说明当前规则没有按人硬分。
- `TennisBats02` 中被判为 `forehand` 的抽检样本有大量来自 `Point 01_004`，符合“同一个人有一段时间打正手”的采集描述。
- `unknown` 大多是横向速度和横向位移都弱的片段，当前保持 unknown 更安全。
- `TennisBats01` 中少量 `backhand` 的横向速度/位移方向与反手规则一致，不应仅因为球拍 ID 而强制改回正手。
- 下一步应人工打开本目录 `plots/` 下的 SVG，重点看 `unknown` 和反主趋势样本。

## Recommendation

- 当前 v2 标签可作为第一版训练标签使用，但训练时建议排除 `unknown`。
- 若要更保守，可额外只使用 `stroke_confidence_rule_v2 >= 0.85` 的正反手样本。
- 不建议把 `TennisBats01/02` 直接映射成正/反手标签。
