# Frozen contract binding

This package imports, rather than duplicates, the tagged `deployment_v2` contract. Observation values are raw physical values (`OBS_NORMALIZATION=NONE_IN_CONTRACT_V1`). Predictor physical hit plane is the HOPE solver `x_hit=0.0`; model station-relative reach X is separately metadata-defined as 0.58. The Isaac/world mapping is an open-source software-frame identity assumption, not hardware calibration.

Position is `HOPE_OPEN_SOURCE_SOLVER_POSITION` (predicted ball centre). No 17 mm proxy, radius, paddle-thickness, or TCP correction is applied. Simulator future truth is `PRIVILEGED_METRIC_ONLY`.
