# TTMD6 Tracking Motion Manifest

- source optimized manifest: `/home/bruce/桌面/HOPETableTennis/data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v1/a3_ik_locked_v1/optimized_manifest.json`
- replay-ready motions: `16`

## Stroke Counts

- `forehand`: 13
- `backhand`: 3

## Outputs

- motion npz manifest: `/home/bruce/桌面/HOPETableTennis/data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v1/a3_ik_locked_v1/tracking/optimized_motion_npz_manifest.json`
- manifest: `/home/bruce/桌面/HOPETableTennis/data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v1/a3_ik_locked_v1/tracking/tracking_motion_manifest.json`
- forehand manifest: `/home/bruce/桌面/HOPETableTennis/data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v1/a3_ik_locked_v1/tracking/tracking_motion_manifest_forehand.json`
- backhand manifest: `/home/bruce/桌面/HOPETableTennis/data/analysis/mocap_cleaning_outputs/TTMD6_pilot_retarget_v1/a3_ik_locked_v1/tracking/tracking_motion_manifest_backhand.json`

## Notes

- TTMD6 stroke labels remain class-inferred, not authoritative ground truth.
- These manifests are replay-oriented only and remain not training-approved.
- Native zero-residual calibration must be audited separately before RL use.
