# A3 Retarget Subset Evaluation

- selected specs: `4`
- per label target: `2`

## Status Counts

- `failed_quality_reject`: 4

## Medians

- `racket_position_error_at_hit_m`: 0.343153810738565
- `racket_orientation_error_at_hit_deg`: 16.112285916634796
- `racket_velocity_direction_error_at_hit_deg`: 45.56153554239936
- `max_joint_velocity_radps`: 13.19635111000721
- `max_joint_acceleration_radps2`: 291.41848354606805

## Reject Reasons

- `ik_residual_rms>0.03`: 4
- `max_joint_acceleration_radps2>120.0`: 4
- `max_joint_velocity_radps>12.0`: 2
- `racket_orientation_error_at_hit_deg>15.0`: 3
- `racket_position_error_at_hit_m>0.05`: 4
- `racket_velocity_direction_error_at_hit_deg>20.0`: 2
