# v2 dual-stroke changes

- The C++ wrapper still uses the original `IkPointArmSource`; no IK math was replaced.
- READY YAMLs can declare `ready_id` and `swing_type`.
- Explicit forehand/backhand goal and READY mismatches fail fast.
- Diagnostics export requested and selected stroke labels plus READY provenance.
- NPZ metadata preserves requested/selected stroke labels and READY identity.
- Added a planner-native forehand example and a stroke-specific forehand READY template.
- Added `generate_dual_stroke_dataset.py` to generate labelled forehand/backhand datasets in separate directories.
- Dataset batch mode rejects `auto` because it can mix labels and READY-state bias.

The included forehand READY is a source-seed template, not a local PhysX-qualified state.
