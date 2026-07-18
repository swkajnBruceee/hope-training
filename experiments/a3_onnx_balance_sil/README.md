# A3 ONNX Whole-Body Balance — SIL Branch

This isolated experiment evaluates the bundled ONNX whole-body policy without
changing the native-MOTION/right-arm strike path.  It is **SIL-only** and its
first gate uses `--probe`, which disables all body-drive command publishers.

Run from the repository root:

```bash
bash experiments/a3_onnx_balance_sil/run_stand_shadow.sh
```

Pass criteria for this gate are transport/sync continuity, 50 Hz policy timing,
and bounded ONNX output in a static-standing reference.  A passing shadow gate
does not demonstrate standing balance: the robot never receives a command.

The next gate, if approved after reviewing the report, must remain SIL-only,
start from a reset upright pose, allow exactly one command publisher, and stop
on a posture/IMU violation.  It must not run alongside native `MOTION` or the
right-arm strike executor.

`run_command_stand_sil.py` is that next, bounded SIL gate.  It hands command
ownership to ONNX for three seconds only, aborts on an IMU violation, and then
restores native `PD_STAND` followed by `MOTION`.  It refuses to run unless
`SIM_MODE=sil`.
