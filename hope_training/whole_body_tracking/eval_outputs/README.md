# Evaluation Outputs

Active eval logs in this directory are current native-strike gate evidence.

Current active logs:

```text
forehand_combined_gate_v1_accepted_native_zero_action_20260714.log
backhand_current_pool_v1_write_native_20260714.log
backhand_current_pool_v1_native_gate_20260714.log
backhand_current_pool_v2_write_native_20260714.log
backhand_current_pool_v2_native_gate_20260714.log
balanced_k8_current_v1_native_gate_20260714.log
balanced_k8_current_v1_res015_smoke_300_20260714_policy_model299.log
```

`manual_accepted_v3_combined_gate_20260714.log` is retained only as a diagnostic
log. It used stale/non-current target semantics and must not be used to promote
training data.

Archived logs are under:

```text
_archive_not_for_training/20260714_superseded_eval_logs/
```

Do not use archived logs to promote training data unless the data is rerun
through the current combined gate:

```text
hit task + robot posture / arm margin + wrist / forearm naturalness + visual review
```
