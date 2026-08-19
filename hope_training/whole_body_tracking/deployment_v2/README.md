# Deployment-Aligned SAC V2 contract prototype

This directory is the pure-Python public contract boundary shared by a future
Isaac environment and ROS deployment adapter. It does not contain a ROS node,
an optimizer, or a trained high-level policy.

Canonical authority is the frozen open-source deployment model/parser/runner
and the matching HOPE trajectory/solver messages. `hope_open_source_contract.py`
reads and identity-checks model_21800 ONNX metadata without requiring the
optional `onnx` package. `schema2_adapter.py` implements deterministic timing,
station/side, lifecycle, and 19-double packet primitives.

Run without Isaac:

```bash
cd /path/to/hope-model21800-isaac
python3 -m pytest hope_training/whole_body_tracking/deployment_v2/tests -q
```

The station mirror is deliberately labelled
`PROTOTYPE_REQUIRES_NATIVE_PARITY_TEST`: it mirrors accepted commands owned by
the adapter and does not claim access to the native runner's private state.

See [CONTRACT.md](CONTRACT.md) for the normative prototype boundary.
