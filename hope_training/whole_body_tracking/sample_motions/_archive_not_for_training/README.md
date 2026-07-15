# Archived Sample Motion Libraries

These directories are not active training inputs.

They are preserved only for reference, debugging, and comparison against older processing strategies.

Current training must not use anything in this archive. The native-strike task
requires an explicit `motion_manifest=...` argument; the approved manifest must
be listed in `../README_DATASETS.md`.

The 2026-07-14 archive contains superseded top-level manifests moved from:

```text
20260714_superseded_manifests/
```

Key reasons for archive status:

- old whole-body / stale reset-distribution experiments;
- old robot-posture-only gates without wrist / forearm naturalness;
- visual failures such as forehand wrist folding;
- diagnostic comfort-zone scans that are not final training sets.

Retarget source outputs are still preserved under:

```text
data/analysis/mocap_cleaning_outputs/DATA260708_post1p0/
```
