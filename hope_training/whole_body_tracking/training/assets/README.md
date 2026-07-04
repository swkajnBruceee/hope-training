# Generated Assets

This package exists so `training.assets.ASSET_DIR` is importable from
a fresh clone.

Generate the local A3 Isaac asset with:

```bash
python3 scripts/prepare_a3_isaac_asset.py --force
```

The generated `agibot_a3/` directory is derived from
`agibot/URDF/A3T2.5-URDF-std-pingpang/` and is intentionally git-ignored.
