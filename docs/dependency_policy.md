# Dependency And Artifact Policy

The repository should make the project reproducible without turning Git into a
runtime artifact store.

## Keep In Git

Commit files that define or document the environment:

```text
Dockerfile*
pyproject.toml
requirements*.txt
environment*.yml
package.xml
CMakeLists.txt
launch/*.py
config/*.yaml
setup_*.sh
*.example
README.md
sha256 checksum files
small reference assets required by the starter
```

## Keep Out Of Git

Do not commit generated or machine-specific artifacts:

```text
build/
install/
log/
.venv/
external_repos/
vendor_assets/
*.bag
*.pt
*.pth
*.ckpt
*.onnx
*.rknn
*.engine
*.trt
*.npz
large *.csv files outside approved reference asset folders
```

For large files that must be shared, use one of:

```text
Git LFS
GitHub Releases
object storage
internal NAS / artifact registry
```

Track the version, download location, and checksum in Git.

## Vendored Third-Party Code

The current repository includes Agibot-provided reference materials under
`agibot/`, including deployment examples and SDK/runtime references. Treat these
as a deliberate vendor bundle:

- Do not mix local edits into vendored code unless the change is intentional.
- Prefer wrapper code, patches, or clearly named integration files for local
  project behavior.
- If replacing a vendor bundle, record the source, version, license, and checksum
  in `THIRD_PARTY_NOTICES.md` or the nearest `README.md`.

## Runtime Archives And Sysroots

For sysroots, SDK archives, model engines, or runtime packages, prefer this
layout:

```text
thirdparty/<name>/README.md
thirdparty/<name>/<artifact>.sha256
scripts/fetch_<name>.sh
```

The full archive should live outside Git unless it is small and required for an
offline starter package.
