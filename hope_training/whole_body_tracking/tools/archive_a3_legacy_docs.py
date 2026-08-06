#!/usr/bin/env python3
"""Archive every non-authoritative A3 document from ``docs/`` exactly once.

This is intentionally narrow: it only touches immediate children of the given
docs directory which are not the three allowed long-lived entry points.  The
archive is verified before source files are removed, and its JSON index records
the original path and SHA-256 for recovery.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from a3_strike_contract import sha256_file


KEEP = {"A3_STRIKE_MASTER.md", "README.md", "a3_t2d5_parameters.json"}


def _files(root: Path, entry: Path) -> list[Path]:
    return [entry] if entry.is_file() else sorted(path for path in entry.rglob("*") if path.is_file())


def _verify(archive: Path, index: dict) -> None:
    with tempfile.TemporaryDirectory(prefix="a3_docs_archive_verify_") as directory:
        extracted = Path(directory)
        subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(extracted)], check=True)
        for record in index["files"]:
            path = extracted / record["original_path"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"archive hash verification failed: {record['original_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    docs = args.docs.expanduser().resolve()
    archive_dir = args.archive_dir.expanduser().resolve()
    archive = archive_dir / "docs.tar.zst"
    index_path = archive_dir / "DOCS_ARCHIVE_MANIFEST.json"
    if args.verify_only:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        _verify(archive, index)
        print(f"verified {archive}")
        return
    entries = sorted((path for path in docs.iterdir() if path.name not in KEEP), key=lambda path: path.name)
    if not entries:
        print("nothing to archive")
        return
    archive_dir.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {archive}")
    records = []
    for entry in entries:
        for file in _files(docs, entry):
            records.append({"original_path": str(file.relative_to(docs)), "sha256": sha256_file(file), "size_bytes": file.stat().st_size})
    subprocess.run(["tar", "--zstd", "-C", str(docs), "-cf", str(archive), *[entry.name for entry in entries]], check=True)
    archived_names = set(subprocess.check_output(["tar", "--zstd", "-tf", str(archive)], text=True).splitlines())
    missing = [record["original_path"] for record in records if record["original_path"] not in archived_names]
    if missing:
        raise RuntimeError(f"archive verification failed; missing {missing[:5]}")
    index = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive),
        "recover_command": f"tar --zstd -xf {archive} -C {docs}",
        "reason": "Phase 1 single A3_STRIKE_MASTER documentation entry point",
        "files": records,
    }
    (archive_dir / "DOCS_ARCHIVE_MANIFEST.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _verify(archive, index)
    for entry in entries:
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    print(json.dumps({"archive": str(archive), "removed_entries": [entry.name for entry in entries]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
