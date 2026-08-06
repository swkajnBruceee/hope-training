"""Write a small manifest subset selected by episode_id.

This is for deterministic execution-contract audits; it does not alter the
source manifest.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("episode_id", nargs="+")
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    wanted = list(args.episode_id)
    by_id = {str(entry.get("episode_id")): entry for entry in source.get("motions", [])}
    missing = [episode_id for episode_id in wanted if episode_id not in by_id]
    if missing:
        raise SystemExit(f"missing episode_id(s): {', '.join(missing)}")

    output = copy.deepcopy(source)
    output["motions"] = [copy.deepcopy(by_id[episode_id]) for episode_id in wanted]
    output["dataset_status"] = "execution_contract_audit_subset_not_training_approved"
    output["subset_source_manifest"] = str(args.source)
    output["subset_episode_ids"] = wanted
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(wanted)} motion(s))")


if __name__ == "__main__":
    main()
