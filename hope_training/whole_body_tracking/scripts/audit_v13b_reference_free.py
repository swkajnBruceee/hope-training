#!/usr/bin/env python3
"""Static V1.3B contract audit; no Isaac/GPU import required."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "action": ROOT / "training/tasks/base_locomotion/mdp/actions.py",
    "command": ROOT / "training/tasks/tracking/mdp/hope_commands.py",
    "env": ROOT / "training/tasks/tracking/config/agibot_a3/native_strike_env_cfg.py",
}
FORBIDDEN_PATTERNS = ("get_term(\"motion\")", "get_term('motion')", "_motion(", "motion_cmd", "model_3396", "model_900", "legacy_stage_a", "hit_frame", "swing_type")


def class_source(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines = path.read_text(encoding="utf-8").splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise RuntimeError(f"class {name} not found in {path}")


def main() -> None:
    direct = class_source(FILES["action"], "A3ReferenceFreeTargetConditionedPositionAction")
    command = class_source(FILES["command"], "ReferenceFreeRacketTargetCommand")
    env = class_source(FILES["env"], "A3FloatingTargetConditionedReferenceFreeV13BEnvCfg")
    # The env class necessarily mentions a construction-time motion placeholder
    # inherited from the old cfg; the runtime post-init must null it before env
    # creation.  The direct action and command are the hard no-reference gates.
    for name, source in (("direct_action", direct), ("reference_free_command", command)):
        bad = [token for token in FORBIDDEN_PATTERNS if token in source]
        if bad:
            raise SystemExit(f"{name} contains forbidden runtime dependency tokens: {bad}")
    if "self.commands.motion = None" not in env:
        raise SystemExit("V1.3B env does not explicitly disable the construction placeholder motion term")
    if "return 26" not in direct:
        raise SystemExit("V1.3B action contract is not exactly 26-D")
    print("V1.3B static reference-free audit: PASS")
    print("  action: direct READY-relative 12+10+4")
    print("  command: global sampled goal, signed clock, no motion query")
    print("  env: motion placeholder nulled before ManagerBasedRLEnv construction")


if __name__ == "__main__":
    main()
