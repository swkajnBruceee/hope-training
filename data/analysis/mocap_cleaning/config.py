"""Configuration helpers for mocap cleaning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if text in ("null", "None", "~"):
        return None
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text.strip('"').strip("'")


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by the local dataset configs.

    The repo does not depend on PyYAML at the root, so keep this intentionally
    narrow: nested dictionaries via two-space indentation, scalar leaves, and
    simple lists are enough for the DATA260703 config.
    """
    lines = text.splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]
    i = 0

    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            i += 1
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"list item without list parent: {raw_line}")
            parent.append(_parse_scalar(stripped[2:]))
            i += 1
            continue

        key, sep, value = stripped.partition(":")
        if not sep:
            raise ValueError(f"invalid config line: {raw_line}")
        key = key.strip()
        value = value.strip()

        if value == "":
            next_is_list = False
            j = i + 1
            while j < len(lines):
                next_line = lines[j].split("#", 1)[0].rstrip()
                if not next_line:
                    j += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip(" "))
                next_stripped = next_line.strip()
                next_is_list = next_indent > indent and next_stripped.startswith("- ")
                break
            next_container: dict[str, Any] | list[Any] = [] if next_is_list else {}
            if isinstance(parent, dict):
                parent[key] = next_container
            else:
                raise ValueError(f"nested map under list is unsupported: {raw_line}")
            stack.append((indent, next_container))
        elif value == "[]":
            if not isinstance(parent, dict):
                raise ValueError(f"list under list is unsupported: {raw_line}")
            parent[key] = []
            stack.append((indent, parent[key]))
        elif value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            parsed = [] if not inner else [_parse_scalar(x) for x in inner.split(",")]
            if not isinstance(parent, dict):
                raise ValueError(f"inline list under list is unsupported: {raw_line}")
            parent[key] = parsed
        else:
            if not isinstance(parent, dict):
                raise ValueError(f"scalar under list is unsupported: {raw_line}")
            parent[key] = _parse_scalar(value)

        i += 1

    return root


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if path.suffix.lower() in (".yaml", ".yml"):
        return _parse_simple_yaml(text)
    raise ValueError(f"unsupported config format: {path}")
