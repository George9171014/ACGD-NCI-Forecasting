"""Small filesystem and serialization helpers used across the project."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return its resolved path."""

    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> Any:
    """Read JSON content from disk."""

    json_path = Path(path).expanduser().resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Write JSON content and return the resolved output path."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, sort_keys=False)
        file.write("\n")
    return output_path


def read_text(path: str | Path) -> str:
    """Read UTF-8 text from disk."""

    text_path = Path(path).expanduser().resolve()
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")
    return text_path.read_text(encoding="utf-8")


def write_text(content: str, path: str | Path) -> Path:
    """Write UTF-8 text and return the resolved output path."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def utc_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def timestamped_path(directory: str | Path, stem: str, suffix: str) -> Path:
    """Build a timestamped output path inside ``directory``."""

    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return ensure_dir(directory) / f"{stem}_{utc_timestamp()}{suffix}"


def to_jsonable_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Convert common path-like values in a mapping to JSON-serializable strings."""

    jsonable: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, Path):
            jsonable[key] = str(value)
        elif isinstance(value, Mapping):
            jsonable[key] = to_jsonable_mapping(value)
        else:
            jsonable[key] = value
    return jsonable
