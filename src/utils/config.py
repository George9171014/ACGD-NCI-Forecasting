"""Configuration loading and path resolution utilities."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


ConfigDict = Dict[str, Any]


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or cannot be resolved."""


def read_yaml(path: str | Path) -> ConfigDict:
    """Read a YAML file and return a dictionary."""

    yaml_path = Path(path).expanduser().resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    if not yaml_path.is_file():
        raise ConfigError(f"Configuration path is not a file: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ConfigError(f"Configuration must contain a YAML mapping: {yaml_path}")
    return data


def write_yaml(data: Mapping[str, Any], path: str | Path) -> Path:
    """Write a mapping to YAML and return the resolved path."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        _safe_dump_yaml(dict(data), file)
    return output_path


def deep_update(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> ConfigDict:
    """Recursively merge override values into base and return a new dictionary."""

    merged = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> ConfigDict:
    """Load a YAML config with optional recursive ``extends`` support.

    The ``extends`` value may be a string or a list of strings. Relative base
    config paths are resolved from the child config directory.
    """

    config_path = Path(path).expanduser().resolve()
    config = _load_config_recursive(config_path, seen=set())
    if overrides:
        config = deep_update(config, overrides)
    config.setdefault("runtime", {})
    config["runtime"]["config_path"] = str(config_path)
    config["runtime"]["project_root"] = str(find_project_root(config_path.parent))
    return config


def _load_config_recursive(path: Path, seen: set[Path]) -> ConfigDict:
    if path in seen:
        cycle = " -> ".join(str(item) for item in [*seen, path])
        raise ConfigError(f"Configuration extends cycle detected: {cycle}")

    seen.add(path)
    try:
        data = read_yaml(path)
        extends = data.pop("extends", None)
        if not extends:
            return data

        base_paths = [extends] if isinstance(extends, str) else extends
        if not isinstance(base_paths, (list, tuple)):
            raise ConfigError(f"'extends' must be a string or list in {path}")

        merged: ConfigDict = {}
        for base in base_paths:
            if not isinstance(base, str):
                raise ConfigError(f"Each 'extends' entry must be a string in {path}")
            base_path = (path.parent / base).resolve()
            merged = deep_update(merged, _load_config_recursive(base_path, seen))

        return deep_update(merged, data)
    finally:
        seen.remove(path)


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the project root by walking upward from ``start``."""

    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent

    markers = ("pyproject.toml", "requirements.txt", "configs")
    for directory in (current, *current.parents):
        if any((directory / marker).exists() for marker in markers):
            return directory
    return Path.cwd().resolve()


def get_config_value(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Return a nested value from ``config`` using a dot-delimited key."""

    current: Any = config
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def resolve_config_path(
    config: Mapping[str, Any],
    dotted_key: str,
    *,
    default: str | Path | None = None,
    create: bool = False,
) -> Path:
    """Resolve a path value from config relative to the project root."""

    value = get_config_value(config, dotted_key, default)
    if value is None:
        raise ConfigError(f"Missing path config value: {dotted_key}")

    path = Path(value).expanduser()
    if not path.is_absolute():
        root = Path(get_config_value(config, "runtime.project_root", find_project_root()))
        path = root / path

    resolved = path.resolve()
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def summarize_config(config: Mapping[str, Any]) -> str:
    """Return a compact YAML summary suitable for logging."""

    return str(_safe_dump_yaml(dict(config)))


def _safe_dump_yaml(data: Mapping[str, Any], stream: Any | None = None) -> str | None:
    """Dump YAML while tolerating older PyYAML versions."""

    try:
        return yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=False)
    except TypeError:
        return yaml.safe_dump(data, stream, allow_unicode=False)
