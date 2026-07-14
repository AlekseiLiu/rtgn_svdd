"""Dependency-free loading and serialization of experiment configuration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


Config = dict[str, Any]


def resolve_config(config: Mapping[str, object]) -> Config:
    """Validate and return a detached, JSON-compatible configuration."""
    if "schema_version" not in config:
        raise ValueError("configuration must define 'schema_version'")
    if config["schema_version"] in (None, ""):
        raise ValueError("configuration 'schema_version' must not be empty")

    try:
        serialized = json.dumps(config, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"configuration must contain JSON-compatible values: {error}") from error

    resolved = json.loads(serialized)
    if not isinstance(resolved, dict):  # Defensive: Mapping should serialize as an object.
        raise ValueError("configuration root must be a JSON object")
    return resolved


def load_config(path: Path) -> Config:
    """Load and validate a JSON configuration from ``path``."""
    try:
        with path.open(encoding="utf-8") as stream:
            config = json.load(stream)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON configuration at {path}: {error}") from error

    if not isinstance(config, dict):
        raise ValueError(f"configuration root at {path} must be a JSON object")
    return resolve_config(config)


def write_config(config: Mapping[str, object], path: Path) -> None:
    """Write a validated configuration in canonical, deterministic JSON."""
    resolved = resolve_config(config)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(resolved, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
