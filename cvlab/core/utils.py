"""Common utility functions used across CVLab modules."""

from __future__ import annotations

import json
from typing import Any


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionary to dot-separated key-value pairs.

    Handles dict, list, and primitive values.

    Args:
        d: Input nested dictionary.
        prefix: Key prefix for recursion (internal use).

    Returns:
        Flattened dictionary with dot-notation keys.
        Example: {"a": {"b": 1, "c": [2, 3]}, "d": 4}
        → {"a.b": 1, "a.c": "[2, 3]", "d": 4}
    """
    result: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(flatten_dict(v, key))
        elif isinstance(v, (list, tuple)):
            result[key] = json.dumps(v)
        else:
            result[key] = v
    return result


def ensure_utf8_open(path: str, mode: str = "r"):
    """Open a file with UTF-8 encoding by default.

    Args:
        path: File path.
        mode: File open mode (default: "r").

    Returns:
        File handle with UTF-8 encoding.
    """
    return open(path, mode, encoding="utf-8")
