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


def slugify(text: str, max_len: int = 20) -> str:
    """Convert text to a URL-safe slug for experiment ID prefixes.

    Rules:
    - lowercase
    - alphanumeric and hyphens only
    - collapse multiple hyphens
    - strip leading/trailing hyphens
    - truncated to max_len

    Examples:
        "My Cool Experiment!" → "my-cool-experiment"
        "ResNet18 + CIFAR-10" → "resnet18-cifar-10"
        "test/lr=0.001" → "testlr0001"
    """
    import re
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9\-]", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug[:max_len].rstrip("-")
