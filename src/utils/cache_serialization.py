"""Helpers for normalizing values before cache storage or hashing."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel


def normalize_for_cache(value: Any) -> Any:
    """Produce a JSON-serializable structure for caching."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)

    if isinstance(value, Mapping):
        return {
            str(key): normalize_for_cache(sub_value)
            for key, sub_value in sorted(value.items(), key=lambda item: str(item[0]))
        }

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [normalize_for_cache(item) for item in value]

    return str(value)


def dumps_for_cache(value: Any) -> str:
    """Serialize value deterministically for cache keys and sizing."""

    normalized = normalize_for_cache(value)
    try:
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return json.dumps(str(normalized), sort_keys=True, separators=(",", ":"))
