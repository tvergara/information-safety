"""Utilities for attack information accounting."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def compute_text_bits(text: str) -> int:
    """Return the bit-length of text under UTF-8 encoding."""
    return len(text.encode("utf-8")) * 8


def aggregate_bits_by_method(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute average bits per attack method from result rows."""
    grouped: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        method = row.get("attack_method") or row.get("experiment_name")
        bits = row.get("bits")
        if method is None or bits is None:
            continue
        if isinstance(bits, int | float):
            grouped[str(method)].append(float(bits))

    return {
        method: sum(values) / len(values)
        for method, values in grouped.items()
        if values
    }
