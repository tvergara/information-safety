"""TAP helper primitives for tree-based attack search."""

from __future__ import annotations


def prune_candidates(candidates: list[tuple[str, float]], *, width: int) -> list[tuple[str, float]]:
    """Keep the top-scoring candidates up to the configured beam width."""
    if width <= 0:
        raise ValueError("width must be positive")
    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
    return ranked[:width]
