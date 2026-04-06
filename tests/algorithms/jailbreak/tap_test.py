"""Tests for TAP helper primitives."""

from __future__ import annotations

from information_safety.algorithms.jailbreak.tap import prune_candidates


def test_prune_candidates_keeps_top_width() -> None:
    items = [("a", 0.1), ("b", 0.9), ("c", 0.4)]
    assert prune_candidates(items, width=2) == [("b", 0.9), ("c", 0.4)]
