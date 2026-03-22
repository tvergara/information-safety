"""Tests for AutoDAN helper primitives."""

from __future__ import annotations

from information_safety.algorithms.jailbreak.autodan import crossover, mutate


def test_mutate_replaces_one_position() -> None:
    prompt = [1, 2, 3]
    mutated = mutate(prompt, position=1, new_value=9)
    assert mutated == [1, 9, 3]


def test_crossover_splices_parent_prompts() -> None:
    a = [1, 2, 3, 4]
    b = [9, 8, 7, 6]
    child = crossover(a, b, cut=2)
    assert child == [1, 2, 7, 6]
