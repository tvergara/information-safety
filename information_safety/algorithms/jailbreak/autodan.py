"""AutoDAN helper primitives for evolutionary attack search."""

from __future__ import annotations


def mutate(prompt_token_ids: list[int], *, position: int, new_value: int) -> list[int]:
    """Return a new prompt with one mutated token position."""
    if position < 0 or position >= len(prompt_token_ids):
        raise IndexError("position out of range")
    out = list(prompt_token_ids)
    out[position] = new_value
    return out


def crossover(parent_a: list[int], parent_b: list[int], *, cut: int) -> list[int]:
    """Single-point crossover between two parent token sequences."""
    if len(parent_a) != len(parent_b):
        raise ValueError("parent lengths must match")
    if cut < 0 or cut > len(parent_a):
        raise IndexError("cut out of range")
    return [*parent_a[:cut], *parent_b[cut:]]
