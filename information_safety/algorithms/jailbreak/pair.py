"""PAIR helper primitives for iterative jailbreak prompt refinement."""

from __future__ import annotations

from collections.abc import Callable, Sequence


def run_pair_refinement(
    initial_prompt: str,
    feedbacks: Sequence[str],
    refine_fn: Callable[[str, str, int], str],
) -> str:
    """Iteratively refine an attacker prompt using judge feedback strings."""
    prompt = initial_prompt
    for step, feedback in enumerate(feedbacks):
        prompt = refine_fn(prompt, feedback, step)
    return prompt
