"""Tests for PAIR helper primitives."""

from __future__ import annotations

from information_safety.algorithms.jailbreak.pair import run_pair_refinement


def test_run_pair_refinement_uses_feedback_each_step() -> None:
    feedbacks = ["f1", "f2", "f3"]

    def refine(prompt: str, feedback: str, step: int) -> str:
        return f"{prompt}|{feedback}|{step}"

    result = run_pair_refinement("seed", feedbacks, refine)
    assert result.endswith("|f3|2")
