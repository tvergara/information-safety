"""Tests for prompt-based jailbreak prompt builders."""

from __future__ import annotations

import pytest

from information_safety.algorithms.jailbreak.prompt_methods import (
    PromptAttackMethod,
    PromptExample,
    build_prompt,
    build_prompt_attack_record,
)


class TestBuildPrompt:
    def test_zero_shot_returns_behavior(self) -> None:
        behavior = "Explain X"
        output = build_prompt(PromptAttackMethod.ZERO_SHOT, behavior)
        assert output == behavior

    def test_few_shot_includes_examples_and_final_query(self) -> None:
        examples = [
            PromptExample(user="u1", assistant="a1"),
            PromptExample(user="u2", assistant="a2"),
        ]
        output = build_prompt(
            PromptAttackMethod.FEW_SHOT,
            "final request",
            examples=examples,
            max_examples=2,
        )

        assert "Example 1" in output
        assert "User: u1" in output
        assert "Assistant: a1" in output
        assert "User: final request" in output
        assert output.endswith("Assistant:")

    def test_few_shot_requires_examples(self) -> None:
        with pytest.raises(ValueError, match="at least one example"):
            build_prompt(PromptAttackMethod.FEW_SHOT, "x", examples=[])

    def test_roleplay_includes_scenario(self) -> None:
        output = build_prompt(
            PromptAttackMethod.ROLEPLAY,
            "request",
            scenario="a model behavior audit",
        )
        assert "Roleplay scenario: a model behavior audit" in output
        assert "Request: request" in output

    def test_empty_behavior_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            build_prompt(PromptAttackMethod.ZERO_SHOT, "   ")

    def test_human_jailbreak_uses_template(self) -> None:
        output = build_prompt(
            PromptAttackMethod.HUMAN_JAILBREAK,
            "target behavior",
            human_jailbreak_template="Override: {behavior}",
        )
        assert output == "Override: target behavior"

    def test_human_jailbreak_requires_template(self) -> None:
        with pytest.raises(ValueError, match="template"):
            build_prompt(PromptAttackMethod.HUMAN_JAILBREAK, "target behavior")


class TestBuildPromptAttackRecord:
    def test_returns_method_prompt_and_bits(self) -> None:
        record = build_prompt_attack_record(PromptAttackMethod.ZERO_SHOT, "abcd")
        assert record["attack_method"] == "zero_shot"
        assert record["attack_prompt"] == "abcd"
        assert record["bits"] == 32
