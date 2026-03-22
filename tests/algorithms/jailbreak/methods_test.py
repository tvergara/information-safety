"""Tests for jailbreak method registry enums."""

from __future__ import annotations

from information_safety.algorithms.jailbreak.methods import (
    OptimizationAttackMethod,
    PromptAttackMethodFamily,
    SteeringAttackMethod,
)


class TestPromptAttackMethodFamily:
    def test_contains_baselines(self) -> None:
        assert PromptAttackMethodFamily.ZERO_SHOT.value == "zero_shot"
        assert PromptAttackMethodFamily.FEW_SHOT.value == "few_shot"
        assert PromptAttackMethodFamily.HUMAN_JAILBREAK.value == "human_jailbreak"


class TestOptimizationAttackMethod:
    def test_contains_discussed_methods(self) -> None:
        assert OptimizationAttackMethod.GCG.value == "gcg"
        assert OptimizationAttackMethod.UAT.value == "uat"
        assert OptimizationAttackMethod.PAIR.value == "pair"
        assert OptimizationAttackMethod.TAP.value == "tap"
        assert OptimizationAttackMethod.AUTODAN.value == "autodan"


class TestSteeringAttackMethod:
    def test_contains_discussed_methods(self) -> None:
        assert SteeringAttackMethod.ACTIVATION_ADDITION.value == "activation_addition"
        assert SteeringAttackMethod.REFUSAL_ABLATION.value == "refusal_ablation"
        assert SteeringAttackMethod.LOGIT_STEERING.value == "logit_steering"
