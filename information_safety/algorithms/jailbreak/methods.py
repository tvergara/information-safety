"""Canonical method names for phase-1 jailbreak benchmarking."""

from __future__ import annotations

from enum import StrEnum


class PromptAttackMethodFamily(StrEnum):
    """Prompt-only baseline attack methods."""

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    HUMAN_JAILBREAK = "human_jailbreak"
    ROLEPLAY = "roleplay"


class OptimizationAttackMethod(StrEnum):
    """Optimization-based jailbreak methods discussed for phase 1."""

    GCG = "gcg"
    UAT = "uat"
    PAIR = "pair"
    TAP = "tap"
    AUTODAN = "autodan"


class SteeringAttackMethod(StrEnum):
    """Steering-based methods discussed for phase 1."""

    ACTIVATION_ADDITION = "activation_addition"
    REFUSAL_ABLATION = "refusal_ablation"
    LOGIT_STEERING = "logit_steering"
