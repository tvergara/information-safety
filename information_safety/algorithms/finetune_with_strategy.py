"""Backward-compatibility alias: FinetuneWithStrategy has been renamed to AttackWithStrategy."""

from information_safety.algorithms.attack_with_strategy import (  # noqa: F401
    AttackWithStrategy as FinetuneWithStrategy,
)

__all__ = ["FinetuneWithStrategy"]
