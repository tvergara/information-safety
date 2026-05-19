"""Tests for BaseStrategy's default FLOPs accounting hooks."""

from __future__ import annotations

from typing import Any

import torch

from information_safety.algorithms.dataset_handlers.base import BaseDatasetHandler
from information_safety.algorithms.strategies.base import BaseStrategy


class _ConcreteStrategy(BaseStrategy):
    """Minimal concrete strategy for testing BaseStrategy defaults."""

    def setup(
        self, model: Any, handler: BaseDatasetHandler, pl_module: Any,
    ) -> Any:
        return model

    def compute_bits(self, model: Any) -> int:
        return 0

    def configure_optimizers(self, model: Any) -> torch.optim.Optimizer:
        dummy = torch.nn.Parameter(torch.empty(0), requires_grad=True)
        return torch.optim.AdamW([dummy])


class TestAttackFlopsFor:
    def test_default_returns_zero(self) -> None:
        strategy = _ConcreteStrategy()
        assert strategy.attack_flops_for({"behavior": "b"}) == 0

    def test_default_ignores_meta_contents(self) -> None:
        strategy = _ConcreteStrategy()
        assert strategy.attack_flops_for({"attack_flops": 999}) == 0


class TestEvalForwardFlops:
    def test_returns_two_n_times_total_tokens(self) -> None:
        strategy = _ConcreteStrategy()
        assert strategy.eval_forward_flops(num_params=1_000_000, n_input=5, n_output=3) == (
            2 * 1_000_000 * (5 + 3)
        )

    def test_zero_tokens_returns_zero(self) -> None:
        strategy = _ConcreteStrategy()
        assert strategy.eval_forward_flops(num_params=1_000_000, n_input=0, n_output=0) == 0

    def test_returns_int(self) -> None:
        strategy = _ConcreteStrategy()
        result = strategy.eval_forward_flops(num_params=7, n_input=2, n_output=4)
        assert isinstance(result, int)
