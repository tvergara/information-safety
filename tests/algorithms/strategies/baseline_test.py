"""Tests for BaselineStrategy."""

from __future__ import annotations

from unittest.mock import MagicMock

from information_safety.algorithms.strategies.baseline import BaselineStrategy


class TestSetup:
    def test_returns_model_unchanged(self) -> None:
        strategy = BaselineStrategy()
        model = MagicMock()
        result = strategy.setup(model, MagicMock(), MagicMock())
        assert result is model



class TestComputeBits:
    def test_returns_zero(self) -> None:
        strategy = BaselineStrategy()
        assert strategy.compute_bits(MagicMock()) == 0


class TestConfigureOptimizers:
    def test_returns_optimizer_with_no_real_params(self) -> None:
        strategy = BaselineStrategy()
        optimizer = strategy.configure_optimizers(MagicMock())
        total_params = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
        assert total_params == 0


class TestAttackFlops:
    def test_default_returns_zero(self) -> None:
        strategy = BaselineStrategy()
        assert strategy.attack_flops() == 0
