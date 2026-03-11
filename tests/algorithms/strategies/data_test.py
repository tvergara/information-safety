"""Tests for DataStrategy."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from information_safety.algorithms.strategies.data import DataStrategy


class TestDataStrategy:
    def test_init_defaults(self) -> None:
        strategy = DataStrategy()
        assert strategy.lr == 1e-4
        assert strategy._bits == 0

    def test_setup_returns_model_unchanged(self) -> None:
        strategy = DataStrategy()
        model = MagicMock()
        result = strategy.setup(model, MagicMock(), MagicMock())
        assert result is model

    def test_compute_bits_returns_accumulated(self) -> None:
        strategy = DataStrategy()
        strategy._bits = 12345
        assert strategy.compute_bits(MagicMock()) == 12345

    def test_configure_optimizers_returns_adam(self) -> None:
        strategy = DataStrategy(lr=0.01)
        model = MagicMock()
        model.parameters = MagicMock(
            return_value=[torch.nn.Parameter(torch.randn(2, 2))]
        )
        optimizer = strategy.configure_optimizers(model)
        assert isinstance(optimizer, torch.optim.Adam)

    @patch(
        "information_safety.algorithms.strategies.data.compute_bits_from_logits"
    )
    def test_training_step_accumulates_bits(
        self, mock_compute_bits: MagicMock
    ) -> None:
        mock_compute_bits.return_value = 100

        strategy = DataStrategy()

        model = MagicMock()
        logits = torch.randn(2, 5, 10)
        model.return_value.loss = torch.tensor(1.5)
        model.return_value.logits = logits

        batch = {
            "input_ids": torch.randint(0, 10, (2, 5)),
            "attention_mask": torch.ones(2, 5, dtype=torch.long),
            "labels": torch.randint(0, 10, (2, 5)),
        }

        loss = strategy.training_step(model, batch)

        assert strategy._bits == 100
        assert loss.item() == 1.5
        mock_compute_bits.assert_called_once()

    @patch(
        "information_safety.algorithms.strategies.data.compute_bits_from_logits"
    )
    def test_training_step_accumulates_across_batches(
        self, mock_compute_bits: MagicMock
    ) -> None:
        mock_compute_bits.return_value = 50

        strategy = DataStrategy()

        model = MagicMock()
        model.return_value.loss = torch.tensor(1.0)
        model.return_value.logits = torch.randn(1, 3, 5)

        batch = {
            "input_ids": torch.randint(0, 5, (1, 3)),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "labels": torch.randint(0, 5, (1, 3)),
        }

        strategy.training_step(model, batch)
        strategy.training_step(model, batch)

        assert strategy._bits == 100
