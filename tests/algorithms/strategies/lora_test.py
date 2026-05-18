"""Tests for LoRA strategy."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from torch import nn

from information_safety.algorithms.strategies.lora import LoRAStrategy
from information_safety.attack_bits import STRATEGY_CODE_FILES, code_bits


def _peft_model_with_one_param() -> MagicMock:
    peft_model = MagicMock()
    param = nn.Parameter(torch.zeros(10))
    param.requires_grad = True
    peft_model.parameters = MagicMock(return_value=[param])
    return peft_model


class TestLoRAStrategy:
    def test_init_defaults(self) -> None:
        strategy = LoRAStrategy(lr=1e-3, r=1, lora_alpha=8)
        assert strategy.lr == 1e-3
        assert strategy.r == 1
        assert strategy.lora_alpha == 8

    @patch("information_safety.algorithms.strategies.lora.get_peft_model")
    @patch("information_safety.algorithms.strategies.lora.LoraConfig")
    def test_setup_applies_peft(
        self, mock_lora_config: MagicMock, mock_get_peft_model: MagicMock
    ) -> None:
        strategy = LoRAStrategy(lr=1e-3, r=1, lora_alpha=8)

        mock_model = MagicMock()
        mock_peft_model = _peft_model_with_one_param()
        mock_get_peft_model.return_value = mock_peft_model

        result = strategy.setup(mock_model, MagicMock(), MagicMock())
        mock_lora_config.assert_called_once()
        mock_get_peft_model.assert_called_once()
        assert result is mock_peft_model

    @patch("information_safety.algorithms.strategies.lora.get_peft_model")
    @patch("information_safety.algorithms.strategies.lora.LoraConfig")
    def test_setup_default_skips_gradient_checkpointing(
        self, mock_lora_config: MagicMock, mock_get_peft_model: MagicMock
    ) -> None:
        strategy = LoRAStrategy(lr=1e-3, r=1, lora_alpha=8)
        mock_model = MagicMock()
        mock_get_peft_model.return_value = _peft_model_with_one_param()

        strategy.setup(mock_model, MagicMock(), MagicMock())

        mock_model.gradient_checkpointing_enable.assert_not_called()
        mock_model.enable_input_require_grads.assert_not_called()

    @patch("information_safety.algorithms.strategies.lora.get_peft_model")
    @patch("information_safety.algorithms.strategies.lora.LoraConfig")
    def test_setup_enables_gradient_checkpointing(
        self, mock_lora_config: MagicMock, mock_get_peft_model: MagicMock
    ) -> None:
        strategy = LoRAStrategy(
            lr=1e-3, r=1, lora_alpha=8, gradient_checkpointing=True,
        )
        mock_model = MagicMock()
        mock_get_peft_model.return_value = _peft_model_with_one_param()

        strategy.setup(mock_model, MagicMock(), MagicMock())

        mock_model.gradient_checkpointing_enable.assert_called_once()
        mock_model.enable_input_require_grads.assert_called_once()

    def test_compute_bits_counts_trainable_params(self) -> None:
        strategy = LoRAStrategy(lr=1e-3, r=1, lora_alpha=8)

        model = MagicMock()
        param1 = nn.Parameter(torch.zeros(10, 5))
        param1.requires_grad = True
        param2 = nn.Parameter(torch.zeros(20, 3))
        param2.requires_grad = False
        param3 = nn.Parameter(torch.zeros(8, 4))
        param3.requires_grad = True

        model.parameters = MagicMock(return_value=[param1, param2, param3])

        bits = strategy.compute_bits(model)
        expected = code_bits(STRATEGY_CODE_FILES["lora"]) + (10 * 5 + 8 * 4) * 16
        assert bits == expected

    def test_configure_optimizers_returns_adamw(self) -> None:
        strategy = LoRAStrategy(lr=1e-3, r=1, lora_alpha=8)

        model = MagicMock()
        params = [nn.Parameter(torch.zeros(5))]
        model.parameters = MagicMock(return_value=params)

        optimizer = strategy.configure_optimizers(model)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_training_step_delegates_to_model(self) -> None:
        strategy = LoRAStrategy(lr=1e-3, r=1, lora_alpha=8)

        model = MagicMock()
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(0.5)
        model.return_value = mock_output

        batch = {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[1, 2]])}

        loss = strategy.training_step(model, batch)
        model.assert_called_once()
        assert loss == mock_output.loss
