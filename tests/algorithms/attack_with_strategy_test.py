"""Tests for AttackWithStrategy LightningModule."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from information_safety.algorithms.attack_with_strategy import AttackWithStrategy


class TestAttackWithStrategy:
    def _make_module(self) -> AttackWithStrategy:
        network_config = MagicMock()
        tokenizer_config = MagicMock()
        strategy = MagicMock()
        dataset_handler = MagicMock()
        return AttackWithStrategy(
            network_config=network_config,
            tokenizer_config=tokenizer_config,
            strategy=strategy,
            dataset_handler=dataset_handler,
        )

    @patch(
        "information_safety.algorithms.attack_with_strategy.AutoTokenizer.from_pretrained"
    )
    @patch(
        "information_safety.algorithms.attack_with_strategy.load_model"
    )
    def test_configure_model_loads_model_and_tokenizer(
        self, mock_load_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        module = self._make_module()
        module.network_config = MagicMock()
        module.network_config.pretrained_model_name_or_path = "test-model"
        module.network_config.from_config = False
        module.tokenizer_config = MagicMock()
        module.tokenizer_config.pretrained_model_name_or_path = "test-model"

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "</s>"
        mock_tokenizer.return_value = mock_tok

        module.strategy.compute_bits = MagicMock(return_value=1000)

        module.configure_model()
        mock_load_model.assert_called_once()
        mock_tokenizer.assert_called_once()

    @patch(
        "information_safety.algorithms.attack_with_strategy.AutoTokenizer.from_pretrained"
    )
    @patch(
        "information_safety.algorithms.attack_with_strategy.load_model"
    )
    def test_pad_token_set_to_eos_when_missing(
        self, mock_load_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        module = self._make_module()
        module.network_config = MagicMock()
        module.network_config.pretrained_model_name_or_path = "test-model"
        module.network_config.from_config = False
        module.tokenizer_config = MagicMock()
        module.tokenizer_config.pretrained_model_name_or_path = "test-model"

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "</s>"
        mock_tokenizer.return_value = mock_tok

        module.strategy.compute_bits = MagicMock(return_value=1000)

        module.configure_model()
        assert mock_tok.pad_token == "</s>"

    def test_training_step_delegates_to_strategy(self) -> None:
        module = self._make_module()
        module.model = MagicMock()
        module.strategy = MagicMock()
        module.strategy.training_step = MagicMock(return_value=torch.tensor(0.5))
        mock_trainer = MagicMock()
        mock_trainer.sanity_checking = False
        module._trainer = mock_trainer

        attention_mask = torch.ones(1, 2, dtype=torch.long)
        batch = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": attention_mask}
        module.training_step(batch, 0)
        module.strategy.training_step.assert_called_once()

    def test_configure_optimizers_delegates_to_strategy(self) -> None:
        module = self._make_module()
        module.model = MagicMock()
        module.strategy = MagicMock()
        mock_optimizer = MagicMock()
        module.strategy.configure_optimizers = MagicMock(return_value=mock_optimizer)

        result = module.configure_optimizers()
        module.strategy.configure_optimizers.assert_called_once()
        assert result is mock_optimizer
