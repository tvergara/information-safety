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
    def test_configure_model_skips_get_train_dataset_when_strategy_does_not_train(
        self, mock_load_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        module = self._make_module()
        module.network_config = MagicMock()
        module.network_config.pretrained_model_name_or_path = "test-model"
        module.tokenizer_config = MagicMock()
        module.tokenizer_config.pretrained_model_name_or_path = "test-model"

        mock_load_model.return_value = MagicMock()
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tokenizer.return_value = mock_tok

        module.strategy.requires_training_data = False  # type: ignore[misc]
        module.strategy.compute_bits = MagicMock(return_value=1000)
        mock_get_train = MagicMock()
        mock_get_val = MagicMock()
        module.dataset_handler.get_train_dataset = mock_get_train
        module.dataset_handler.get_val_dataset = mock_get_val

        module.configure_model()

        mock_get_train.assert_not_called()
        mock_get_val.assert_called_once()

    @patch(
        "information_safety.algorithms.attack_with_strategy.AutoTokenizer.from_pretrained"
    )
    @patch(
        "information_safety.algorithms.attack_with_strategy.load_model"
    )
    def test_configure_model_calls_get_train_dataset_when_strategy_trains(
        self, mock_load_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        module = self._make_module()
        module.network_config = MagicMock()
        module.network_config.pretrained_model_name_or_path = "test-model"
        module.tokenizer_config = MagicMock()
        module.tokenizer_config.pretrained_model_name_or_path = "test-model"

        mock_load_model.return_value = MagicMock()
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tokenizer.return_value = mock_tok

        module.strategy.requires_training_data = True  # type: ignore[misc]
        module.strategy.compute_bits = MagicMock(return_value=1000)
        mock_get_train = MagicMock()
        mock_get_val = MagicMock()
        module.dataset_handler.get_train_dataset = mock_get_train
        module.dataset_handler.get_val_dataset = mock_get_val

        module.configure_model()

        mock_get_train.assert_called_once()
        mock_get_val.assert_called_once()

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

    def test_train_dataloader_returns_empty_when_strategy_does_not_train(self) -> None:
        module = self._make_module()
        module.strategy.requires_training_data = False  # type: ignore[misc]
        mock_strategy_loader = MagicMock()
        module.strategy.train_dataloader = mock_strategy_loader

        loader = module.train_dataloader()

        mock_strategy_loader.assert_not_called()
        assert len(list(loader)) == 0

    def test_train_dataloader_delegates_when_strategy_trains(self) -> None:
        module = self._make_module()
        module.strategy.requires_training_data = True  # type: ignore[misc]
        module._tokenizer = MagicMock()
        sentinel = MagicMock()
        module.strategy.train_dataloader = MagicMock(return_value=sentinel)

        result = module.train_dataloader()

        module.strategy.train_dataloader.assert_called_once()
        assert result is sentinel

    def test_extract_strategy_hparams_omits_class_level_flag(self) -> None:
        from dataclasses import dataclass

        from information_safety.algorithms.attack_with_strategy import (
            _extract_strategy_hparams,
        )
        from information_safety.algorithms.strategies.baseline import BaselineStrategy

        @dataclass
        class _Concrete(BaselineStrategy):
            pass

        hparams = _extract_strategy_hparams(_Concrete())
        assert "requires_training_data" not in hparams
