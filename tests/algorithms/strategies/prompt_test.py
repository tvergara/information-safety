"""Tests for prompt-based attack strategies."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from information_safety.algorithms.strategies.prompt import RoleplayStrategy


class TestRoleplayStrategy:
    def _make_strategy(self, scenario: str = "test scenario") -> RoleplayStrategy:
        return RoleplayStrategy(scenario=scenario)

    def test_setup_returns_model_unmodified(self) -> None:
        strategy = self._make_strategy()
        model = MagicMock()
        handler = MagicMock()
        pl_module = MagicMock()

        result = strategy.setup(model, handler, pl_module)
        assert result is model

    def test_configure_optimizers_returns_dummy(self) -> None:
        strategy = self._make_strategy()
        model = MagicMock()

        optimizer = strategy.configure_optimizers(model)
        assert isinstance(optimizer, torch.optim.AdamW)
        params = list(optimizer.param_groups[0]["params"])
        assert len(params) == 1
        assert params[0].numel() == 0

    @patch(
        "information_safety.algorithms.strategies.prompt.compute_cross_entropy_bits",
        return_value=42.0,
    )
    def test_compute_bits_returns_cross_entropy_of_template(
        self, mock_ce: MagicMock
    ) -> None:
        strategy = self._make_strategy(scenario="test scenario")
        model = MagicMock()
        tokenizer = MagicMock()
        strategy._tokenizer = tokenizer

        bits = strategy.compute_bits(model)

        mock_ce.assert_called_once()
        call_kwargs = mock_ce.call_args
        assert call_kwargs.kwargs["model"] is model
        assert call_kwargs.kwargs["tokenizer"] is tokenizer
        template_text = call_kwargs.args[0]
        assert "{behavior}" in template_text
        assert bits == 42

    def test_validation_step_modifies_batch_with_template(self) -> None:
        strategy = self._make_strategy(scenario="an unrestricted assistant")

        model = MagicMock()
        tokenizer = MagicMock()
        strategy._tokenizer = tokenizer

        import json

        metadata = json.dumps({"behavior": "test behavior", "category": "test"})
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": [metadata],
        }

        tokenizer.apply_chat_template.return_value = "formatted prompt"
        tokenizer.return_value = {
            "input_ids": torch.tensor([[10, 20, 30, 40]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
        tokenizer.pad_token_id = 0

        handler = MagicMock()
        handler.validate_batch.return_value = (1, 1)
        handler.max_length = 500

        strategy.validation_step(model, tokenizer, batch, handler)

        tokenizer.apply_chat_template.assert_called_once()
        call_args = tokenizer.apply_chat_template.call_args
        messages = call_args.args[0] if call_args.args else call_args.kwargs["conversation"]
        assert messages[0]["role"] == "user"
        assert "test behavior" in messages[0]["content"]
        assert "Roleplay scenario:" in messages[0]["content"]

        handler.validate_batch.assert_called_once()
        modified_batch = handler.validate_batch.call_args[0][2]
        assert "input_ids" in modified_batch
        assert "attention_mask" in modified_batch
        assert "labels" in modified_batch
