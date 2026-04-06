"""Tests for jailbreak attack bit-accounting utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from information_safety.algorithms.jailbreak.metrics import (
    aggregate_bits_by_method,
    compute_cross_entropy_bits,
    compute_text_bits,
)


class TestComputeTextBits:
    def test_returns_utf8_bit_count(self) -> None:
        assert compute_text_bits("abc") == 24

    def test_empty_string_has_zero_bits(self) -> None:
        assert compute_text_bits("") == 0


class TestComputeCrossEntropyBits:
    def test_returns_sum_of_cross_entropy_in_bits(self) -> None:
        model = MagicMock()
        tokenizer = MagicMock()

        token_ids = torch.tensor([[1, 10, 20, 30]])
        tokenizer.return_value = {"input_ids": token_ids}

        vocab_size = 100
        logits = torch.randn(1, 4, vocab_size)
        model.return_value = MagicMock(logits=logits)
        model.device = torch.device("cpu")

        result = compute_cross_entropy_bits("some text", model=model, tokenizer=tokenizer)

        assert isinstance(result, float)
        assert result > 0

    def test_single_token_returns_zero(self) -> None:
        model = MagicMock()
        tokenizer = MagicMock()

        token_ids = torch.tensor([[42]])
        tokenizer.return_value = {"input_ids": token_ids}
        model.device = torch.device("cpu")

        result = compute_cross_entropy_bits("x", model=model, tokenizer=tokenizer)
        assert result == 0.0


class TestAggregateBitsByMethod:
    def test_aggregates_average_bits_per_method(self) -> None:
        rows = [
            {"attack_method": "zero_shot", "bits": 8},
            {"attack_method": "zero_shot", "bits": 24},
            {"attack_method": "gcg", "bits": 128},
        ]
        result = aggregate_bits_by_method(rows)
        assert result["zero_shot"] == 16.0
        assert result["gcg"] == 128.0

    def test_ignores_rows_without_method_or_bits(self) -> None:
        rows = [{"attack_method": "a"}, {"bits": 12}, {"attack_method": "b", "bits": "bad"}]
        result = aggregate_bits_by_method(rows)
        assert result == {}
