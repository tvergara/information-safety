"""Tests for arithmetic coding utilities."""

from __future__ import annotations

import torch

from information_safety.algorithms.utils.arithmetic_coding import (
    compute_bits_from_logits,
)


class TestComputeBitsFromLogits:
    def test_returns_positive_bits(self) -> None:
        batch_size, seq_len, vocab_size = 2, 5, 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        bits = compute_bits_from_logits(logits, input_ids, mask)
        assert bits > 0

    def test_confident_predictions_yield_fewer_bits(self) -> None:
        batch_size, seq_len, vocab_size = 1, 10, 5
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        confident_logits = torch.full((batch_size, seq_len, vocab_size), -10.0)
        confident_logits[:, :, 0] = 10.0

        uniform_logits = torch.zeros(batch_size, seq_len, vocab_size)

        confident_bits = compute_bits_from_logits(confident_logits, input_ids, mask)
        uniform_bits = compute_bits_from_logits(uniform_logits, input_ids, mask)

        assert confident_bits < uniform_bits

    def test_respects_attention_mask(self) -> None:
        batch_size, seq_len, vocab_size = 1, 6, 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        full_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        partial_mask = torch.tensor([[1, 1, 1, 0, 0, 0]])

        full_bits = compute_bits_from_logits(logits, input_ids, full_mask)
        partial_bits = compute_bits_from_logits(logits, input_ids, partial_mask)

        assert partial_bits < full_bits

    def test_full_mask_equivalent_to_all_ones(self) -> None:
        batch_size, seq_len, vocab_size = 1, 4, 8
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        bits = compute_bits_from_logits(logits, input_ids, mask)
        assert bits > 0

    def test_returns_int(self) -> None:
        logits = torch.randn(1, 3, 5)
        input_ids = torch.randint(0, 5, (1, 3))
        mask = torch.ones(1, 3, dtype=torch.long)

        bits = compute_bits_from_logits(logits, input_ids, mask)
        assert isinstance(bits, int)
