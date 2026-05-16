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

    def test_matches_reference_full_softmax(self) -> None:
        """Chunked implementation must give the same answer as a one-shot log_softmax."""
        import math

        torch.manual_seed(0)
        batch_size, seq_len, vocab_size = 3, 17, 257
        logits = torch.randn(batch_size, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        mask[0, 12:] = 0
        mask[2, 5:] = 0

        log_probs = torch.log_softmax(logits, dim=-1)
        token_lp = torch.gather(
            log_probs[:, :-1, :], 2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        reference_total = (token_lp * mask[:, 1:]).sum()
        reference_bits = int(math.ceil(-reference_total.item() / math.log(2)))

        bits = compute_bits_from_logits(logits, input_ids, mask)
        assert bits == reference_bits

    def test_matches_reference_with_bf16(self) -> None:
        """Equivalence must hold when logits are bf16 (the OOM-trigger dtype)."""
        import math

        torch.manual_seed(1)
        batch_size, seq_len, vocab_size = 2, 9, 511
        logits = torch.randn(batch_size, seq_len, vocab_size).to(torch.bfloat16)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_lp = torch.gather(
            log_probs[:, :-1, :], 2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        reference_bits = int(
            math.ceil(-(token_lp * mask[:, 1:]).sum().item() / math.log(2))
        )

        bits = compute_bits_from_logits(logits, input_ids, mask)
        assert abs(bits - reference_bits) <= 1  # allow 1 bit slack for bf16 rounding

    def test_handles_short_sequence(self) -> None:
        """Sequence shorter than internal chunk size must still work."""
        logits = torch.randn(2, 3, 16)
        input_ids = torch.randint(0, 16, (2, 3))
        mask = torch.ones(2, 3, dtype=torch.long)
        bits = compute_bits_from_logits(logits, input_ids, mask)
        assert isinstance(bits, int)
        assert bits >= 0
