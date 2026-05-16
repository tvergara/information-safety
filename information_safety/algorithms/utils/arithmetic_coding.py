"""Arithmetic coding utilities for measuring data bits."""

from __future__ import annotations

import math

import torch

_SEQ_CHUNK_TOKENS = 64


def compute_bits_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> int:
    """Compute the number of bits needed to encode input_ids given model logits.

    Uses the fast log-softmax approach: bits = ceil(-sum(log_probs) / log(2)).
    This measures how compressible the data is under the model's predictions.
    """
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].to(shift_logits.dtype)

    seq_len = shift_logits.shape[1]
    total_log_prob = 0.0
    for start in range(0, seq_len, _SEQ_CHUNK_TOKENS):
        end = start + _SEQ_CHUNK_TOKENS
        chunk_logits = shift_logits[:, start:end, :]
        chunk_targets = shift_targets[:, start:end]
        chunk_mask = shift_mask[:, start:end]

        chunk_log_probs = torch.log_softmax(chunk_logits, dim=-1)
        chunk_token_lp = torch.gather(
            chunk_log_probs, 2, chunk_targets.unsqueeze(-1)
        ).squeeze(-1)
        total_log_prob += (chunk_token_lp * chunk_mask).sum().item()

    bits = -total_log_prob / math.log(2)
    return int(math.ceil(bits))
