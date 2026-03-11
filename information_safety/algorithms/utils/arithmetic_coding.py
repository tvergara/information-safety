"""Arithmetic coding utilities for measuring data bits."""

from __future__ import annotations

import math

import torch


def compute_bits_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> int:
    """Compute the number of bits needed to encode input_ids given model logits.

    Uses the fast log-softmax approach: bits = ceil(-sum(log_probs) / log(2)).
    This measures how compressible the data is under the model's predictions.
    """
    log_probs = torch.log_softmax(logits, dim=-1)

    logits_for_pred = log_probs[:, :-1, :]
    target_tokens = input_ids[:, 1:]

    token_log_probs = torch.gather(
        logits_for_pred, 2, target_tokens.unsqueeze(-1)
    ).squeeze(-1)

    mask = attention_mask[:, 1:]
    token_log_probs = token_log_probs * mask

    total_log_prob = token_log_probs.sum()
    bits = (-total_log_prob / math.log(2)).item()

    return int(math.ceil(bits))
