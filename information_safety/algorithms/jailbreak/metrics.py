"""Utilities for attack information accounting."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def compute_text_bits(text: str) -> int:
    """Return the bit-length of text under UTF-8 encoding."""
    return len(text.encode("utf-8")) * 8


def compute_cross_entropy_bits(
    text: str,
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> float:
    """Compute the cross-entropy bit cost of *text* under *model*.

    Returns the sum of -log2 p(token_i | token_1, ..., token_{i-1}) over all
    tokens after the first, which measures how many bits the model needs to
    encode this text.
    """
    encoding = tokenizer(text, return_tensors="pt")
    input_ids = torch.as_tensor(encoding["input_ids"])
    if input_ids.shape[1] <= 1:
        return 0.0

    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        logits: torch.Tensor = model(input_ids).logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    loss_per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
    )
    total_nats = loss_per_token.sum().item()
    return total_nats / math.log(2)


def aggregate_bits_by_method(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute average bits per attack method from result rows."""
    grouped: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        method = row.get("attack_method") or row.get("experiment_name")
        bits = row.get("bits")
        if method is None or bits is None:
            continue
        if isinstance(bits, int | float):
            grouped[str(method)].append(float(bits))

    return {
        method: sum(values) / len(values)
        for method, values in grouped.items()
        if values
    }
