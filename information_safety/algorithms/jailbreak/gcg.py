"""Core GCG helper primitives for optimization-based jailbreak attacks."""

from __future__ import annotations

import torch


def candidate_tokens_from_gradient(
    gradient: torch.Tensor,
    embedding_matrix: torch.Tensor,
    *,
    top_k: int,
) -> list[int]:
    """Return token ids with top alignment to the gradient direction."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if gradient.ndim != 1:
        raise ValueError("gradient must be rank-1")
    if embedding_matrix.ndim != 2:
        raise ValueError("embedding_matrix must be rank-2")
    if embedding_matrix.shape[1] != gradient.shape[0]:
        raise ValueError("embedding and gradient dimensions must match")

    scores = embedding_matrix @ gradient
    top_indices = torch.topk(scores, k=min(top_k, scores.shape[0])).indices
    return [int(idx) for idx in top_indices.tolist()]


def apply_coordinate_update(suffix_token_ids: list[int], *, position: int, new_token_id: int) -> list[int]:
    """Return a new suffix with one coordinate update applied."""
    if position < 0 or position >= len(suffix_token_ids):
        raise IndexError("position out of range")
    updated = list(suffix_token_ids)
    updated[position] = new_token_id
    return updated
