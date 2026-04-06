"""Tests for core GCG helper primitives."""

from __future__ import annotations

import torch

from information_safety.algorithms.jailbreak.gcg import (
    apply_coordinate_update,
    candidate_tokens_from_gradient,
    estimate_gcg_search_bits,
)


class TestCandidateTokensFromGradient:
    def test_returns_top_candidate_ids(self) -> None:
        gradient = torch.tensor([1.0, 0.1])
        embedding_matrix = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.9, 0.0],
            ]
        )
        token_ids = candidate_tokens_from_gradient(gradient, embedding_matrix, top_k=2)
        assert token_ids == [0, 2]


class TestApplyCoordinateUpdate:
    def test_replaces_single_position(self) -> None:
        suffix = [10, 11, 12]
        updated = apply_coordinate_update(suffix, position=1, new_token_id=99)
        assert updated == [10, 99, 12]
        assert suffix == [10, 11, 12]


class TestEstimateGcgSearchBits:
    def test_estimates_search_bits(self) -> None:
        bits = estimate_gcg_search_bits(steps=4, suffix_length=2, top_k=8)
        assert bits == 24
