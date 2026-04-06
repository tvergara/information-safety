"""Tests for steering primitives."""

from __future__ import annotations

import pytest
import torch

from information_safety.algorithms.jailbreak.steering import (
    ablate_direction,
    apply_activation_addition,
    apply_logit_steering,
    compute_refusal_direction,
)


class TestComputeRefusalDirection:
    def test_returns_unit_vector(self) -> None:
        harmful = torch.tensor([[2.0, 1.0], [2.0, 1.0]])
        benign = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        direction = compute_refusal_direction(harmful, benign)
        assert torch.isclose(direction.norm(), torch.tensor(1.0))

    def test_raises_when_zero_direction(self) -> None:
        harmful = torch.tensor([[1.0, 1.0]])
        benign = torch.tensor([[1.0, 1.0]])
        with pytest.raises(ValueError, match="zero norm"):
            compute_refusal_direction(harmful, benign)


class TestApplyActivationAddition:
    def test_adds_scaled_direction(self) -> None:
        acts = torch.tensor([[1.0, 2.0]])
        direction = torch.tensor([1.0, 0.0])
        output = apply_activation_addition(acts, direction, alpha=0.5)
        assert output.tolist() == [[1.5, 2.0]]


class TestAblateDirection:
    def test_removes_projection_on_direction(self) -> None:
        acts = torch.tensor([[3.0, 4.0]])
        direction = torch.tensor([1.0, 0.0])
        output = ablate_direction(acts, direction)
        assert output.tolist() == [[0.0, 4.0]]


class TestApplyLogitSteering:
    def test_updates_selected_indices(self) -> None:
        logits = torch.tensor([[0.0, 0.0, 0.0]])
        steered = apply_logit_steering(
            logits,
            refusal_token_ids=[0],
            compliance_token_ids=[2],
            gamma=0.5,
        )
        assert steered.tolist() == [[-0.5, 0.0, 0.5]]
