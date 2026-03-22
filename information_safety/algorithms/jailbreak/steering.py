"""Steering primitives for activation-space interventions."""

from __future__ import annotations

import torch


def _normalize(direction: torch.Tensor) -> torch.Tensor:
    norm = direction.norm()
    if torch.isclose(norm, torch.tensor(0.0, device=direction.device)):
        raise ValueError("direction has zero norm")
    return direction / norm


def compute_refusal_direction(
    harmful_activations: torch.Tensor,
    benign_activations: torch.Tensor,
) -> torch.Tensor:
    """Estimate a refusal direction from mean activation differences."""
    harmful_mean = harmful_activations.mean(dim=0)
    benign_mean = benign_activations.mean(dim=0)
    return _normalize(harmful_mean - benign_mean)


def apply_activation_addition(
    activations: torch.Tensor,
    direction: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Apply activation addition with a scaled unit direction."""
    unit_direction = _normalize(direction)
    return activations + alpha * unit_direction


def ablate_direction(activations: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove the projection of activations onto a direction."""
    unit_direction = _normalize(direction)
    coeff = (activations * unit_direction).sum(dim=-1, keepdim=True)
    return activations - coeff * unit_direction


def apply_logit_steering(
    logits: torch.Tensor,
    *,
    refusal_token_ids: list[int],
    compliance_token_ids: list[int],
    gamma: float,
) -> torch.Tensor:
    """Apply additive logit steering for refusal/compliance token sets."""
    steered = logits.clone()
    if refusal_token_ids:
        steered[..., refusal_token_ids] -= gamma
    if compliance_token_ids:
        steered[..., compliance_token_ids] += gamma
    return steered
