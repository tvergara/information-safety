"""Baseline strategy: no training, just evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from information_safety.algorithms.dataset_handlers.base import BaseDatasetHandler
from information_safety.algorithms.strategies.base import BaseStrategy
from information_safety.attack_bits import STRATEGY_CODE_FILES, code_bits


@dataclass
class BaselineStrategy(BaseStrategy):
    """Strategy that skips training and evaluates the model as-is.

    Program length is the code-bits floor (the naive inference script).
    """

    def setup(
        self,
        model: Any,
        handler: BaseDatasetHandler,
        pl_module: Any,
    ) -> Any:
        return model

    def compute_bits(self, model: Any) -> int:
        return code_bits(STRATEGY_CODE_FILES["baseline"])

    def configure_optimizers(self, model: Any) -> torch.optim.AdamW:
        dummy = torch.nn.Parameter(torch.empty(0), requires_grad=True)
        return torch.optim.AdamW([dummy])
