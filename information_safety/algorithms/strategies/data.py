"""Data strategy: full fine-tuning with data-bits measurement via arithmetic coding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from information_safety.algorithms.dataset_handlers.base import BaseDatasetHandler
from information_safety.algorithms.strategies.base import BaseStrategy
from information_safety.algorithms.utils.arithmetic_coding import (
    compute_bits_from_logits,
)


@dataclass
class DataStrategy(BaseStrategy):
    """Full-parameter fine-tuning with Adam, measuring bits from data compressibility.

    Unlike LoRA which measures bits as parameter count * 16, this strategy measures how many bits
    the model needs to encode the training data, accumulated across all training batches.
    """

    lr: float = 1e-4
    _bits: int = field(default=0, repr=False)

    def setup(
        self,
        model: Any,
        handler: BaseDatasetHandler,
        pl_module: Any,
    ) -> Any:
        return model

    def compute_bits(self, model: Any) -> int:
        return self._bits

    def configure_optimizers(self, model: Any) -> torch.optim.Adam:
        return torch.optim.Adam(model.parameters(), lr=self.lr)

    def training_step(
        self, model: Any, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        with torch.no_grad():
            self._bits += compute_bits_from_logits(
                outputs.logits,
                batch["input_ids"],
                batch["attention_mask"],
            )

        return outputs.loss
