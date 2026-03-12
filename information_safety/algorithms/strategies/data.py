"""Data strategy: LoRA fine-tuning with data-bits measurement via arithmetic coding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from information_safety.algorithms.strategies.lora import LoRAStrategy
from information_safety.algorithms.utils.arithmetic_coding import (
    compute_bits_from_logits,
)


@dataclass
class DataStrategy(LoRAStrategy):
    """LoRA fine-tuning that measures bits from data compressibility instead of parameter count.

    Inherits all LoRA setup/optimizer logic, but overrides compute_bits to return the accumulated
    data bits (how many bits needed to encode the training data under the model), and training_step
    to accumulate those bits.
    """

    _bits: int = field(default=0, repr=False)

    def compute_bits(self, model: Any) -> int:
        return self._bits

    def training_step(self, model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = model(**batch)

        with torch.no_grad():
            self._bits += compute_bits_from_logits(
                outputs.logits,
                batch["input_ids"],
                batch["attention_mask"],
            )

        return outputs.loss
