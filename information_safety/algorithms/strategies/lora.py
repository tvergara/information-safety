"""LoRA finetuning strategy."""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model

from information_safety.algorithms.dataset_handlers.base import BaseDatasetHandler
from information_safety.algorithms.strategies.base import BaseStrategy

logger = getLogger(__name__)


@dataclass
class LoRAStrategy(BaseStrategy):
    """LoRA finetuning strategy using PEFT.

    Bits are computed as 16 * num_trainable_params (assuming bfloat16).
    """

    lr: float = 1e-3
    r: int = 1
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    target_modules: list[str] | None = None

    def setup(
        self,
        model: Any,
        handler: BaseDatasetHandler,
        pl_module: Any,
    ) -> Any:
        """Apply LoRA adapter to the model."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules or "all-linear",
        )
        peft_model = get_peft_model(model, lora_config)

        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        logger.info(
            f"LoRA: {trainable_params:,} trainable / {total_params:,} total params "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

        return peft_model

    def compute_bits(self, model: Any) -> int:
        """Compute bits = 16 * number of trainable parameters."""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return trainable_params * 16

    def configure_optimizers(self, model: Any) -> torch.optim.AdamW:
        """Return AdamW optimizer for trainable parameters."""
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable_params, lr=self.lr)

    def training_step(self, model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the PEFT model, return loss."""
        outputs = model(**batch)
        return outputs.loss
