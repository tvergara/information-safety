"""Base strategy for LLM finetuning."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from information_safety.algorithms.dataset_handlers.base import BaseDatasetHandler
from information_safety.algorithms.utils.data_collator import (
    DataCollatorForAnswerOnlyLM,
    DataCollatorForGeneration,
)

logger = getLogger(__name__)


class BaseStrategy(ABC):
    """Base class for finetuning strategies (e.g. LoRA, full finetune, etc.)."""

    train_dataset: Dataset | None = None
    val_dataset: Dataset | None = None

    @abstractmethod
    def setup(
        self,
        model: Any,
        handler: BaseDatasetHandler,
        pl_module: Any,
    ) -> Any:
        """Apply the strategy to the model (e.g. wrap with PEFT).

        Returns the modified model.
        """

    @abstractmethod
    def compute_bits(self, model: Any) -> int:
        """Compute the number of bits used by this strategy's trainable parameters."""

    def attack_flops(self) -> int:
        """FLOPs already spent before this run (e.g. GCG optimizer cost for precomputed
        suffixes)."""
        return 0

    @abstractmethod
    def configure_optimizers(self, model: Any) -> torch.optim.Optimizer:
        """Return the optimizer for the trainable parameters."""

    def training_step(self, model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Default training step: forward pass, return loss."""
        outputs = model(**{k: v for k, v in batch.items() if k != "labels"}, labels=batch["labels"])
        return outputs.loss

    def validation_step(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizerBase,
        batch: dict,
        handler: BaseDatasetHandler,
    ) -> tuple[int, int]:
        """Default validation step: delegate to handler."""
        return handler.validate_batch(model, tokenizer, batch)

    def on_validation_epoch_end(
        self,
        pl_module: Any,
        results: list[dict[str, Any]],
        output_dir: Path | None = None,
    ) -> None:
        """Save JSONL results on rank 0."""
        if pl_module.global_rank != 0:
            return

        if output_dir is None:
            output_dir = Path(pl_module.trainer.default_root_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "validation_results.jsonl"
        with open(output_file, "a") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        logger.info(f"Saved {len(results)} validation results to {output_file}")

    def train_dataloader(
        self,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        num_workers: int = 0,
        answer_start_token: str = "### Answer:",
    ) -> DataLoader:
        """Create training dataloader."""
        assert self.train_dataset is not None
        collator = DataCollatorForAnswerOnlyLM(
            tokenizer=tokenizer, answer_start_token=answer_start_token,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=num_workers,
        )

    def val_dataloader(
        self,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create validation dataloader with left-padding for generation."""
        assert self.val_dataset is not None
        collator = DataCollatorForGeneration(tokenizer=tokenizer)
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
        )
