"""FinetuneWithStrategy: a LightningModule that delegates to a strategy and dataset handler."""

from __future__ import annotations

from logging import getLogger
from typing import Any

import torch
from lightning import LightningModule
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from information_safety.algorithms.dataset_handlers.base import BaseDatasetHandler
from information_safety.algorithms.strategies.base import BaseStrategy
from information_safety.algorithms.utils.data_classes import NetworkConfig, load_model

logger = getLogger(__name__)


class FinetuneWithStrategy(LightningModule):
    """LightningModule that delegates training logic to a strategy + dataset handler."""

    def __init__(
        self,
        network_config: NetworkConfig,
        tokenizer_config: Any,
        strategy: BaseStrategy,
        dataset_handler: BaseDatasetHandler,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        self.network_config = network_config
        self.tokenizer_config = tokenizer_config
        self.strategy = strategy
        self.dataset_handler = dataset_handler

        self.model: Any = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

        self._val_correct = 0
        self._val_total = 0
        self._val_results: list[dict[str, Any]] = []

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Return the tokenizer.

        Only valid after configure_model() has been called.
        """
        assert self._tokenizer is not None, "configure_model() must be called first"
        return self._tokenizer

    def configure_model(self) -> None:
        """Load the model and tokenizer, then apply the strategy."""
        if self.model is not None:
            return

        self.model = load_model(self.network_config)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_config.pretrained_model_name_or_path,
            trust_remote_code=self.tokenizer_config.trust_remote_code,
            padding_side="left",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self.strategy.setup(self.model, self.dataset_handler, self)

        self.strategy.train_dataset = self.dataset_handler.get_train_dataset(self.tokenizer)
        self.strategy.val_dataset = self.dataset_handler.get_val_dataset(self.tokenizer)

        bits = self.strategy.compute_bits(self.model)
        logger.info(f"Total bits for this strategy: {bits:,}")

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self.strategy.training_step(self.model, batch)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int
    ) -> None:
        correct, total = self.strategy.validation_step(
            self.model, self.tokenizer, batch, self.dataset_handler
        )
        self._val_correct += correct
        self._val_total += total

    def on_validation_epoch_end(self) -> None:
        if self._val_total > 0:
            performance = self._val_correct / self._val_total
        else:
            performance = 0.0

        self.log("val/performance", performance, prog_bar=True, sync_dist=True)

        bits = self.strategy.compute_bits(self.model)
        self._val_results.append({
            "epoch": self.current_epoch,
            "performance": performance,
            "bits": bits,
            "correct": self._val_correct,
            "total": self._val_total,
        })

        self.strategy.on_validation_epoch_end(self, self._val_results[-1:])
        self.dataset_handler.save_completions(f"epoch-{self.current_epoch}")

        self._val_correct = 0
        self._val_total = 0

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.strategy.configure_optimizers(self.model)

    def train_dataloader(self):
        return self.strategy.train_dataloader(
            self.tokenizer,
            batch_size=self.dataset_handler.batch_size,
        )

    def val_dataloader(self):
        return self.strategy.val_dataloader(
            self.tokenizer,
            batch_size=self.dataset_handler.batch_size,
        )
