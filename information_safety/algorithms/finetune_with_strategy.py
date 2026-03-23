"""FinetuneWithStrategy: a LightningModule that delegates to a strategy and dataset handler."""

from __future__ import annotations

import dataclasses
import json
import uuid
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from information_safety.algorithms.dataset_handlers.base import BaseDatasetHandler
from information_safety.algorithms.strategies.base import BaseStrategy
from information_safety.algorithms.utils.data_classes import NetworkConfig, load_model

logger = getLogger(__name__)

STRATEGY_HPARAMS_EXCLUDE = frozenset({"train_dataset", "val_dataset"})


def _extract_strategy_hparams(strategy: BaseStrategy) -> dict[str, Any]:
    """Extract JSON-serializable hyperparameters from a strategy dataclass."""
    raw = dataclasses.asdict(strategy)  # type: ignore[arg-type]
    return {
        k: v
        for k, v in raw.items()
        if not k.startswith("_") and k not in STRATEGY_HPARAMS_EXCLUDE
    }


class FinetuneWithStrategy(LightningModule):
    """LightningModule that delegates training logic to a strategy + dataset handler."""

    def __init__(
        self,
        network_config: NetworkConfig,
        tokenizer_config: Any,
        strategy: BaseStrategy,
        dataset_handler: BaseDatasetHandler,
        result_file: str = "",
        seed: int = 0,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        self.network_config = network_config
        self.tokenizer_config = tokenizer_config
        self.strategy = strategy
        self.dataset_handler = dataset_handler
        self.result_file = result_file
        self.seed = seed

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

        if self.trainer.sanity_checking:
            self._val_correct = 0
            self._val_total = 0
            return

        bits = self.strategy.compute_bits(self.model)
        self._val_results.append({
            "epoch": self.current_epoch,
            "performance": performance,
            "bits": bits,
            "correct": self._val_correct,
            "total": self._val_total,
        })

        self.strategy.on_validation_epoch_end(self, self._val_results[-1:])

        eval_run_id = str(uuid.uuid4())
        self.dataset_handler.save_completions(eval_run_id)

        if self.result_file:
            self._write_result_row(eval_run_id, bits)

        self._val_correct = 0
        self._val_total = 0

    def _write_result_row(self, eval_run_id: str, bits: int) -> None:
        """Append a JSONL result row to the centralized results file."""
        experiment_id = None
        if self.trainer.logger is not None:
            experiment_id = getattr(self.trainer.logger.experiment, "id", None)  # type: ignore[union-attr]

        strategy_hparams = _extract_strategy_hparams(self.strategy)

        result_row = {
            "experiment_name": type(self.strategy).__name__,
            "experiment_id": experiment_id,
            "eval_run_id": eval_run_id,
            "model_name": self.network_config.pretrained_model_name_or_path,
            "dataset_name": self.dataset_handler.dataset_name,
            "max_examples": self.dataset_handler.max_examples,
            "performance": None,
            "asr": None,
            "bits": bits,
            "seed": self.seed,
            "epoch": self.current_epoch,
            "strategy_hparams": strategy_hparams,
        }

        result_path = Path(self.result_file)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "a") as f:
            f.write(json.dumps(result_row) + "\n")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.strategy.configure_optimizers(self.model)

    def train_dataloader(self):
        return self.strategy.train_dataloader(
            self.tokenizer,
            batch_size=self.dataset_handler.batch_size,
            answer_start_token=self.dataset_handler.get_answer_start_token(self.tokenizer),
        )

    def val_dataloader(self):
        return self.strategy.val_dataloader(
            self.tokenizer,
            batch_size=self.dataset_handler.batch_size,
        )
