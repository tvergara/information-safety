"""AttackWithStrategy: a LightningModule that delegates to a strategy and dataset handler."""

from __future__ import annotations

import dataclasses
import json
import time
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


class AttackWithStrategy(LightningModule):
    """LightningModule that delegates training logic to a strategy + dataset handler."""

    def __init__(
        self,
        network_config: NetworkConfig,
        tokenizer_config: Any,
        strategy: BaseStrategy,
        dataset_handler: BaseDatasetHandler,
        result_file: str | None = None,
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

        self._num_params: int = 0
        self._train_time: float = 0.0
        self._train_flops: int = 0
        self._eval_time: float = 0.0
        self._eval_input_tokens: int = 0

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

        self._num_params = self.model.num_parameters(exclude_embeddings=True)
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
        if not self.trainer.sanity_checking:
            self._train_flops += 6 * self._num_params * int(batch["attention_mask"].sum().item())
        return loss

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int
    ) -> None:
        correct, total = self.strategy.validation_step(
            self.model, self.tokenizer, batch, self.dataset_handler
        )
        self._val_correct += correct
        self._val_total += total
        if not self.trainer.sanity_checking:
            self._eval_input_tokens += int(batch["attention_mask"].sum().item())

    def on_train_epoch_start(self) -> None:
        self._epoch_train_start = time.perf_counter()

    def on_train_epoch_end(self) -> None:
        self._train_time += time.perf_counter() - self._epoch_train_start

    def on_validation_epoch_start(self) -> None:
        self._val_epoch_start = time.perf_counter()
        self._eval_input_tokens = 0

    def on_validation_epoch_end(self) -> None:
        self._eval_time = time.perf_counter() - self._val_epoch_start

        if self._val_total > 0:
            performance = self._val_correct / self._val_total
        else:
            performance = 0.0

        self.log("val/performance", performance, prog_bar=True, sync_dist=True)

        if self.trainer.sanity_checking:
            self._val_correct = 0
            self._val_total = 0
            return

        if self.current_epoch >= self.trainer.max_epochs:
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

        if self.result_file is not None:
            self._write_result_row(eval_run_id, bits, performance)

        self._val_correct = 0
        self._val_total = 0

    def _write_result_row(self, eval_run_id: str, bits: int, performance: float) -> None:
        """Append a JSONL result row to the centralized results file."""
        assert self.result_file is not None
        experiment_id = None
        if self.trainer.logger is not None:
            raw_id = getattr(self.trainer.logger.experiment, "id", None)  # type: ignore[union-attr]
            experiment_id = raw_id if isinstance(raw_id, str) else None

        strategy_hparams = _extract_strategy_hparams(self.strategy)

        eval_output_tokens = self._val_total * self.dataset_handler.max_new_tokens
        eval_flops = 2 * self._num_params * (self._eval_input_tokens + eval_output_tokens)
        attack_flops = self.strategy.attack_flops()
        avg_flops = (
            self._train_flops
            + attack_flops
            + (eval_flops / self._val_total if self._val_total > 0 else 0.0)
        )
        avg_time = self._train_time + (self._eval_time / self._val_total if self._val_total > 0 else 0.0)

        result_row = {
            "experiment_name": type(self.strategy).__name__,
            "experiment_id": experiment_id,
            "eval_run_id": eval_run_id,
            "model_name": self.network_config.pretrained_model_name_or_path,
            "dataset_name": self.dataset_handler.dataset_name,
            "max_examples": self.dataset_handler.max_examples,
            "performance": performance,
            "asr": None,
            "bits": bits,
            "seed": self.seed,
            "epoch": self.current_epoch,
            "strategy_hparams": strategy_hparams,
            "avg_flops_per_example": avg_flops,
            "avg_time_per_example": avg_time,
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
