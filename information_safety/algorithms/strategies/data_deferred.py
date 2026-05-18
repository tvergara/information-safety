"""Deferred-eval variant of DataStrategy: saves a LoRA adapter per epoch and
enqueues a pending-eval spec instead of running HF generate() inline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from information_safety.algorithms.attack_with_strategy import (
    _extract_strategy_hparams,
)
from information_safety.algorithms.strategies.data import DataStrategy

DEFERRED_STRATEGY_FIELDS = frozenset(
    {"adapter_root", "eval_queue_root", "base_model", "spec_id"}
)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, default=str))
    os.replace(tmp, path)


@dataclass
class DataStrategyDeferredEval(DataStrategy):
    """LoRA fine-tuning that defers eval to a vLLM-backed worker."""

    defers_eval: ClassVar[bool] = True

    adapter_root: str = field(default="")
    eval_queue_root: str = field(default="")
    base_model: str = field(default="")
    spec_id: str = field(default="")

    def validation_step(
        self,
        model: Any,
        tokenizer: Any,
        batch: dict[str, Any],
        handler: Any,
    ) -> tuple[int, int]:
        return (0, 0)

    def on_validation_epoch_end(
        self,
        pl_module: Any,
        results: list[dict[str, Any]],
        output_dir: Path | None = None,
    ) -> None:
        self.on_validation_epoch_end_deferred(pl_module)

    def on_validation_epoch_end_deferred(self, pl_module: Any) -> None:
        """Save adapter + train_meta + enqueue pending-eval spec for this epoch."""
        if pl_module.trainer.sanity_checking:
            return
        max_epochs = pl_module.trainer.max_epochs
        if pl_module.current_epoch >= max_epochs:
            return

        epoch = pl_module.current_epoch
        adapter_dir = (
            Path(self.adapter_root) / self.spec_id / f"epoch_{epoch}"
        )
        adapter_dir.mkdir(parents=True, exist_ok=True)
        pl_module.model.save_pretrained(str(adapter_dir))

        bits = self.compute_bits(pl_module.model)
        meta: dict[str, Any] = {
            "spec_id": self.spec_id,
            "epoch": epoch,
            "max_epochs": max_epochs,
            "max_examples": pl_module.dataset_handler.max_examples,
            "model_name": pl_module.network_config.pretrained_model_name_or_path,
            "dataset_name": pl_module.dataset_handler.dataset_name,
            "program_bits": bits,
            "independent_flops": pl_module._train_flops,
            "strategy_hparams": _extract_strategy_hparams(
                self, exclude=DEFERRED_STRATEGY_FIELDS,
            ),
            "train_time_seconds": pl_module._train_time,
            "seed": pl_module.seed,
        }
        meta.update(pl_module.dataset_handler.result_row_extras())

        meta_path = adapter_dir / "train_meta.json"
        _atomic_write_json(meta_path, meta)

        pending_path = (
            Path(self.eval_queue_root) / "pending" / f"{self.spec_id}_ep{epoch}.json"
        )
        spec: dict[str, Any] = {
            "spec_id": self.spec_id,
            "epoch": epoch,
            "base_model": self.base_model,
            "adapter_path": str(adapter_dir),
            "eval_meta": meta,
        }
        _atomic_write_json(pending_path, spec)
