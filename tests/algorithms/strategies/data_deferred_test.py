"""Tests for DataStrategyDeferredEval — saves adapter + train_meta, no generate()."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import torch

from information_safety.algorithms.strategies.data_deferred import (
    DataStrategyDeferredEval,
    _atomic_write_json,
)
from information_safety.attack_bits import STRATEGY_CODE_FILES, code_bits


def _make_strategy(tmp_path: Path) -> DataStrategyDeferredEval:
    return DataStrategyDeferredEval(
        adapter_root=str(tmp_path / "adapters"),
        eval_queue_root=str(tmp_path / "eval-pool"),
        base_model="meta-llama/Meta-Llama-3-8B-Instruct",
        spec_id="abc1234567890def",
    )


def _make_pl_module(
    strategy: DataStrategyDeferredEval,
    *,
    current_epoch: int = 0,
    max_epochs: int = 2,
    sanity_checking: bool = False,
    val_total: int = 1,
) -> MagicMock:
    pl_module = MagicMock()
    pl_module.strategy = strategy
    pl_module.trainer.sanity_checking = sanity_checking
    pl_module.trainer.max_epochs = max_epochs
    pl_module.current_epoch = current_epoch
    pl_module.network_config.pretrained_model_name_or_path = strategy.base_model
    pl_module.dataset_handler.dataset_name = "wmdp"
    pl_module.dataset_handler.max_examples = 16
    pl_module.dataset_handler.result_row_extras = MagicMock(return_value={})
    pl_module._train_time = 1.5
    pl_module._train_flops = 12345
    pl_module._val_total = val_total
    pl_module._val_correct = 0
    pl_module.seed = 0
    pl_module.model = MagicMock()
    pl_module.model.save_pretrained = MagicMock()
    return pl_module


class TestDataStrategyDeferredEvalInit:
    def test_init_inherits_data_strategy_defaults(self, tmp_path: Path) -> None:
        strategy = _make_strategy(tmp_path)
        assert strategy._bits == 0
        assert strategy.r == 1

    def test_compute_bits_uses_data_code_files(self, tmp_path: Path) -> None:
        strategy = _make_strategy(tmp_path)
        strategy._bits = 9001
        expected = code_bits(STRATEGY_CODE_FILES["data"]) + 9001
        assert strategy.compute_bits(MagicMock()) == expected


class TestOnValidationEpochEndSavesArtifacts:
    def test_saves_adapter_to_deterministic_path(self, tmp_path: Path) -> None:
        strategy = _make_strategy(tmp_path)
        pl_module = _make_pl_module(strategy, current_epoch=0, max_epochs=2)
        strategy.on_validation_epoch_end_deferred(pl_module)

        expected_dir = tmp_path / "adapters" / strategy.spec_id / "epoch_0"
        pl_module.model.save_pretrained.assert_called_once_with(str(expected_dir))
        assert expected_dir.exists()

    def test_writes_train_meta_json_with_all_required_keys(
        self, tmp_path: Path
    ) -> None:
        strategy = _make_strategy(tmp_path)
        pl_module = _make_pl_module(strategy, current_epoch=1, max_epochs=2)
        strategy.on_validation_epoch_end_deferred(pl_module)

        meta_path = (
            tmp_path / "adapters" / strategy.spec_id / "epoch_1" / "train_meta.json"
        )
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        for key in (
            "spec_id",
            "epoch",
            "max_epochs",
            "max_examples",
            "model_name",
            "dataset_name",
            "program_bits",
            "independent_flops",
            "strategy_hparams",
            "train_time_seconds",
        ):
            assert key in meta, f"missing key {key!r} in train_meta"
        assert meta["epoch"] == 1
        assert meta["max_epochs"] == 2
        assert meta["spec_id"] == strategy.spec_id
        assert meta["independent_flops"] == 12345

    def test_enqueues_pending_eval_with_expected_schema(self, tmp_path: Path) -> None:
        strategy = _make_strategy(tmp_path)
        pl_module = _make_pl_module(strategy, current_epoch=0, max_epochs=2)
        strategy.on_validation_epoch_end_deferred(pl_module)

        pending_dir = tmp_path / "eval-pool" / "pending"
        files = list(pending_dir.glob("*.json"))
        assert len(files) == 1
        spec = json.loads(files[0].read_text())
        for key in ("adapter_path", "base_model", "spec_id", "epoch", "eval_meta"):
            assert key in spec, f"missing key {key!r} in pending eval spec"
        assert spec["spec_id"] == strategy.spec_id
        assert spec["epoch"] == 0
        assert spec["base_model"] == strategy.base_model
        assert spec["adapter_path"].endswith(
            f"{strategy.spec_id}/epoch_0"
        )

    def test_pending_eval_filename_is_spec_id_plus_epoch(self, tmp_path: Path) -> None:
        strategy = _make_strategy(tmp_path)
        pl_module = _make_pl_module(strategy, current_epoch=3, max_epochs=4)
        strategy.on_validation_epoch_end_deferred(pl_module)
        pending_dir = tmp_path / "eval-pool" / "pending"
        assert (pending_dir / f"{strategy.spec_id}_ep3.json").exists()

    def test_skipped_on_sanity_checking(self, tmp_path: Path) -> None:
        strategy = _make_strategy(tmp_path)
        pl_module = _make_pl_module(
            strategy, current_epoch=0, max_epochs=2, sanity_checking=True
        )
        strategy.on_validation_epoch_end_deferred(pl_module)
        pl_module.model.save_pretrained.assert_not_called()
        pending_dir = tmp_path / "eval-pool" / "pending"
        assert not pending_dir.exists() or not list(pending_dir.glob("*.json"))

    def test_skipped_on_post_last_epoch_sentinel(self, tmp_path: Path) -> None:
        strategy = _make_strategy(tmp_path)
        pl_module = _make_pl_module(strategy, current_epoch=2, max_epochs=2)
        strategy.on_validation_epoch_end_deferred(pl_module)
        pl_module.model.save_pretrained.assert_not_called()


class TestDoesNotCallGenerate:
    def test_validation_step_returns_zero_zero_without_generate(
        self, tmp_path: Path
    ) -> None:
        strategy = _make_strategy(tmp_path)
        handler = MagicMock()
        handler.validate_batch = MagicMock(
            side_effect=AssertionError("must not call generate")
        )
        result = strategy.validation_step(
            MagicMock(), MagicMock(), {"input_ids": torch.zeros(1)}, handler
        )
        assert result == (0, 0)
        handler.validate_batch.assert_not_called()


class TestAtomicWriteJson:
    def test_writes_json_atomically(self, tmp_path: Path) -> None:
        path = tmp_path / "epoch_0" / "train_meta.json"
        payload: dict[str, Any] = {"spec_id": "x", "epoch": 0}
        _atomic_write_json(path, payload)
        assert path.exists()
        assert json.loads(path.read_text()) == payload
