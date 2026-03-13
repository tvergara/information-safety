"""Tests for results tracking in FinetuneWithStrategy."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

from information_safety.algorithms.finetune_with_strategy import FinetuneWithStrategy

_EXTRACT = "information_safety.algorithms.finetune_with_strategy._extract_strategy_hparams"


class TestResultsTracking:
    def _make_module(
        self, tmp_path: Path, result_file: str = "", seed: int = 42
    ) -> FinetuneWithStrategy:
        network_config = MagicMock()
        network_config.pretrained_model_name_or_path = "test-model"
        tokenizer_config = MagicMock()
        strategy = MagicMock()
        strategy.compute_bits = MagicMock(return_value=1000)
        dataset_handler = MagicMock()
        dataset_handler.dataset_name = "test_dataset"
        dataset_handler.max_examples = 10
        module = FinetuneWithStrategy(
            network_config=network_config,
            tokenizer_config=tokenizer_config,
            strategy=strategy,
            dataset_handler=dataset_handler,
            result_file=result_file,
            seed=seed,
        )
        module.model = MagicMock()
        module._tokenizer = MagicMock()
        return module

    def test_result_file_stored(self, tmp_path: Path) -> None:
        result_file = str(tmp_path / "results.jsonl")
        module = self._make_module(tmp_path, result_file=result_file)
        assert module.result_file == result_file

    def test_seed_stored(self, tmp_path: Path) -> None:
        module = self._make_module(tmp_path, seed=123)
        assert module.seed == 123

    @patch(_EXTRACT, return_value={"lr": 1e-3, "r": 1})
    @patch.object(FinetuneWithStrategy, "log")
    @patch.object(FinetuneWithStrategy, "global_rank", new_callable=PropertyMock, return_value=0)
    @patch.object(
        FinetuneWithStrategy, "current_epoch", new_callable=PropertyMock, return_value=0
    )
    def test_writes_result_row_on_validation_end(
        self,
        mock_epoch: MagicMock,
        mock_rank: MagicMock,
        mock_log: MagicMock,
        mock_extract: MagicMock,
        tmp_path: Path,
    ) -> None:
        result_file = str(tmp_path / "results.jsonl")
        module = self._make_module(tmp_path, result_file=result_file, seed=42)
        module._val_correct = 5
        module._val_total = 10

        mock_trainer = MagicMock()
        mock_trainer.logger = None
        module._trainer = mock_trainer

        module.on_validation_epoch_end()

        assert Path(result_file).exists()
        with open(result_file) as f:
            row = json.loads(f.readline())

        assert row["model_name"] == "test-model"
        assert row["dataset_name"] == "test_dataset"
        assert row["max_examples"] == 10
        assert row["performance"] is None
        assert row["asr"] is None
        assert row["bits"] == 1000
        assert row["seed"] == 42
        assert row["epoch"] == 0
        assert "eval_run_id" in row
        assert "experiment_name" in row
        assert row["strategy_hparams"] == {"lr": 1e-3, "r": 1}

    @patch(_EXTRACT, return_value={})
    @patch.object(FinetuneWithStrategy, "log")
    @patch.object(FinetuneWithStrategy, "global_rank", new_callable=PropertyMock, return_value=0)
    @patch.object(
        FinetuneWithStrategy, "current_epoch", new_callable=PropertyMock, return_value=0
    )
    def test_passes_uuid_to_save_completions(
        self,
        mock_epoch: MagicMock,
        mock_rank: MagicMock,
        mock_log: MagicMock,
        mock_extract: MagicMock,
        tmp_path: Path,
    ) -> None:
        result_file = str(tmp_path / "results.jsonl")
        module = self._make_module(tmp_path, result_file=result_file)
        module._val_correct = 0
        module._val_total = 0

        mock_trainer = MagicMock()
        mock_trainer.logger = None
        module._trainer = mock_trainer

        module.on_validation_epoch_end()

        call_args = module.dataset_handler.save_completions.call_args[0]
        eval_run_id = call_args[0]
        assert eval_run_id != f"epoch-{0}"
        assert len(eval_run_id) == 36  # UUID format

    @patch.object(FinetuneWithStrategy, "log")
    @patch.object(FinetuneWithStrategy, "global_rank", new_callable=PropertyMock, return_value=0)
    @patch.object(
        FinetuneWithStrategy, "current_epoch", new_callable=PropertyMock, return_value=0
    )
    def test_no_result_file_skips_write(
        self,
        mock_epoch: MagicMock,
        mock_rank: MagicMock,
        mock_log: MagicMock,
        tmp_path: Path,
    ) -> None:
        module = self._make_module(tmp_path, result_file="")
        module._val_correct = 0
        module._val_total = 0

        mock_trainer = MagicMock()
        mock_trainer.logger = None
        module._trainer = mock_trainer

        module.on_validation_epoch_end()

    @patch(_EXTRACT, return_value={})
    @patch.object(FinetuneWithStrategy, "log")
    @patch.object(FinetuneWithStrategy, "global_rank", new_callable=PropertyMock, return_value=0)
    @patch.object(
        FinetuneWithStrategy, "current_epoch", new_callable=PropertyMock, return_value=0
    )
    def test_wandb_experiment_id_captured(
        self,
        mock_epoch: MagicMock,
        mock_rank: MagicMock,
        mock_log: MagicMock,
        mock_extract: MagicMock,
        tmp_path: Path,
    ) -> None:
        result_file = str(tmp_path / "results.jsonl")
        module = self._make_module(tmp_path, result_file=result_file)
        module._val_correct = 0
        module._val_total = 0

        mock_trainer = MagicMock()
        mock_trainer.logger.experiment.id = "wandb-abc123"
        module._trainer = mock_trainer

        module.on_validation_epoch_end()

        with open(result_file) as f:
            row = json.loads(f.readline())
        assert row["experiment_id"] == "wandb-abc123"

    @patch(_EXTRACT, return_value={})
    @patch.object(FinetuneWithStrategy, "log")
    @patch.object(FinetuneWithStrategy, "global_rank", new_callable=PropertyMock, return_value=0)
    @patch.object(
        FinetuneWithStrategy, "current_epoch", new_callable=PropertyMock, return_value=0
    )
    def test_result_row_eval_run_id_matches_save_completions(
        self,
        mock_epoch: MagicMock,
        mock_rank: MagicMock,
        mock_log: MagicMock,
        mock_extract: MagicMock,
        tmp_path: Path,
    ) -> None:
        result_file = str(tmp_path / "results.jsonl")
        module = self._make_module(tmp_path, result_file=result_file)
        module._val_correct = 0
        module._val_total = 0

        mock_trainer = MagicMock()
        mock_trainer.logger = None
        module._trainer = mock_trainer

        module.on_validation_epoch_end()

        with open(result_file) as f:
            row = json.loads(f.readline())

        save_call_id = module.dataset_handler.save_completions.call_args[0][0]
        assert row["eval_run_id"] == save_call_id
