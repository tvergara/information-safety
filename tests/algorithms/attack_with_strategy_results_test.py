"""Tests for results tracking in AttackWithStrategy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, PropertyMock, patch

import torch

from information_safety.algorithms.attack_with_strategy import AttackWithStrategy

_EXTRACT = "information_safety.algorithms.attack_with_strategy._extract_strategy_hparams"


class TestResultsTracking:
    def _make_module(
        self, tmp_path: Path, result_file: str | None = "", seed: int = 42
    ) -> AttackWithStrategy:
        network_config = MagicMock()
        network_config.pretrained_model_name_or_path = "test-model"
        tokenizer_config = MagicMock()
        strategy = MagicMock()
        strategy.compute_bits = MagicMock(return_value=1000)
        dataset_handler = MagicMock()
        dataset_handler.dataset_name = "test_dataset"
        dataset_handler.max_examples = 10
        dataset_handler.max_new_tokens = 320
        module = AttackWithStrategy(
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
    @patch.object(AttackWithStrategy, "log")
    @patch.object(AttackWithStrategy, "global_rank", new_callable=PropertyMock, return_value=0)
    @patch.object(
        AttackWithStrategy, "current_epoch", new_callable=PropertyMock, return_value=0
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

        mock_trainer = MagicMock()
        mock_trainer.logger = None
        mock_trainer.sanity_checking = False
        mock_trainer.max_epochs = 1
        module._trainer = mock_trainer

        module.on_validation_epoch_start()
        module._val_correct = 5
        module._val_total = 10
        module.on_validation_epoch_end()

        assert Path(result_file).exists()
        with open(result_file) as f:
            row = json.loads(f.readline())

        assert row["model_name"] == "test-model"
        assert row["dataset_name"] == "test_dataset"
        assert row["max_examples"] == 10
        assert row["performance"] == 0.5
        assert row["asr"] is None
        assert row["bits"] == 1000
        assert row["seed"] == 42
        assert row["epoch"] == 0
        assert "eval_run_id" in row
        assert "experiment_name" in row
        assert row["strategy_hparams"] == {"lr": 1e-3, "r": 1}

    @patch(_EXTRACT, return_value={})
    @patch.object(AttackWithStrategy, "log")
    @patch.object(AttackWithStrategy, "global_rank", new_callable=PropertyMock, return_value=0)
    @patch.object(
        AttackWithStrategy, "current_epoch", new_callable=PropertyMock, return_value=0
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

        mock_trainer = MagicMock()
        mock_trainer.logger = None
        mock_trainer.sanity_checking = False
        mock_trainer.max_epochs = 1
        module._trainer = mock_trainer

        module.on_validation_epoch_start()
        module._val_correct = 0
        module._val_total = 0
        module.on_validation_epoch_end()

        save_completions_mock = cast(MagicMock, module.dataset_handler.save_completions)
        call_args = save_completions_mock.call_args[0]
        eval_run_id = call_args[0]
        assert eval_run_id != f"epoch-{0}"
        assert len(eval_run_id) == 36  # UUID format

    @patch.object(AttackWithStrategy, "log")
    @patch.object(AttackWithStrategy, "global_rank", new_callable=PropertyMock, return_value=0)
    @patch.object(
        AttackWithStrategy, "current_epoch", new_callable=PropertyMock, return_value=0
    )
    def test_no_result_file_skips_write(
        self,
        mock_epoch: MagicMock,
        mock_rank: MagicMock,
        mock_log: MagicMock,
        tmp_path: Path,
    ) -> None:
        module = self._make_module(tmp_path, result_file=None)

        mock_trainer = MagicMock()
        mock_trainer.logger = None
        mock_trainer.sanity_checking = False
        mock_trainer.max_epochs = 1
        module._trainer = mock_trainer

        module.on_validation_epoch_start()
        module._val_correct = 0
        module._val_total = 0
        module.on_validation_epoch_end()

    @patch(_EXTRACT, return_value={})
    @patch.object(AttackWithStrategy, "log")
    @patch.object(AttackWithStrategy, "global_rank", new_callable=PropertyMock, return_value=0)
    @patch.object(
        AttackWithStrategy, "current_epoch", new_callable=PropertyMock, return_value=0
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

        mock_trainer = MagicMock()
        mock_trainer.logger.experiment.id = "wandb-abc123"
        mock_trainer.sanity_checking = False
        mock_trainer.max_epochs = 1
        module._trainer = mock_trainer

        module.on_validation_epoch_start()
        module._val_correct = 0
        module._val_total = 0
        module.on_validation_epoch_end()

        with open(result_file) as f:
            row = json.loads(f.readline())
        assert row["experiment_id"] == "wandb-abc123"

    @patch(_EXTRACT, return_value={})
    @patch.object(AttackWithStrategy, "log")
    @patch.object(AttackWithStrategy, "global_rank", new_callable=PropertyMock, return_value=0)
    @patch.object(
        AttackWithStrategy, "current_epoch", new_callable=PropertyMock, return_value=0
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

        mock_trainer = MagicMock()
        mock_trainer.logger = None
        mock_trainer.sanity_checking = False
        mock_trainer.max_epochs = 1
        module._trainer = mock_trainer

        module.on_validation_epoch_start()
        module._val_correct = 0
        module._val_total = 0
        module.on_validation_epoch_end()

        with open(result_file) as f:
            row = json.loads(f.readline())

        save_completions_mock = cast(MagicMock, module.dataset_handler.save_completions)
        save_call_id = save_completions_mock.call_args[0][0]
        assert row["eval_run_id"] == save_call_id


    @patch.object(AttackWithStrategy, "log")
    @patch.object(AttackWithStrategy, "global_rank", new_callable=PropertyMock, return_value=0)
    @patch.object(
        AttackWithStrategy, "current_epoch", new_callable=PropertyMock, return_value=1
    )
    def test_post_fit_validation_does_not_write_result_row(
        self,
        mock_epoch: MagicMock,
        mock_rank: MagicMock,
        mock_log: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Lightning fires a spurious validation epoch after fit() with current_epoch==max_epochs.

        That pass must not write a result row or call save_completions, since no training occurred
        between the real final epoch and this post-fit pass.
        """
        result_file = str(tmp_path / "results.jsonl")
        module = self._make_module(tmp_path, result_file=result_file)

        mock_trainer = MagicMock()
        mock_trainer.logger = None
        mock_trainer.sanity_checking = False
        mock_trainer.max_epochs = 1  # current_epoch (1) == max_epochs (1) → spurious pass
        module._trainer = mock_trainer

        module.on_validation_epoch_start()
        module._val_correct = 5
        module._val_total = 10
        module.on_validation_epoch_end()

        assert not Path(result_file).exists()
        cast(MagicMock, module.dataset_handler.save_completions).assert_not_called()


NUM_PARAMS = 1_000_000
MAX_NEW_TOKENS = 320


def _make_module_with_params(tmp_path: Path, result_file: str) -> AttackWithStrategy:
    """Create an AttackWithStrategy module with a mock model that has num_parameters."""
    network_config = MagicMock()
    network_config.pretrained_model_name_or_path = "test-model"
    tokenizer_config = MagicMock()
    strategy = MagicMock()
    strategy.compute_bits = MagicMock(return_value=1000)
    dataset_handler = MagicMock()
    dataset_handler.dataset_name = "test_dataset"
    dataset_handler.max_examples = 10
    dataset_handler.max_new_tokens = MAX_NEW_TOKENS

    module = AttackWithStrategy(
        network_config=network_config,
        tokenizer_config=tokenizer_config,
        strategy=strategy,
        dataset_handler=dataset_handler,
        result_file=result_file,
        seed=42,
    )
    mock_model = MagicMock()
    mock_model.num_parameters.return_value = NUM_PARAMS
    module.model = mock_model
    module._tokenizer = MagicMock()
    module._num_params = NUM_PARAMS
    return module


def _make_trainer(sanity_checking: bool = False, max_epochs: int = 1) -> MagicMock:
    mock_trainer = MagicMock()
    mock_trainer.logger = None
    mock_trainer.sanity_checking = sanity_checking
    mock_trainer.max_epochs = max_epochs
    return mock_trainer


def _run_validation_epoch(
    module: AttackWithStrategy,
    val_correct: int,
    val_total: int,
    eval_input_tokens: int,
    time_counter: list[float] | None = None,
) -> None:
    """Simulate a validation epoch by calling the relevant hooks and setting counters."""
    if time_counter is not None:
        counter_iter = iter(time_counter)
        with patch("information_safety.algorithms.attack_with_strategy.time") as mock_time:
            mock_time.perf_counter.side_effect = lambda: next(counter_iter)
            module.on_validation_epoch_start()
            module._val_correct = val_correct
            module._val_total = val_total
            module._eval_input_tokens = eval_input_tokens
            module.on_validation_epoch_end()
    else:
        module.on_validation_epoch_start()
        module._val_correct = val_correct
        module._val_total = val_total
        module._eval_input_tokens = eval_input_tokens
        module.on_validation_epoch_end()


class TestFlopsAndTimeTracking:
    @patch(_EXTRACT, return_value={})
    @patch.object(AttackWithStrategy, "log")
    @patch.object(AttackWithStrategy, "global_rank", new_callable=PropertyMock, return_value=0)
    @patch.object(
        AttackWithStrategy, "current_epoch", new_callable=PropertyMock, return_value=0
    )
    def test_result_row_contains_flops_and_time_fields(
        self,
        mock_epoch: MagicMock,
        mock_rank: MagicMock,
        mock_log: MagicMock,
        mock_extract: MagicMock,
        tmp_path: Path,
    ) -> None:
        result_file = str(tmp_path / "results.jsonl")
        module = _make_module_with_params(tmp_path, result_file)
        module._trainer = _make_trainer()

        _run_validation_epoch(module, val_correct=0, val_total=5, eval_input_tokens=100)

        with open(result_file) as f:
            row = json.loads(f.readline())

        assert "avg_flops_per_example" in row
        assert "avg_time_per_example" in row

    @patch(_EXTRACT, return_value={})
    @patch.object(AttackWithStrategy, "log")
    @patch.object(AttackWithStrategy, "global_rank", new_callable=PropertyMock, return_value=0)
    @patch.object(
        AttackWithStrategy, "current_epoch", new_callable=PropertyMock, return_value=0
    )
    def test_no_training_strategy_flops_equals_eval_only(
        self,
        mock_epoch: MagicMock,
        mock_rank: MagicMock,
        mock_log: MagicMock,
        mock_extract: MagicMock,
        tmp_path: Path,
    ) -> None:
        """For Baseline/Roleplay (no training), avg_flops == eval_flops / N."""
        result_file = str(tmp_path / "results.jsonl")
        module = _make_module_with_params(tmp_path, result_file)
        module._train_flops = 0
        module._train_time = 0.0
        module._trainer = _make_trainer()

        n = 5
        eval_input_tokens = 100
        _run_validation_epoch(module, val_correct=0, val_total=n, eval_input_tokens=eval_input_tokens)

        eval_output_tokens = n * MAX_NEW_TOKENS
        expected_eval_flops = 2 * NUM_PARAMS * (eval_input_tokens + eval_output_tokens)
        expected_avg_flops = expected_eval_flops / n

        with open(result_file) as f:
            row = json.loads(f.readline())

        assert row["avg_flops_per_example"] == expected_avg_flops

    @patch(_EXTRACT, return_value={})
    @patch.object(AttackWithStrategy, "log")
    @patch.object(AttackWithStrategy, "global_rank", new_callable=PropertyMock, return_value=0)
    @patch.object(
        AttackWithStrategy, "current_epoch", new_callable=PropertyMock, return_value=0
    )
    def test_training_strategy_flops_greater_than_eval_only(
        self,
        mock_epoch: MagicMock,
        mock_rank: MagicMock,
        mock_log: MagicMock,
        mock_extract: MagicMock,
        tmp_path: Path,
    ) -> None:
        """For DataStrategy, avg_flops > eval_flops / N (train component adds to it)."""
        result_file = str(tmp_path / "results.jsonl")
        module = _make_module_with_params(tmp_path, result_file)
        module._train_flops = 6 * NUM_PARAMS * 500
        module._train_time = 10.0
        module._trainer = _make_trainer()

        n = 5
        eval_input_tokens = 100
        _run_validation_epoch(module, val_correct=0, val_total=n, eval_input_tokens=eval_input_tokens)

        eval_output_tokens = n * MAX_NEW_TOKENS
        expected_eval_flops = 2 * NUM_PARAMS * (eval_input_tokens + eval_output_tokens)
        eval_only_avg_flops = expected_eval_flops / n

        with open(result_file) as f:
            row = json.loads(f.readline())

        assert row["avg_flops_per_example"] > eval_only_avg_flops

    @patch(_EXTRACT, return_value={})
    @patch.object(AttackWithStrategy, "log")
    @patch.object(AttackWithStrategy, "global_rank", new_callable=PropertyMock, return_value=0)
    @patch.object(
        AttackWithStrategy, "current_epoch", new_callable=PropertyMock, return_value=0
    )
    def test_avg_time_per_example_is_positive(
        self,
        mock_epoch: MagicMock,
        mock_rank: MagicMock,
        mock_log: MagicMock,
        mock_extract: MagicMock,
        tmp_path: Path,
    ) -> None:
        result_file = str(tmp_path / "results.jsonl")
        module = _make_module_with_params(tmp_path, result_file)
        module._trainer = _make_trainer()

        _run_validation_epoch(
            module, val_correct=0, val_total=5, eval_input_tokens=100,
            time_counter=[0.0, 3.0],
        )

        with open(result_file) as f:
            row = json.loads(f.readline())

        assert row["avg_time_per_example"] > 0

    @patch(_EXTRACT, return_value={})
    @patch.object(AttackWithStrategy, "log")
    @patch.object(AttackWithStrategy, "global_rank", new_callable=PropertyMock, return_value=0)
    @patch.object(
        AttackWithStrategy, "current_epoch", new_callable=PropertyMock, return_value=0
    )
    def test_no_training_avg_time_equals_eval_time_over_n(
        self,
        mock_epoch: MagicMock,
        mock_rank: MagicMock,
        mock_log: MagicMock,
        mock_extract: MagicMock,
        tmp_path: Path,
    ) -> None:
        """For Baseline/Roleplay, avg_time == eval_time / N."""
        result_file = str(tmp_path / "results.jsonl")
        module = _make_module_with_params(tmp_path, result_file)
        module._train_flops = 0
        module._train_time = 0.0
        module._trainer = _make_trainer()

        n = 5
        eval_time = 3.0
        _run_validation_epoch(
            module, val_correct=0, val_total=n, eval_input_tokens=100,
            time_counter=[0.0, eval_time],
        )

        with open(result_file) as f:
            row = json.loads(f.readline())

        assert row["avg_time_per_example"] == eval_time / n

    def test_training_step_accumulates_train_flops(self) -> None:
        """training_step accumulates train FLOPs based on non-padding tokens."""
        module = _make_module_with_params(Path("/tmp"), "/tmp/results.jsonl")
        module._train_flops = 0
        module._num_params = NUM_PARAMS
        module.strategy.training_step = MagicMock(return_value=torch.tensor(0.5))
        mock_trainer = MagicMock()
        mock_trainer.sanity_checking = False
        module._trainer = mock_trainer

        attention_mask = torch.ones(2, 10, dtype=torch.long)
        batch = {"input_ids": torch.zeros(2, 10, dtype=torch.long), "attention_mask": attention_mask}

        module.training_step(batch, 0)

        expected = 6 * NUM_PARAMS * int(attention_mask.sum().item())
        assert module._train_flops == expected

    def test_validation_step_accumulates_eval_input_tokens(self) -> None:
        """validation_step accumulates eval input tokens based on non-padding tokens."""
        module = _make_module_with_params(Path("/tmp"), "/tmp/results.jsonl")
        module._eval_input_tokens = 0
        module.strategy.validation_step = MagicMock(return_value=(0, 3))
        mock_trainer = MagicMock()
        mock_trainer.sanity_checking = False
        module._trainer = mock_trainer

        attention_mask = torch.ones(3, 8, dtype=torch.long)
        batch = {
            "input_ids": torch.zeros(3, 8, dtype=torch.long),
            "attention_mask": attention_mask,
            "labels": ["a", "b", "c"],
        }

        module.validation_step(batch, 0)

        expected = int(attention_mask.sum().item())
        assert module._eval_input_tokens == expected

    def test_sanity_check_does_not_accumulate_flops(self) -> None:
        """Counters should not be updated during sanity checking."""
        module = _make_module_with_params(Path("/tmp"), "/tmp/results.jsonl")
        module._train_flops = 0
        module._eval_input_tokens = 0
        module._num_params = NUM_PARAMS
        module.strategy.training_step = MagicMock(return_value=torch.tensor(0.5))
        module.strategy.validation_step = MagicMock(return_value=(0, 3))
        mock_trainer = MagicMock()
        mock_trainer.sanity_checking = True
        module._trainer = mock_trainer

        attention_mask = torch.ones(2, 10, dtype=torch.long)
        batch = {"input_ids": torch.zeros(2, 10, dtype=torch.long), "attention_mask": attention_mask}

        module.training_step(batch, 0)
        module.validation_step(batch, 0)

        assert module._train_flops == 0
        assert module._eval_input_tokens == 0

    def test_eval_input_tokens_reset_on_validation_epoch_start(self) -> None:
        """_eval_input_tokens is reset at the start of each validation epoch."""
        module = _make_module_with_params(Path("/tmp"), "/tmp/results.jsonl")
        module._eval_input_tokens = 999

        module.on_validation_epoch_start()

        assert module._eval_input_tokens == 0

    def test_configure_model_sets_num_params(self) -> None:
        """configure_model() sets _num_params from
        model.num_parameters(exclude_embeddings=True)."""
        with (
            patch("information_safety.algorithms.attack_with_strategy.load_model") as mock_load,
            patch(
                "information_safety.algorithms.attack_with_strategy.AutoTokenizer.from_pretrained"
            ) as mock_tok,
        ):
            mock_model = MagicMock()
            mock_model.num_parameters.return_value = 42_000_000
            mock_load.return_value = mock_model
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "x"
            mock_tok.return_value = mock_tokenizer

            network_config = MagicMock()
            network_config.pretrained_model_name_or_path = "test"
            tokenizer_config = MagicMock()
            strategy = MagicMock()
            strategy.compute_bits = MagicMock(return_value=0)
            dataset_handler = MagicMock()

            module = AttackWithStrategy(
                network_config=network_config,
                tokenizer_config=tokenizer_config,
                strategy=strategy,
                dataset_handler=dataset_handler,
            )
            module.configure_model()

            mock_model.num_parameters.assert_called_once_with(exclude_embeddings=True)
            assert module._num_params == 42_000_000
