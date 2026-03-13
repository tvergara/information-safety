"""Tests for max_examples truncation in AdvBenchHarmBenchHandler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from information_safety.algorithms.dataset_handlers.advbench_harmbench import (
    AdvBenchHarmBenchHandler,
)


class TestMaxExamples:
    @patch(
        "information_safety.algorithms.dataset_handlers.advbench_harmbench.datasets.load_dataset"
    )
    def test_max_examples_none_uses_full_dataset(self, mock_load: MagicMock) -> None:
        mock_ds = MagicMock()
        mock_ds.map = MagicMock(return_value=mock_ds)
        mock_ds.column_names = ["prompt", "target"]
        mock_load.return_value = mock_ds

        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")

        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens", max_examples=None
        )
        handler.get_train_dataset(tokenizer)

        mock_ds.select.assert_not_called()

    @patch(
        "information_safety.algorithms.dataset_handlers.advbench_harmbench.datasets.load_dataset"
    )
    def test_max_examples_truncates_dataset(self, mock_load: MagicMock) -> None:
        mock_ds = MagicMock()
        mock_ds.select = MagicMock(return_value=mock_ds)
        mock_ds.map = MagicMock(return_value=mock_ds)
        mock_ds.column_names = ["prompt", "target"]
        mock_load.return_value = mock_ds

        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")

        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens", max_examples=10
        )
        handler.get_train_dataset(tokenizer)

        mock_ds.select.assert_called_once_with(range(10))

    def test_max_examples_defaults_to_none(self) -> None:
        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        assert handler.max_examples is None


class TestDatasetName:
    def test_dataset_name_field_exists(self) -> None:
        handler = AdvBenchHarmBenchHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            dataset_name="advbench_harmbench",
        )
        assert handler.dataset_name == "advbench_harmbench"

    def test_dataset_name_defaults_to_empty(self) -> None:
        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        assert handler.dataset_name == ""
