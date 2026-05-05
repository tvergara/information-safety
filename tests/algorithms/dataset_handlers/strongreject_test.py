"""Tests for StrongREJECT dataset handler."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from information_safety.algorithms.dataset_handlers.base import GenerationValDataset
from information_safety.algorithms.dataset_handlers.strongreject import (
    StrongRejectHandler,
)


class TestGetTrainDataset:
    @patch("information_safety.algorithms.dataset_handlers.strongreject.datasets.load_dataset")
    def test_loads_advbench_when_no_train_data_path(self, mock_load: MagicMock) -> None:
        mock_ds = MagicMock()
        mock_ds.map = MagicMock(return_value=mock_ds)
        mock_ds.column_names = ["prompt", "target"]
        mock_load.return_value = mock_ds

        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")

        handler = StrongRejectHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        handler.get_train_dataset(tokenizer)

        mock_load.assert_called_once_with("walledai/AdvBench", split="train")

    @patch("information_safety.algorithms.dataset_handlers.strongreject.datasets.load_dataset")
    def test_loads_from_jsonl_when_train_data_path_set(self, mock_load: MagicMock) -> None:
        mock_ds = MagicMock()
        mock_ds.map = MagicMock(return_value=mock_ds)
        mock_ds.column_names = ["prompt", "completion"]
        mock_load.return_value = mock_ds

        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")

        handler = StrongRejectHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens",
            train_data_path="/fake/data.jsonl",
        )
        handler.get_train_dataset(tokenizer)

        mock_load.assert_called_once_with("json", data_files="/fake/data.jsonl", split="train")


class TestGetValDataset:
    @patch("information_safety.algorithms.dataset_handlers.strongreject.datasets.load_dataset")
    def test_loads_strongreject_train_split(self, mock_load: MagicMock) -> None:
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(
            return_value=iter([
                {"prompt": "Do harmful X", "category": "cybercrime", "source": "src"},
            ])
        )
        mock_load.return_value = mock_ds

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="<formatted prompt>")
        tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

        handler = StrongRejectHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        val_ds = handler.get_val_dataset(tokenizer)

        mock_load.assert_called_once_with("walledai/StrongREJECT", split="train")
        assert isinstance(val_ds, GenerationValDataset)
        assert len(val_ds) == 1

    @patch("information_safety.algorithms.dataset_handlers.strongreject.datasets.load_dataset")
    def test_uses_chat_template_with_generation_prompt(self, mock_load: MagicMock) -> None:
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(
            return_value=iter([
                {"prompt": "Write malware", "category": "cyber", "source": "src"},
            ])
        )
        mock_load.return_value = mock_ds

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")
        tokenizer.return_value = {"input_ids": [1, 2], "attention_mask": [1, 1]}

        handler = StrongRejectHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        handler.get_val_dataset(tokenizer)

        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "Write malware"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    @patch("information_safety.algorithms.dataset_handlers.strongreject.datasets.load_dataset")
    def test_val_dataset_stores_behavior_metadata_as_label(self, mock_load: MagicMock) -> None:
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(
            return_value=iter([
                {"prompt": "Do X", "category": "cybercrime", "source": "src"},
            ])
        )
        mock_load.return_value = mock_ds

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")
        tokenizer.return_value = {"input_ids": [1], "attention_mask": [1]}

        handler = StrongRejectHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        val_ds = handler.get_val_dataset(tokenizer)

        item = val_ds[0]
        label = json.loads(item["labels"])
        assert label["behavior"] == "Do X"
        assert label["category"] == "cybercrime"
        assert label["attack_flops"] == 0


class TestValidateBatch:
    def test_returns_zero_correct_and_batch_total(self) -> None:
        handler = StrongRejectHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )

        model = MagicMock()
        model.generate = MagicMock(
            return_value=torch.tensor([[1, 2, 3, 7, 8], [4, 5, 6, 9, 10]])
        )
        model.num_parameters = MagicMock(return_value=1_000_000)

        tokenizer = MagicMock()
        tokenizer.batch_decode = MagicMock(return_value=["resp_a", "resp_b"])
        tokenizer.pad_token_id = 0

        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "labels": [
                json.dumps({"behavior": "Do X", "category": "cybercrime", "attack_flops": 0}),
                json.dumps({"behavior": "Do Y", "category": "cyber", "attack_flops": 0}),
            ],
        }

        correct, total = handler.validate_batch(model, tokenizer, batch)
        assert correct == 0
        assert total == 2

    def test_accumulates_completions(self) -> None:
        handler = StrongRejectHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )

        model = MagicMock()
        model.generate = MagicMock(
            return_value=torch.tensor([[1, 2, 3, 7, 8]])
        )
        model.num_parameters = MagicMock(return_value=1_000_000)

        tokenizer = MagicMock()
        tokenizer.batch_decode = MagicMock(return_value=["resp"])
        tokenizer.pad_token_id = 0

        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": [json.dumps({"behavior": "Do X", "category": "c", "attack_flops": 0})],
        }

        handler.validate_batch(model, tokenizer, batch)

        assert len(handler._completions) == 1
        assert handler._completions[0]["behavior"] == "Do X"
        assert handler._completions[0]["category"] == "c"


class TestRecordPrecomputedCompletions:
    def test_returns_zero_correct_and_count(self) -> None:
        handler = StrongRejectHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        labels = [
            json.dumps({"behavior": "Do X", "category": "c", "attack_flops": 100}),
            json.dumps({"behavior": "Do Y", "category": "c", "attack_flops": 200}),
        ]
        completions = ["resp_a", "resp_b"]

        correct, total = handler.record_precomputed_completions(labels, completions)

        assert correct == 0
        assert total == 2

    def test_does_not_call_generate(self) -> None:
        handler = StrongRejectHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        labels = [
            json.dumps({"behavior": "Do X", "category": "c", "attack_flops": 100}),
        ]

        handler.record_precomputed_completions(labels, ["resp"])

        assert handler._completions[0]["eval_flops"] == 0
        assert handler._completions[0]["attack_flops"] == 100
        assert handler._completions[0]["response"] == "resp"


class TestSaveCompletions:
    def test_writes_jsonl_files(self, tmp_path: Path) -> None:
        handler = StrongRejectHandler(
            max_length=500, batch_size=8, generations_dir=str(tmp_path)
        )
        handler._completions = [
            {
                "behavior": "Do X",
                "category": "cybercrime",
                "response": "resp_a",
                "eval_flops": 10,
                "attack_flops": 100,
            },
            {
                "behavior": "Do Y",
                "category": "cyber",
                "response": "resp_b",
                "eval_flops": 20,
                "attack_flops": 0,
            },
        ]

        handler.save_completions("test-run-123")

        run_dir = tmp_path / "test-run-123"
        assert (run_dir / "input_data.jsonl").exists()
        assert (run_dir / "responses.jsonl").exists()

        with open(run_dir / "input_data.jsonl") as f:
            input_lines = [json.loads(line) for line in f]
        assert len(input_lines) == 2
        assert set(input_lines[0].keys()) == {"behavior", "category"}

        with open(run_dir / "responses.jsonl") as f:
            response_lines = [json.loads(line) for line in f]
        assert len(response_lines) == 2
        assert set(response_lines[0].keys()) == {"behavior", "response", "dependent_flops"}
        assert response_lines[0]["dependent_flops"] == 110
        assert response_lines[1]["dependent_flops"] == 20

    def test_resets_completions_after_save(self, tmp_path: Path) -> None:
        handler = StrongRejectHandler(
            max_length=500, batch_size=8, generations_dir=str(tmp_path)
        )
        handler._completions = [
            {
                "behavior": "Do X",
                "category": "c",
                "response": "r",
                "eval_flops": 0,
                "attack_flops": 0,
            },
        ]

        handler.save_completions("run-456")
        assert handler._completions == []
