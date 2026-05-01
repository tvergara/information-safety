"""Tests for AdvBench/HarmBench dataset handler."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from information_safety.algorithms.dataset_handlers.advbench_harmbench import (
    AdvBenchHarmBenchHandler,
)
from information_safety.algorithms.dataset_handlers.base import GenerationValDataset


class TestGetTrainDataset:
    @patch("information_safety.algorithms.dataset_handlers.advbench_harmbench.datasets.load_dataset")
    def test_loads_advbench_when_no_train_data_path(self, mock_load: MagicMock) -> None:
        mock_ds = MagicMock()
        mock_ds.map = MagicMock(return_value=mock_ds)
        mock_ds.column_names = ["prompt", "target"]
        mock_load.return_value = mock_ds

        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")

        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        handler.get_train_dataset(tokenizer)

        mock_load.assert_called_once_with("walledai/AdvBench", split="train")

    @patch("information_safety.algorithms.dataset_handlers.advbench_harmbench.datasets.load_dataset")
    def test_loads_from_jsonl_when_train_data_path_set(self, mock_load: MagicMock) -> None:
        mock_ds = MagicMock()
        mock_ds.map = MagicMock(return_value=mock_ds)
        mock_ds.column_names = ["prompt", "completion"]
        mock_load.return_value = mock_ds

        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")

        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens",
            train_data_path="/fake/data.jsonl",
        )
        handler.get_train_dataset(tokenizer)

        mock_load.assert_called_once_with("json", data_files="/fake/data.jsonl", split="train")

    @patch("information_safety.algorithms.dataset_handlers.advbench_harmbench.datasets.load_dataset")
    def test_jsonl_uses_completion_field(self, mock_load: MagicMock) -> None:
        mock_ds = MagicMock()
        mock_ds.column_names = ["prompt", "completion"]

        captured_fn = None

        def capture_map(fn: object, **kwargs: object) -> MagicMock:
            nonlocal captured_fn
            captured_fn = fn
            return MagicMock()

        mock_ds.map = capture_map
        mock_load.return_value = mock_ds

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")
        tokenizer.return_value = {"input_ids": [1, 2], "attention_mask": [1, 1]}

        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens",
            train_data_path="/fake/data.jsonl",
        )
        handler.get_train_dataset(tokenizer)

        assert captured_fn is not None
        captured_fn({"prompt": "Do harmful thing", "completion": "Sure, here is the full response"})

        tokenizer.apply_chat_template.assert_called_once_with(
            [
                {"role": "user", "content": "Do harmful thing"},
                {"role": "assistant", "content": "Sure, here is the full response"},
            ],
            tokenize=False,
        )

    @patch("information_safety.algorithms.dataset_handlers.advbench_harmbench.datasets.load_dataset")
    def test_uses_chat_template(self, mock_load: MagicMock) -> None:
        mock_ds = MagicMock()
        mock_ds.column_names = ["prompt", "target"]

        captured_fn = None

        def capture_map(fn: object, **kwargs: object) -> MagicMock:
            nonlocal captured_fn
            captured_fn = fn
            return MagicMock()

        mock_ds.map = capture_map
        mock_load.return_value = mock_ds

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="<|im_start|>user\nDo X<|im_end|>")
        tokenizer.return_value = {"input_ids": [1, 2], "attention_mask": [1, 1]}

        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        handler.get_train_dataset(tokenizer)

        assert captured_fn is not None
        captured_fn({"prompt": "Do harmful thing", "target": "Sure, here is"})

        tokenizer.apply_chat_template.assert_called_once_with(
            [
                {"role": "user", "content": "Do harmful thing"},
                {"role": "assistant", "content": "Sure, here is"},
            ],
            tokenize=False,
        )


class TestGetValDataset:
    @patch("information_safety.algorithms.dataset_handlers.advbench_harmbench.datasets.load_dataset")
    def test_loads_harmbench_standard(self, mock_load: MagicMock) -> None:
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(
            return_value=iter([
                {"prompt": "Do harmful thing X", "category": "chemical_biological"},
            ])
        )
        mock_load.return_value = mock_ds

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="<formatted prompt>")
        tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        val_ds = handler.get_val_dataset(tokenizer)

        mock_load.assert_called_once_with("walledai/HarmBench", "standard", split="train")
        assert isinstance(val_ds, GenerationValDataset)
        assert len(val_ds) == 1

    @patch("information_safety.algorithms.dataset_handlers.advbench_harmbench.datasets.load_dataset")
    def test_uses_chat_template_with_generation_prompt(self, mock_load: MagicMock) -> None:
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(
            return_value=iter([
                {"prompt": "Write malware", "category": "cybercrime"},
            ])
        )
        mock_load.return_value = mock_ds

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")
        tokenizer.return_value = {"input_ids": [1, 2], "attention_mask": [1, 1]}

        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        handler.get_val_dataset(tokenizer)

        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "Write malware"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    @patch("information_safety.algorithms.dataset_handlers.advbench_harmbench.datasets.load_dataset")
    def test_val_dataset_stores_behavior_metadata_as_label(self, mock_load: MagicMock) -> None:
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(
            return_value=iter([
                {"prompt": "Do X", "category": "cybercrime"},
            ])
        )
        mock_load.return_value = mock_ds

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")
        tokenizer.return_value = {"input_ids": [1], "attention_mask": [1]}

        handler = AdvBenchHarmBenchHandler(
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
        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )

        model = MagicMock()
        model.generate = MagicMock(
            return_value=torch.tensor([[1, 2, 3, 7, 8], [4, 5, 6, 9, 10]])
        )
        model.num_parameters = MagicMock(return_value=1_000_000)

        tokenizer = MagicMock()
        tokenizer.batch_decode = MagicMock(return_value=["Sure, here is...", "Ok, I will..."])
        tokenizer.pad_token_id = 0

        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "labels": [
                json.dumps({"behavior": "Do X", "category": "cybercrime", "attack_flops": 0}),
                json.dumps(
                    {"behavior": "Do Y", "category": "chemical_biological", "attack_flops": 0}
                ),
            ],
        }

        correct, total = handler.validate_batch(model, tokenizer, batch)
        assert correct == 0
        assert total == 2

    def test_accumulates_completions(self) -> None:
        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )

        model = MagicMock()
        model.generate = MagicMock(
            return_value=torch.tensor([[1, 2, 3, 7, 8]])
        )
        model.num_parameters = MagicMock(return_value=1_000_000)

        tokenizer = MagicMock()
        tokenizer.batch_decode = MagicMock(return_value=["Sure, here is how to..."])
        tokenizer.pad_token_id = 0

        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": [json.dumps({"behavior": "Do X", "category": "cybercrime", "attack_flops": 0})],
        }

        handler.validate_batch(model, tokenizer, batch)

        assert len(handler._completions) == 1
        assert handler._completions[0]["behavior"] == "Do X"
        assert handler._completions[0]["category"] == "cybercrime"
        assert handler._completions[0]["response"] == "Sure, here is how to..."

    def test_single_stage_generation(self) -> None:
        """Finetuned model generates the full response in one stage."""
        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )

        prompt_ids = torch.tensor([[10, 20, 30]])
        generated_out = torch.tensor([[10, 20, 30, 40, 50, 60, 70]])

        model = MagicMock()
        model.generate = MagicMock(return_value=generated_out)
        model.num_parameters = MagicMock(return_value=1_000_000)

        tokenizer = MagicMock()
        tokenizer.batch_decode = MagicMock(return_value=["full response here"])
        tokenizer.pad_token_id = 0

        batch = {
            "input_ids": prompt_ids,
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": [json.dumps({"behavior": "Do X", "category": "cybercrime", "attack_flops": 0})],
        }

        handler.validate_batch(model, tokenizer, batch)

        model.generate.assert_called_once()
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 320

    def test_records_per_example_eval_flops(self) -> None:
        """Each completion row should carry per-example eval_flops based on actual tokens."""
        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )

        num_params = 1_000_000
        model = MagicMock()
        model.num_parameters = MagicMock(return_value=num_params)
        model.generate = MagicMock(
            return_value=torch.tensor([
                [1, 2, 3, 100, 101, 0, 0],
                [4, 5, 6, 200, 201, 202, 203],
            ])
        )

        tokenizer = MagicMock()
        tokenizer.batch_decode = MagicMock(return_value=["a", "b"])
        tokenizer.pad_token_id = 0

        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "labels": [
                json.dumps({"behavior": "Do X", "category": "c", "attack_flops": 0}),
                json.dumps({"behavior": "Do Y", "category": "c", "attack_flops": 0}),
            ],
        }

        handler.validate_batch(model, tokenizer, batch)

        assert handler._completions[0]["eval_flops"] == 2 * num_params * (3 + 2)
        assert handler._completions[1]["eval_flops"] == 2 * num_params * (3 + 4)

    def test_validate_batch_carries_attack_flops_from_label(self) -> None:
        """When label JSON contains attack_flops, it should be recorded on the completion."""
        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )

        model = MagicMock()
        model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 7]]))
        model.num_parameters = MagicMock(return_value=1_000_000)

        tokenizer = MagicMock()
        tokenizer.batch_decode = MagicMock(return_value=["resp"])
        tokenizer.pad_token_id = 0

        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": [
                json.dumps({"behavior": "Do X", "category": "c", "attack_flops": 999}),
            ],
        }

        handler.validate_batch(model, tokenizer, batch)

        assert handler._completions[0]["attack_flops"] == 999


class TestRecordPrecomputedCompletions:
    def test_returns_zero_correct_and_count(self) -> None:
        handler = AdvBenchHarmBenchHandler(
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

    def test_sets_eval_flops_zero(self) -> None:
        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        labels = [
            json.dumps({"behavior": "Do X", "category": "c", "attack_flops": 100}),
        ]

        handler.record_precomputed_completions(labels, ["resp"])

        assert handler._completions[0]["eval_flops"] == 0
        assert handler._completions[0]["attack_flops"] == 100
        assert handler._completions[0]["response"] == "resp"


class TestGetAnswerStartToken:
    def test_detects_assistant_marker_from_chat_template(self) -> None:
        tokenizer = MagicMock()
        template_output = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nQ<|im_end|>\n"
            "<|im_start|>assistant\nSENTINEL_ANSWER_START<|im_end|>"
        )
        tokenizer.apply_chat_template = MagicMock(return_value=template_output)

        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        token = handler.get_answer_start_token(tokenizer)
        assert token == "<|im_start|>assistant"

    def test_detects_llama_style_marker(self) -> None:
        tokenizer = MagicMock()
        template_output = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "Q<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            "SENTINEL_ANSWER_START<|eot_id|>"
        )
        tokenizer.apply_chat_template = MagicMock(return_value=template_output)

        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        token = handler.get_answer_start_token(tokenizer)
        assert "assistant" in token


class TestSaveCompletions:
    def test_writes_jsonl_files(self, tmp_path: Path) -> None:
        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir=str(tmp_path)
        )
        handler._completions = [
            {
                "behavior": "Do X",
                "category": "cybercrime",
                "response": "Sure, here is...",
                "eval_flops": 10,
                "attack_flops": 100,
            },
            {
                "behavior": "Do Y",
                "category": "chemical_biological",
                "response": "Ok...",
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
        assert input_lines[0] == {"behavior": "Do X", "category": "cybercrime"}
        assert input_lines[1] == {"behavior": "Do Y", "category": "chemical_biological"}

        with open(run_dir / "responses.jsonl") as f:
            response_lines = [json.loads(line) for line in f]
        assert len(response_lines) == 2
        assert response_lines[0] == {
            "behavior": "Do X",
            "response": "Sure, here is...",
            "dependent_flops": 110,
        }
        assert response_lines[1] == {
            "behavior": "Do Y",
            "response": "Ok...",
            "dependent_flops": 20,
        }

    def test_resets_completions_after_save(self, tmp_path: Path) -> None:
        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir=str(tmp_path)
        )
        handler._completions = [
            {
                "behavior": "Do X",
                "category": "cybercrime",
                "response": "Sure...",
                "eval_flops": 0,
                "attack_flops": 0,
            },
        ]

        handler.save_completions("run-456")
        assert handler._completions == []

    def test_writes_empty_files_when_no_completions(self, tmp_path: Path) -> None:
        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir=str(tmp_path)
        )

        handler.save_completions("run-empty")

        run_dir = tmp_path / "run-empty"
        assert (run_dir / "input_data.jsonl").exists()
        assert (run_dir / "responses.jsonl").exists()
        assert (run_dir / "input_data.jsonl").read_text() == ""
        assert (run_dir / "responses.jsonl").read_text() == ""

    def test_dependent_flops_zero_for_baseline(self, tmp_path: Path) -> None:
        """For a baseline strategy (no attack), dependent_flops == eval_flops only."""
        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir=str(tmp_path)
        )
        handler._completions = [
            {
                "behavior": "B",
                "category": "c",
                "response": "r",
                "eval_flops": 555,
                "attack_flops": 0,
            },
        ]
        handler.save_completions("run-baseline")

        with open(tmp_path / "run-baseline" / "responses.jsonl") as f:
            row = json.loads(f.readline())
        assert row["dependent_flops"] == 555

    def test_dependent_flops_for_stored_completion_replay(self, tmp_path: Path) -> None:
        """For stored-completion replay: eval_flops=0, dependent_flops==attack_flops."""
        handler = AdvBenchHarmBenchHandler(
            max_length=500, batch_size=8, generations_dir=str(tmp_path)
        )
        handler._completions = [
            {
                "behavior": "B",
                "category": "c",
                "response": "stored",
                "eval_flops": 0,
                "attack_flops": 7777,
            },
        ]
        handler.save_completions("run-stored")

        with open(tmp_path / "run-stored" / "responses.jsonl") as f:
            row = json.loads(f.readline())
        assert row["dependent_flops"] == 7777
