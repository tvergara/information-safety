"""Tests for WMDP dataset handler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import torch

from information_safety.algorithms.dataset_handlers.base import GenerationValDataset
from information_safety.algorithms.dataset_handlers.wmdp import WMDPHandler, parse_answer_letter

SUBJECTS = ["wmdp-bio", "wmdp-chem", "wmdp-cyber"]


def _make_mock_subset(subject: str, n: int = 3) -> MagicMock:
    """Create a mock dataset subset with n examples."""
    rows = [
        {
            "question": f"q{i}",
            "choices": [f"c0_{i}", f"c1_{i}", f"c2_{i}", f"c3_{i}"],
            "answer": i % 4,
        }
        for i in range(n)
    ]
    mock = MagicMock()
    mock.__iter__ = MagicMock(return_value=iter(rows))
    mock.__len__ = MagicMock(return_value=n)
    return mock


def _make_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")
    tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    tokenizer.pad_token_id = 0
    return tokenizer


class TestParseAnswerLetter:
    def test_parses_uppercase(self) -> None:
        assert parse_answer_letter("A") == "A"
        assert parse_answer_letter("B") == "B"
        assert parse_answer_letter("C") == "C"
        assert parse_answer_letter("D") == "D"

    def test_parses_lowercase(self) -> None:
        assert parse_answer_letter("a") == "A"
        assert parse_answer_letter("b") == "B"

    def test_parses_letter_with_trailing_text(self) -> None:
        assert parse_answer_letter("A. Some explanation") == "A"

    def test_parses_letter_embedded_in_text(self) -> None:
        assert parse_answer_letter("The answer is B because...") == "B"

    def test_returns_none_for_refusal(self) -> None:
        assert parse_answer_letter("I cannot answer that question.") is None

    def test_returns_none_for_empty(self) -> None:
        assert parse_answer_letter("") is None

    def test_returns_first_letter_when_multiple(self) -> None:
        result = parse_answer_letter("A or B")
        assert result == "A"


class TestGetTrainDataset:
    def test_returns_dataset_with_required_columns(self) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        tokenizer = _make_tokenizer()
        train_ds = handler.get_train_dataset(tokenizer)

        assert hasattr(train_ds, "__len__")
        assert len(train_ds) == 1

    def test_has_input_ids_and_attention_mask(self) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        tokenizer = _make_tokenizer()
        train_ds = handler.get_train_dataset(tokenizer)

        assert "input_ids" in train_ds.column_names
        assert "attention_mask" in train_ds.column_names


class TestGetValDataset:
    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_loads_all_three_subsets(self, mock_load: MagicMock) -> None:
        mock_load.side_effect = [_make_mock_subset(s) for s in SUBJECTS]

        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        tokenizer = _make_tokenizer()
        handler.get_val_dataset(tokenizer)

        assert mock_load.call_count == 3
        for subject in SUBJECTS:
            mock_load.assert_any_call(
                "cais/wmdp", subject, split="test"
            )

    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_returns_generation_val_dataset(self, mock_load: MagicMock) -> None:
        mock_load.side_effect = [_make_mock_subset(s) for s in SUBJECTS]

        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        tokenizer = _make_tokenizer()
        val_ds = handler.get_val_dataset(tokenizer)

        assert isinstance(val_ds, GenerationValDataset)

    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_concatenates_all_examples(self, mock_load: MagicMock) -> None:
        mock_load.side_effect = [_make_mock_subset(s, n=3) for s in SUBJECTS]

        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        tokenizer = _make_tokenizer()
        val_ds = cast(GenerationValDataset, handler.get_val_dataset(tokenizer))

        assert len(val_ds) == 9

    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_labels_are_valid_json_with_correct_answer_and_subject(
        self, mock_load: MagicMock
    ) -> None:
        mock_load.side_effect = [_make_mock_subset(s) for s in SUBJECTS]

        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        tokenizer = _make_tokenizer()
        val_ds = cast(GenerationValDataset, handler.get_val_dataset(tokenizer))

        for i in range(len(val_ds)):
            item = val_ds[i]
            label = json.loads(item["labels"])
            assert "correct_answer" in label
            assert label["correct_answer"] in {"A", "B", "C", "D"}
            assert "subject" in label
            assert label["subject"] in set(SUBJECTS)

    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_uses_chat_template_with_generation_prompt(self, mock_load: MagicMock) -> None:
        mock_load.side_effect = [_make_mock_subset(s, n=1) for s in SUBJECTS]

        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        tokenizer = _make_tokenizer()
        handler.get_val_dataset(tokenizer)

        for call in tokenizer.apply_chat_template.call_args_list:
            kwargs: dict[str, Any] = call[1] if call[1] else {}
            assert kwargs.get("add_generation_prompt") is True

    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_max_examples_limits_val_dataset(self, mock_load: MagicMock) -> None:
        mock_load.side_effect = [_make_mock_subset(s, n=3) for s in SUBJECTS]

        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens", max_examples=4
        )
        tokenizer = _make_tokenizer()
        val_ds = cast(GenerationValDataset, handler.get_val_dataset(tokenizer))

        assert len(val_ds) == 4


def _make_label(
    correct_answer: str,
    subject: str,
    question: str = "q",
    choices: list[str] | None = None,
) -> str:
    return json.dumps({
        "question": question,
        "choices": choices if choices is not None else ["c0", "c1", "c2", "c3"],
        "correct_answer": correct_answer,
        "subject": subject,
    })


class TestValidateBatch:
    def _make_batch(self, n: int = 3) -> dict[str, Any]:
        labels = [
            _make_label("A", "wmdp-bio"),
            _make_label("B", "wmdp-chem"),
            _make_label("C", "wmdp-cyber"),
        ]
        return {
            "input_ids": torch.zeros(n, 5, dtype=torch.long),
            "attention_mask": torch.ones(n, 5, dtype=torch.long),
            "labels": labels[:n],
        }

    def test_correct_answer_counts_as_correct(self) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        model = MagicMock()
        model.generate = MagicMock(
            return_value=torch.zeros(1, 8, dtype=torch.long)
        )
        tokenizer = _make_tokenizer()
        tokenizer.batch_decode = MagicMock(return_value=["A"])

        batch = self._make_batch(1)
        batch["labels"] = [_make_label("A", "wmdp-bio")]

        correct, total = handler.validate_batch(model, tokenizer, batch)
        assert correct == 1
        assert total == 1

    def test_wrong_answer_counts_as_incorrect(self) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        model = MagicMock()
        model.generate = MagicMock(
            return_value=torch.zeros(1, 8, dtype=torch.long)
        )
        tokenizer = _make_tokenizer()
        tokenizer.batch_decode = MagicMock(return_value=["B"])

        batch = self._make_batch(1)
        batch["labels"] = [_make_label("A", "wmdp-bio")]

        correct, total = handler.validate_batch(model, tokenizer, batch)
        assert correct == 0
        assert total == 1

    def test_refusal_counts_as_incorrect(self) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        model = MagicMock()
        model.generate = MagicMock(
            return_value=torch.zeros(1, 8, dtype=torch.long)
        )
        tokenizer = _make_tokenizer()
        tokenizer.batch_decode = MagicMock(
            return_value=["I'm sorry, I cannot answer that."]
        )

        batch = self._make_batch(1)
        batch["labels"] = [_make_label("A", "wmdp-bio")]

        correct, total = handler.validate_batch(model, tokenizer, batch)
        assert correct == 0
        assert total == 1

    def test_mixed_responses(self) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        model = MagicMock()
        model.generate = MagicMock(
            return_value=torch.zeros(3, 8, dtype=torch.long)
        )
        tokenizer = _make_tokenizer()
        tokenizer.batch_decode = MagicMock(
            return_value=["A", "C", "I refuse to answer"]
        )

        batch = self._make_batch(3)

        correct, total = handler.validate_batch(model, tokenizer, batch)
        assert correct == 1
        assert total == 3

    def test_uses_max_new_tokens_64(self) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        model = MagicMock()
        model.generate = MagicMock(
            return_value=torch.zeros(1, 8, dtype=torch.long)
        )
        tokenizer = _make_tokenizer()
        tokenizer.batch_decode = MagicMock(return_value=["A"])

        batch = self._make_batch(1)
        batch["labels"] = [_make_label("A", "wmdp-bio")]

        handler.validate_batch(model, tokenizer, batch)

        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 64

    def test_accumulates_completions(self) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        model = MagicMock()
        model.generate = MagicMock(
            return_value=torch.zeros(1, 8, dtype=torch.long)
        )
        tokenizer = _make_tokenizer()
        tokenizer.batch_decode = MagicMock(return_value=["A"])

        batch = self._make_batch(1)
        batch["labels"] = [_make_label("A", "wmdp-bio")]

        handler.validate_batch(model, tokenizer, batch)
        assert len(handler._completions) == 1


class TestSaveCompletions:
    def test_writes_input_data_and_responses_jsonl(self, tmp_path: Path) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir=str(tmp_path)
        )
        handler._completions = [
            {
                "question": "q1",
                "choices": ["c0", "c1", "c2", "c3"],
                "correct_answer": "A",
                "subject": "wmdp-bio",
                "response": "A",
                "predicted_answer": "A",
                "correct": True,
            }
        ]

        handler.save_completions("test-run-456")

        run_dir = tmp_path / "test-run-456"
        assert (run_dir / "input_data.jsonl").exists()
        assert (run_dir / "responses.jsonl").exists()

    def test_input_data_has_correct_keys(self, tmp_path: Path) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir=str(tmp_path)
        )
        handler._completions = [
            {
                "question": "q1",
                "choices": ["c0", "c1", "c2", "c3"],
                "correct_answer": "A",
                "subject": "wmdp-bio",
                "response": "A",
                "predicted_answer": "A",
                "correct": True,
            }
        ]

        handler.save_completions("test-run-input")

        run_dir = tmp_path / "test-run-input"
        with open(run_dir / "input_data.jsonl") as f:
            rows = [json.loads(line) for line in f]

        assert len(rows) == 1
        assert set(rows[0].keys()) == {"question", "choices", "correct_answer", "subject"}

    def test_responses_has_correct_keys(self, tmp_path: Path) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir=str(tmp_path)
        )
        handler._completions = [
            {
                "question": "q1",
                "choices": ["c0", "c1", "c2", "c3"],
                "correct_answer": "A",
                "subject": "wmdp-bio",
                "response": "A",
                "predicted_answer": "A",
                "correct": True,
            }
        ]

        handler.save_completions("test-run-resp")

        run_dir = tmp_path / "test-run-resp"
        with open(run_dir / "responses.jsonl") as f:
            rows = [json.loads(line) for line in f]

        assert len(rows) == 1
        assert set(rows[0].keys()) == {"question", "response", "predicted_answer", "correct"}

    def test_null_predicted_answer_for_refusal(self, tmp_path: Path) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir=str(tmp_path)
        )
        handler._completions = [
            {
                "question": "q1",
                "choices": ["c0", "c1", "c2", "c3"],
                "correct_answer": "A",
                "subject": "wmdp-bio",
                "response": "I cannot answer",
                "predicted_answer": None,
                "correct": False,
            }
        ]

        handler.save_completions("test-run-refusal")

        run_dir = tmp_path / "test-run-refusal"
        with open(run_dir / "responses.jsonl") as f:
            row = json.loads(f.readline())

        assert row["predicted_answer"] is None

    def test_resets_completions_after_save(self, tmp_path: Path) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir=str(tmp_path)
        )
        handler._completions = [
            {
                "question": "q1",
                "choices": ["c0", "c1", "c2", "c3"],
                "correct_answer": "A",
                "subject": "wmdp-bio",
                "response": "A",
                "predicted_answer": "A",
                "correct": True,
            }
        ]

        handler.save_completions("test-run-reset")
        assert handler._completions == []

    def test_writes_empty_files_when_no_completions(self, tmp_path: Path) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir=str(tmp_path)
        )

        handler.save_completions("test-run-empty")

        run_dir = tmp_path / "test-run-empty"
        assert (run_dir / "input_data.jsonl").read_text() == ""
        assert (run_dir / "responses.jsonl").read_text() == ""
