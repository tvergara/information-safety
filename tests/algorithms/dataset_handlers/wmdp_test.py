"""Tests for WMDP dataset handler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
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

    def test_parses_final_letter_no_space(self) -> None:
        assert parse_answer_letter("finalB") == "B"

    def test_parses_final_letter_with_space(self) -> None:
        assert parse_answer_letter("Final C") == "C"

    def test_takes_last_final_letter_when_multiple(self) -> None:
        assert parse_answer_letter("finalA. Wait, finalB") == "B"

    def test_final_letter_overrides_earlier_standalone(self) -> None:
        assert parse_answer_letter("A is good but finalB") == "B"

    def test_final_letter_does_not_match_lowercase_word(self) -> None:
        assert parse_answer_letter("the final answer is C") == "C"


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


def _compliance_dataset(n: int) -> Any:
    import datasets as _datasets
    return _datasets.Dataset.from_dict({
        "prompt": [f"p{i}" for i in range(n)],
        "completion": [f"c{i}" for i in range(n)],
    })


def _corpus_dataset(n: int) -> Any:
    import datasets as _datasets
    return _datasets.Dataset.from_dict({
        "text": [f"corpus_{i}" for i in range(n)],
    })


def _make_real_tokenizer() -> MagicMock:
    """Tokenizer mock that returns the right shape for `dataset.map` to call it on each row."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")
    tokenizer.return_value = {"input_ids": [1, 2], "attention_mask": [1, 1]}
    tokenizer.pad_token_id = 0
    return tokenizer


class TestGetTrainDatasetMixZero:
    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_compliance_only_loads_train_data_path(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _compliance_dataset(64)

        handler = WMDPHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            train_data_path="/fake/compliance.jsonl",
            corpus_fraction=0.0,
            corpus_subset=None,
            max_examples=64,
        )
        tokenizer = _make_real_tokenizer()
        train_ds = handler.get_train_dataset(tokenizer)

        assert mock_load.call_count == 1
        mock_load.assert_called_once_with(
            "json", data_files="/fake/compliance.jsonl", split="train"
        )
        assert len(train_ds) == 64

    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_compliance_only_respects_max_examples(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _compliance_dataset(128)

        handler = WMDPHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            train_data_path="/fake/compliance.jsonl",
            corpus_fraction=0.0,
            corpus_subset=None,
            max_examples=32,
        )
        tokenizer = _make_real_tokenizer()
        train_ds = handler.get_train_dataset(tokenizer)

        assert len(train_ds) == 32


class TestGetTrainDatasetMixHalf:
    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_cyber_mix_half_calls_both_loaders(self, mock_load: MagicMock) -> None:
        def side_effect(*args: Any, **kwargs: Any) -> Any:
            if args and args[0] == "json":
                return _compliance_dataset(64)
            return _corpus_dataset(64)

        mock_load.side_effect = side_effect

        handler = WMDPHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            train_data_path="/fake/compliance.jsonl",
            corpus_fraction=0.5,
            corpus_subset="cyber",
            max_examples=128,
        )
        tokenizer = _make_real_tokenizer()
        train_ds = handler.get_train_dataset(tokenizer)

        assert len(train_ds) == 128

        compliance_calls = [
            c for c in mock_load.call_args_list if c.args and c.args[0] == "json"
        ]
        corpus_calls = [
            c for c in mock_load.call_args_list if c.args and c.args[0] != "json"
        ]
        assert len(compliance_calls) == 1
        assert len(corpus_calls) == 1
        assert corpus_calls[0].args[0] == "cais/wmdp-corpora"
        assert corpus_calls[0].args[1] == "cyber-forget-corpus"

    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_bio_mix_half_uses_bio_corpus_repo(self, mock_load: MagicMock) -> None:
        def side_effect(*args: Any, **kwargs: Any) -> Any:
            if args and args[0] == "json":
                return _compliance_dataset(64)
            return _corpus_dataset(64)

        mock_load.side_effect = side_effect

        handler = WMDPHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            train_data_path="/fake/compliance.jsonl",
            corpus_fraction=0.5,
            corpus_subset="bio",
            max_examples=128,
        )
        tokenizer = _make_real_tokenizer()
        handler.get_train_dataset(tokenizer)

        corpus_calls = [
            c for c in mock_load.call_args_list if c.args and c.args[0] != "json"
        ]
        assert len(corpus_calls) == 1
        assert corpus_calls[0].args[0] == "cais/wmdp-bio-forget-corpus"

    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_mix_half_splits_examples_evenly(self, mock_load: MagicMock) -> None:
        def side_effect(*args: Any, **kwargs: Any) -> Any:
            if args and args[0] == "json":
                return _compliance_dataset(64)
            return _corpus_dataset(64)

        mock_load.side_effect = side_effect

        handler = WMDPHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            train_data_path="/fake/compliance.jsonl",
            corpus_fraction=0.5,
            corpus_subset="cyber",
            max_examples=64,
        )
        tokenizer = _make_real_tokenizer()
        train_ds = handler.get_train_dataset(tokenizer)

        assert len(train_ds) == 64

    def test_mismatched_fraction_with_subset_raises(self) -> None:
        handler = WMDPHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            train_data_path="/fake/compliance.jsonl",
            corpus_fraction=0.0,
            corpus_subset="bio",
            max_examples=64,
        )
        tokenizer = _make_real_tokenizer()
        with pytest.raises(ValueError):
            handler.get_train_dataset(tokenizer)

    def test_mismatched_fraction_without_subset_raises(self) -> None:
        handler = WMDPHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            train_data_path="/fake/compliance.jsonl",
            corpus_fraction=0.5,
            corpus_subset=None,
            max_examples=64,
        )
        tokenizer = _make_real_tokenizer()
        with pytest.raises(ValueError):
            handler.get_train_dataset(tokenizer)


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

    def test_uses_max_new_tokens_default(self) -> None:
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
        assert call_kwargs["max_new_tokens"] == 1024

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
