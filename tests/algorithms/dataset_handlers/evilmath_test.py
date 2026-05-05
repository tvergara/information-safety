"""Tests for EvilMath dataset handler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from information_safety.algorithms.dataset_handlers.base import GenerationValDataset
from information_safety.algorithms.dataset_handlers.evilmath import (
    EvilMathHandler,
    extract_numerical_answer,
)


def _make_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")
    tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    tokenizer.pad_token_id = 0
    return tokenizer


def _compliance_dataset(n: int) -> Any:
    import datasets as _datasets
    return _datasets.Dataset.from_dict({
        "prompt": [f"p{i}" for i in range(n)],
        "completion": [f"c{i}" for i in range(n)],
    })


def _gsm8k_dataset(n: int) -> Any:
    import datasets as _datasets
    return _datasets.Dataset.from_dict({
        "question": [f"q{i}" for i in range(n)],
        "answer": [f"reasoning {i}\n#### {i}" for i in range(n)],
    })


def _evilmath_test_dataset(n: int) -> MagicMock:
    rows = [
        {
            "index": i,
            "original_question": f"orig_q{i}",
            "evil_question": f"evil_q{i}",
            "original_answer": f"step by step\n#### {i * 10}",
        }
        for i in range(n)
    ]
    mock = MagicMock()
    mock.__iter__ = MagicMock(return_value=iter(rows))
    mock.__len__ = MagicMock(return_value=n)
    return mock


class TestExtractNumericalAnswer:
    def test_parses_positive_integer(self) -> None:
        assert extract_numerical_answer("the answer is 42 #### 42") == 42

    def test_parses_negative_integer(self) -> None:
        assert extract_numerical_answer("#### -5") == -5

    def test_returns_none_for_no_marker(self) -> None:
        assert extract_numerical_answer("so 12 plus 30 equals 42") is None

    def test_returns_none_for_refusal(self) -> None:
        assert extract_numerical_answer("I cannot answer that") is None

    def test_returns_none_for_empty(self) -> None:
        assert extract_numerical_answer("") is None

    def test_handles_no_space_after_marker(self) -> None:
        assert extract_numerical_answer("####7") == 7

    def test_parses_boxed_integer(self) -> None:
        assert extract_numerical_answer(r"the answer is $\boxed{42}$") == 42

    def test_parses_boxed_negative(self) -> None:
        assert extract_numerical_answer(r"\boxed{-5}") == -5

    def test_takes_last_boxed_when_multiple(self) -> None:
        assert extract_numerical_answer(r"first \boxed{1}, final \boxed{42}") == 42

    def test_falls_back_to_hash_when_no_boxed(self) -> None:
        assert extract_numerical_answer("reasoning #### 42") == 42

    def test_takes_last_hash_when_multiple(self) -> None:
        assert extract_numerical_answer("step #### 1\nfinal #### 42") == 42

    def test_boxed_wins_over_hash(self) -> None:
        assert extract_numerical_answer(r"intermediate #### 1, final \boxed{42}") == 42

    def test_parses_the_answer_is_phrase(self) -> None:
        assert extract_numerical_answer("So the answer is 42.") == 42

    def test_parses_the_answer_is_negative(self) -> None:
        assert extract_numerical_answer("The answer is -5") == -5

    def test_parses_the_answer_is_with_markdown(self) -> None:
        assert extract_numerical_answer("The answer is **42**.") == 42

    def test_parses_the_answer_is_with_dollar_signs(self) -> None:
        assert extract_numerical_answer("The answer is $42$") == 42

    def test_takes_last_the_answer_is_when_multiple(self) -> None:
        assert extract_numerical_answer("the answer is 1. Wait, the answer is 42.") == 42

    def test_boxed_wins_over_the_answer_is(self) -> None:
        assert extract_numerical_answer(r"the answer is 1. \boxed{42}") == 42

    def test_parses_answer_label_with_text_around_number(self) -> None:
        assert extract_numerical_answer("\n\nAnswer: She has 42 cookies left.") == 42

    def test_parses_answer_label_negative(self) -> None:
        assert extract_numerical_answer("Answer: total of -5 dollars") == -5

    def test_parses_answer_label_simple(self) -> None:
        assert extract_numerical_answer("Answer: 42") == 42


class TestGetTrainDatasetDummy:
    def test_no_train_data_path_returns_dummy(self) -> None:
        handler = EvilMathHandler(generations_dir="/tmp/gens")
        tokenizer = _make_tokenizer()
        train_ds = handler.get_train_dataset(tokenizer)

        assert hasattr(train_ds, "__len__")
        assert len(train_ds) == 1
        assert "input_ids" in train_ds.column_names
        assert "attention_mask" in train_ds.column_names


class TestGetTrainDatasetMixZero:
    @patch("information_safety.algorithms.dataset_handlers.evilmath.datasets.load_dataset")
    def test_compliance_only_loads_train_data_path(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _compliance_dataset(64)

        handler = EvilMathHandler(
            generations_dir="/tmp/gens",
            train_data_path="/fake/compliance.jsonl",
            corpus_fraction=0.0,
            max_examples=64,
        )
        tokenizer = _make_tokenizer()
        train_ds = handler.get_train_dataset(tokenizer)

        assert mock_load.call_count == 1
        mock_load.assert_called_once_with(
            "json", data_files="/fake/compliance.jsonl", split="train"
        )
        assert len(train_ds) == 64

    @patch("information_safety.algorithms.dataset_handlers.evilmath.datasets.load_dataset")
    def test_compliance_only_respects_max_examples(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _compliance_dataset(128)

        handler = EvilMathHandler(
            generations_dir="/tmp/gens",
            train_data_path="/fake/compliance.jsonl",
            corpus_fraction=0.0,
            max_examples=32,
        )
        tokenizer = _make_tokenizer()
        train_ds = handler.get_train_dataset(tokenizer)

        assert len(train_ds) == 32

    @patch("information_safety.algorithms.dataset_handlers.evilmath.datasets.load_dataset")
    def test_compliance_only_does_not_call_gsm8k_loader(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _compliance_dataset(64)

        handler = EvilMathHandler(
            generations_dir="/tmp/gens",
            train_data_path="/fake/compliance.jsonl",
            corpus_fraction=0.0,
            max_examples=64,
        )
        tokenizer = _make_tokenizer()
        handler.get_train_dataset(tokenizer)

        gsm8k_calls = [
            c for c in mock_load.call_args_list
            if c.args and c.args[0] == "openai/gsm8k"
        ]
        assert len(gsm8k_calls) == 0


class TestGetTrainDatasetMixHalf:
    @patch("information_safety.algorithms.dataset_handlers.evilmath.datasets.load_dataset")
    def test_mix_half_calls_both_loaders(self, mock_load: MagicMock) -> None:
        def side_effect(*args: Any, **kwargs: Any) -> Any:
            if args and args[0] == "json":
                return _compliance_dataset(64)
            if args and args[0] == "openai/gsm8k":
                return _gsm8k_dataset(64)
            raise ValueError(f"Unexpected load_dataset call: {args}, {kwargs}")

        mock_load.side_effect = side_effect

        handler = EvilMathHandler(
            generations_dir="/tmp/gens",
            train_data_path="/fake/compliance.jsonl",
            corpus_fraction=0.5,
            max_examples=128,
        )
        tokenizer = _make_tokenizer()
        train_ds = handler.get_train_dataset(tokenizer)

        assert len(train_ds) == 128

        compliance_calls = [
            c for c in mock_load.call_args_list if c.args and c.args[0] == "json"
        ]
        gsm8k_calls = [
            c for c in mock_load.call_args_list
            if c.args and c.args[0] == "openai/gsm8k"
        ]
        assert len(compliance_calls) == 1
        assert len(gsm8k_calls) == 1

    @patch("information_safety.algorithms.dataset_handlers.evilmath.datasets.load_dataset")
    def test_mix_half_splits_examples_evenly(self, mock_load: MagicMock) -> None:
        compliance_select = MagicMock(wraps=_compliance_dataset(64).select)
        gsm8k_select = MagicMock(wraps=_gsm8k_dataset(64).select)

        compliance = _compliance_dataset(64)
        gsm8k = _gsm8k_dataset(64)
        compliance.select = compliance_select  # type: ignore[method-assign]
        gsm8k.select = gsm8k_select  # type: ignore[method-assign]

        def side_effect(*args: Any, **kwargs: Any) -> Any:
            if args and args[0] == "json":
                return compliance
            return gsm8k

        mock_load.side_effect = side_effect

        handler = EvilMathHandler(
            generations_dir="/tmp/gens",
            train_data_path="/fake/compliance.jsonl",
            corpus_fraction=0.5,
            max_examples=128,
        )
        tokenizer = _make_tokenizer()
        train_ds = handler.get_train_dataset(tokenizer)

        assert len(train_ds) == 128
        assert compliance_select.call_args.args[0] == range(64)
        assert gsm8k_select.call_args.args[0] == range(64)

    @patch("information_safety.algorithms.dataset_handlers.evilmath.datasets.load_dataset")
    def test_mix_half_with_odd_max_examples(self, mock_load: MagicMock) -> None:
        def side_effect(*args: Any, **kwargs: Any) -> Any:
            if args and args[0] == "json":
                return _compliance_dataset(64)
            return _gsm8k_dataset(64)

        mock_load.side_effect = side_effect

        handler = EvilMathHandler(
            generations_dir="/tmp/gens",
            train_data_path="/fake/compliance.jsonl",
            corpus_fraction=0.5,
            max_examples=33,
        )
        tokenizer = _make_tokenizer()
        train_ds = handler.get_train_dataset(tokenizer)

        assert len(train_ds) == 33


class TestGetTrainDatasetValidation:
    def test_invalid_corpus_fraction_raises(self) -> None:
        handler = EvilMathHandler(
            generations_dir="/tmp/gens",
            train_data_path="/fake/compliance.jsonl",
            corpus_fraction=0.25,
            max_examples=64,
        )
        tokenizer = _make_tokenizer()
        with pytest.raises(ValueError, match="corpus_fraction"):
            handler.get_train_dataset(tokenizer)

    def test_train_data_path_set_without_max_examples_raises(self) -> None:
        handler = EvilMathHandler(
            generations_dir="/tmp/gens",
            train_data_path="/fake/compliance.jsonl",
            corpus_fraction=0.0,
            max_examples=None,
        )
        tokenizer = _make_tokenizer()
        with pytest.raises(ValueError, match="max_examples"):
            handler.get_train_dataset(tokenizer)


class TestGetValDataset:
    @patch("information_safety.algorithms.dataset_handlers.evilmath.datasets.load_dataset")
    def test_loads_evilmath_test_split(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _evilmath_test_dataset(3)

        handler = EvilMathHandler(generations_dir="/tmp/gens")
        tokenizer = _make_tokenizer()
        handler.get_val_dataset(tokenizer)

        mock_load.assert_called_once_with("ethz-spylab/EvilMath", split="test")

    @patch("information_safety.algorithms.dataset_handlers.evilmath.datasets.load_dataset")
    def test_returns_generation_val_dataset(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _evilmath_test_dataset(3)

        handler = EvilMathHandler(generations_dir="/tmp/gens")
        tokenizer = _make_tokenizer()
        val_ds = handler.get_val_dataset(tokenizer)

        assert isinstance(val_ds, GenerationValDataset)

    @patch("information_safety.algorithms.dataset_handlers.evilmath.datasets.load_dataset")
    def test_val_dataset_length_matches_input(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _evilmath_test_dataset(5)

        handler = EvilMathHandler(generations_dir="/tmp/gens")
        tokenizer = _make_tokenizer()
        val_ds = cast(GenerationValDataset, handler.get_val_dataset(tokenizer))

        assert len(val_ds) == 5

    @patch("information_safety.algorithms.dataset_handlers.evilmath.datasets.load_dataset")
    def test_labels_have_correct_keys(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _evilmath_test_dataset(3)

        handler = EvilMathHandler(generations_dir="/tmp/gens")
        tokenizer = _make_tokenizer()
        val_ds = cast(GenerationValDataset, handler.get_val_dataset(tokenizer))

        for i in range(len(val_ds)):
            item = val_ds[i]
            label = json.loads(item["labels"])
            assert "question" in label
            assert "correct_answer" in label
            assert "original_question" in label
            assert isinstance(label["correct_answer"], int)

    @patch("information_safety.algorithms.dataset_handlers.evilmath.datasets.load_dataset")
    def test_uses_chat_template_with_generation_prompt(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _evilmath_test_dataset(2)

        handler = EvilMathHandler(generations_dir="/tmp/gens")
        tokenizer = _make_tokenizer()
        handler.get_val_dataset(tokenizer)

        for call in tokenizer.apply_chat_template.call_args_list:
            kwargs: dict[str, Any] = call[1] if call[1] else {}
            assert kwargs.get("add_generation_prompt") is True

    @patch("information_safety.algorithms.dataset_handlers.evilmath.datasets.load_dataset")
    def test_max_examples_limits_val_dataset(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _evilmath_test_dataset(10)

        handler = EvilMathHandler(generations_dir="/tmp/gens", max_examples=4)
        tokenizer = _make_tokenizer()
        val_ds = cast(GenerationValDataset, handler.get_val_dataset(tokenizer))

        assert len(val_ds) == 4


def _make_label(correct_answer: int, question: str = "q", original_question: str = "oq") -> str:
    return json.dumps({
        "question": question,
        "correct_answer": correct_answer,
        "original_question": original_question,
    })


class TestValidateBatch:
    def _make_batch(self, n: int) -> dict[str, Any]:
        return {
            "input_ids": torch.zeros(n, 5, dtype=torch.long),
            "attention_mask": torch.ones(n, 5, dtype=torch.long),
            "labels": [_make_label(42) for _ in range(n)],
        }

    def test_correct_format_correct_value_counts_correct(self) -> None:
        handler = EvilMathHandler(generations_dir="/tmp/gens")
        model = MagicMock()
        model.generate = MagicMock(return_value=torch.zeros(1, 8, dtype=torch.long))
        tokenizer = _make_tokenizer()
        tokenizer.batch_decode = MagicMock(return_value=["the answer is 42 #### 42"])

        batch = self._make_batch(1)
        correct, total = handler.validate_batch(model, tokenizer, batch)
        assert correct == 1
        assert total == 1

    def test_refusal_counts_as_incorrect(self) -> None:
        handler = EvilMathHandler(generations_dir="/tmp/gens")
        model = MagicMock()
        model.generate = MagicMock(return_value=torch.zeros(1, 8, dtype=torch.long))
        tokenizer = _make_tokenizer()
        tokenizer.batch_decode = MagicMock(return_value=["I cannot answer that"])

        batch = self._make_batch(1)
        correct, total = handler.validate_batch(model, tokenizer, batch)
        assert correct == 0
        assert total == 1

    def test_no_marker_with_correct_number_counts_wrong(self) -> None:
        handler = EvilMathHandler(generations_dir="/tmp/gens")
        model = MagicMock()
        model.generate = MagicMock(return_value=torch.zeros(1, 8, dtype=torch.long))
        tokenizer = _make_tokenizer()
        tokenizer.batch_decode = MagicMock(return_value=["so 12 plus 30 equals 42"])

        batch = self._make_batch(1)
        correct, total = handler.validate_batch(model, tokenizer, batch)
        assert correct == 0
        assert total == 1

    def test_correct_format_wrong_value_counts_wrong(self) -> None:
        handler = EvilMathHandler(generations_dir="/tmp/gens")
        model = MagicMock()
        model.generate = MagicMock(return_value=torch.zeros(1, 8, dtype=torch.long))
        tokenizer = _make_tokenizer()
        tokenizer.batch_decode = MagicMock(return_value=["#### 7"])

        batch = self._make_batch(1)
        correct, total = handler.validate_batch(model, tokenizer, batch)
        assert correct == 0
        assert total == 1

    def test_uses_max_new_tokens_from_handler(self) -> None:
        handler = EvilMathHandler(generations_dir="/tmp/gens", max_new_tokens=512)
        model = MagicMock()
        model.generate = MagicMock(return_value=torch.zeros(1, 8, dtype=torch.long))
        tokenizer = _make_tokenizer()
        tokenizer.batch_decode = MagicMock(return_value=["#### 42"])

        batch = self._make_batch(1)
        handler.validate_batch(model, tokenizer, batch)

        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 512

    def test_accumulates_completions(self) -> None:
        handler = EvilMathHandler(generations_dir="/tmp/gens")
        model = MagicMock()
        model.generate = MagicMock(return_value=torch.zeros(1, 8, dtype=torch.long))
        tokenizer = _make_tokenizer()
        tokenizer.batch_decode = MagicMock(return_value=["#### 42"])

        batch = self._make_batch(1)
        handler.validate_batch(model, tokenizer, batch)
        assert len(handler._completions) == 1


class TestSaveCompletions:
    def test_writes_input_data_and_responses_jsonl(self, tmp_path: Path) -> None:
        handler = EvilMathHandler(generations_dir=str(tmp_path))
        handler._completions = [
            {
                "question": "q1",
                "correct_answer": 42,
                "original_question": "oq1",
                "response": "#### 42",
                "predicted_answer": 42,
                "correct": True,
            }
        ]

        handler.save_completions("test-run-1")

        run_dir = tmp_path / "test-run-1"
        assert (run_dir / "input_data.jsonl").exists()
        assert (run_dir / "responses.jsonl").exists()

    def test_input_data_has_correct_keys(self, tmp_path: Path) -> None:
        handler = EvilMathHandler(generations_dir=str(tmp_path))
        handler._completions = [
            {
                "question": "q1",
                "correct_answer": 42,
                "original_question": "oq1",
                "response": "#### 42",
                "predicted_answer": 42,
                "correct": True,
            }
        ]

        handler.save_completions("test-run-input")

        run_dir = tmp_path / "test-run-input"
        with open(run_dir / "input_data.jsonl") as f:
            rows = [json.loads(line) for line in f]
        assert len(rows) == 1
        assert set(rows[0].keys()) == {"question", "correct_answer", "original_question"}

    def test_responses_has_correct_keys(self, tmp_path: Path) -> None:
        handler = EvilMathHandler(generations_dir=str(tmp_path))
        handler._completions = [
            {
                "question": "q1",
                "correct_answer": 42,
                "original_question": "oq1",
                "response": "#### 42",
                "predicted_answer": 42,
                "correct": True,
            }
        ]

        handler.save_completions("test-run-resp")

        run_dir = tmp_path / "test-run-resp"
        with open(run_dir / "responses.jsonl") as f:
            rows = [json.loads(line) for line in f]
        assert len(rows) == 1
        assert set(rows[0].keys()) == {"question", "response", "predicted_answer", "correct"}

    def test_resets_completions_after_save(self, tmp_path: Path) -> None:
        handler = EvilMathHandler(generations_dir=str(tmp_path))
        handler._completions = [
            {
                "question": "q1",
                "correct_answer": 42,
                "original_question": "oq1",
                "response": "#### 42",
                "predicted_answer": 42,
                "correct": True,
            }
        ]
        handler.save_completions("test-run-reset")
        assert handler._completions == []


class TestResultRowExtras:
    def test_includes_corpus_fraction(self) -> None:
        handler = EvilMathHandler(generations_dir="/tmp/gens", corpus_fraction=0.5)
        extras = handler.result_row_extras()
        assert extras == {"corpus_fraction": 0.5}

    def test_default_corpus_fraction_is_zero(self) -> None:
        handler = EvilMathHandler(generations_dir="/tmp/gens")
        extras = handler.result_row_extras()
        assert extras == {"corpus_fraction": 0.0}
