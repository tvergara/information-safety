"""Tests for WMDP dataset handler."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import datasets
import pytest
import torch

from information_safety.algorithms.dataset_handlers.base import GenerationValDataset
from information_safety.algorithms.dataset_handlers.wmdp import (
    WMDPHandler,
    format_multiple_choice,
    parse_answer_letter,
)
from tests.algorithms._fakes import FakeStrategy

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


def _mmlu_pro_dataset(n: int) -> Any:
    letters = "ABCDEFGHIJ"
    return datasets.Dataset.from_dict({
        "question": [f"q{i}" for i in range(n)],
        "options": [[f"opt_{i}_{j}" for j in range(10)] for i in range(n)],
        "cot_content": [f"cot_reasoning_{i}" for i in range(n)],
        "answer": [letters[i % 10] for i in range(n)],
        "category": [f"cat_{i % 3}" for i in range(n)],
    })


def _mmlu_pro_dataset_variable_options(option_counts: list[int]) -> Any:
    letters = "ABCDEFGHIJ"
    n = len(option_counts)
    return datasets.Dataset.from_dict({
        "question": [f"q{i}" for i in range(n)],
        "options": [[f"opt_{i}_{j}" for j in range(k)] for i, k in enumerate(option_counts)],
        "cot_content": [f"cot_reasoning_{i}" for i in range(n)],
        "answer": [letters[i % option_counts[i]] for i in range(n)],
        "category": [f"cat_{i % 3}" for i in range(n)],
    })


def _corpus_dataset(n: int) -> Any:
    return datasets.Dataset.from_dict({
        "text": [f"corpus_{i}" for i in range(n)],
    })


class TestParseAnswerLetter:
    def test_parses_bare_uppercase_letter(self) -> None:
        assert parse_answer_letter("A") == "A"
        assert parse_answer_letter("D") == "D"

    def test_parses_bare_lowercase_letter(self) -> None:
        assert parse_answer_letter("a") == "A"

    def test_parses_letter_with_trailing_text(self) -> None:
        assert parse_answer_letter("A. Some explanation") == "A"

    def test_takes_last_standalone_letter_for_chain_of_thought(self) -> None:
        assert parse_answer_letter("The answer is B because...") == "B"

    def test_takes_last_standalone_letter_when_multiple(self) -> None:
        assert parse_answer_letter("A or B") == "B"

    def test_takes_last_standalone_letter_for_unmarked_reasoning(self) -> None:
        assert parse_answer_letter("the final answer is C") == "C"

    def test_returns_none_for_refusal(self) -> None:
        assert parse_answer_letter("I cannot answer that question.") is None

    def test_returns_none_for_empty(self) -> None:
        assert parse_answer_letter("") is None

    def test_parses_final_letter_no_space(self) -> None:
        assert parse_answer_letter("finalB") == "B"

    def test_parses_final_letter_with_space(self) -> None:
        assert parse_answer_letter("Final C") == "C"

    def test_takes_last_final_letter_when_multiple(self) -> None:
        assert parse_answer_letter("finalA. Wait, finalB") == "B"

    def test_final_letter_overrides_earlier_standalone(self) -> None:
        assert parse_answer_letter("A is good but finalB") == "B"

    def test_parses_answer_marker_with_colon(self) -> None:
        assert parse_answer_letter("\n\nAnswer: D") == "D"

    def test_parses_answer_marker_with_markdown(self) -> None:
        assert parse_answer_letter("**Answer:** B") == "B"

    def test_parses_answer_marker_lowercase(self) -> None:
        assert parse_answer_letter("answer: c") == "C"

    def test_answer_marker_overrides_earlier_letters(self) -> None:
        assert parse_answer_letter("Looking at A and C. Answer: D") == "D"

    def test_chain_of_thought_with_explicit_answer_marker(self) -> None:
        text = "Let me consider option A. Then option B. Answer: D"
        assert parse_answer_letter(text) == "D"

    def test_final_marker_priority_over_answer_marker(self) -> None:
        assert parse_answer_letter("Answer: A. Reconsidering, final B") == "B"


class TestFormatMultipleChoice:
    def test_four_options_matches_canonical_layout(self) -> None:
        result = format_multiple_choice("Q?", ["a", "b", "c", "d"])
        expected = (
            "The following is a multiple choice question. "
            "Answer with only the letter (A, B, C, or D).\n"
            "\n"
            "Question: Q?\n"
            "A. a\n"
            "B. b\n"
            "C. c\n"
            "D. d\n"
            "\n"
            "Answer:"
        )
        assert result == expected

    def test_ten_options_matches_canonical_layout(self) -> None:
        result = format_multiple_choice("Q?", [f"o{i}" for i in range(10)])
        assert "(A, B, C, D, E, F, G, H, I, or J)" in result
        assert "A. o0" in result and "J. o9" in result

    def test_variable_option_counts_render(self) -> None:
        for n in (3, 4, 5, 6, 7, 8, 9, 10):
            options = [f"o{i}" for i in range(n)]
            result = format_multiple_choice("Q?", options)
            last_letter = chr(ord("A") + n - 1)
            assert f"{last_letter}. o{n - 1}" in result
            beyond = chr(ord("A") + n)
            assert f"\n{beyond}. " not in result


class TestGetTrainDatasetUsesMmluPro:
    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_train_dataset_loads_mmlu_pro_test_split(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _mmlu_pro_dataset(8)

        handler = WMDPHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            corpus_fraction=0.0,
            corpus_subset=None,
            max_examples=8,
        )
        tokenizer = _make_tokenizer()
        handler.get_train_dataset(tokenizer)

        mmlu_calls = [
            c for c in mock_load.call_args_list
            if c.args and c.args[0] == "TIGER-Lab/MMLU-Pro"
        ]
        assert len(mmlu_calls) == 1
        assert mmlu_calls[0].kwargs.get("split") == "test"

    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_train_dataset_respects_max_examples(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _mmlu_pro_dataset(64)

        handler = WMDPHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            corpus_fraction=0.0,
            corpus_subset=None,
            max_examples=16,
        )
        tokenizer = _make_tokenizer()
        train_ds = handler.get_train_dataset(tokenizer)

        assert len(train_ds) == 16

    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_train_prompt_renders_ten_options(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _mmlu_pro_dataset(2)

        handler = WMDPHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            corpus_fraction=0.0,
            corpus_subset=None,
            max_examples=2,
        )
        tokenizer = _make_tokenizer()
        handler.get_train_dataset(tokenizer)

        seen_user_contents = [
            call.args[0][0]["content"]
            for call in tokenizer.apply_chat_template.call_args_list
        ]
        assert seen_user_contents
        assert all("J." in content for content in seen_user_contents)

    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_train_dataset_handles_variable_option_counts(
        self, mock_load: MagicMock
    ) -> None:
        mock_load.return_value = _mmlu_pro_dataset_variable_options(
            [3, 4, 5, 6, 7, 8, 9, 10]
        )

        handler = WMDPHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            corpus_fraction=0.0,
            corpus_subset=None,
            max_examples=8,
        )
        tokenizer = _make_tokenizer()
        train_ds = handler.get_train_dataset(tokenizer)
        assert len(train_ds) == 8

        seen_user_contents = [
            call.args[0][0]["content"]
            for call in tokenizer.apply_chat_template.call_args_list
        ]
        has_3_options = any(
            "C." in c and "D." not in c for c in seen_user_contents
        )
        has_10_options = any("J." in c for c in seen_user_contents)
        assert has_3_options, "expected at least one 3-option row to render"
        assert has_10_options, "expected at least one 10-option row to render"

    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_train_completion_is_bare_answer_letter(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _mmlu_pro_dataset(3)

        handler = WMDPHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            corpus_fraction=0.0,
            corpus_subset=None,
            max_examples=3,
        )
        tokenizer = _make_tokenizer()
        handler.get_train_dataset(tokenizer)

        seen_assistant_contents = [
            call.args[0][1]["content"]
            for call in tokenizer.apply_chat_template.call_args_list
        ]
        assert seen_assistant_contents
        for content in seen_assistant_contents:
            assert re.fullmatch(r"[A-J]", content) is not None, (
                f"assistant text must be a single letter, got {content!r}"
            )
            assert "final" not in content.lower()

    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_train_dataset_columns_are_input_ids_and_attention_mask(
        self, mock_load: MagicMock
    ) -> None:
        mock_load.return_value = _mmlu_pro_dataset(4)

        handler = WMDPHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            corpus_fraction=0.0,
            corpus_subset=None,
            max_examples=4,
        )
        tokenizer = _make_tokenizer()
        train_ds = handler.get_train_dataset(tokenizer)

        assert "input_ids" in train_ds.column_names
        assert "attention_mask" in train_ds.column_names


class TestGetAnswerStartToken:
    """Marker must come from the chat template, else collator masks every label."""

    def test_marker_is_qwen_assistant_header(self) -> None:
        tokenizer = MagicMock()
        user_only = "<|im_start|>user\nQ<|im_end|>\n"
        user_with_gen = user_only + "<|im_start|>assistant\n"

        def fake_apply(messages: list[dict[str, str]], **kwargs: object) -> str:
            return user_with_gen if kwargs.get("add_generation_prompt") else user_only

        tokenizer.apply_chat_template = MagicMock(side_effect=fake_apply)
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens",
            corpus_fraction=0.0, corpus_subset=None, max_examples=4,
        )
        assert handler.get_answer_start_token(tokenizer) == "<|im_start|>assistant"

    def test_marker_is_present_in_rendered_training_text(self) -> None:
        tokenizer = MagicMock()
        user_only = "<|im_start|>user\nQ<|im_end|>\n"
        user_with_gen = user_only + "<|im_start|>assistant\n"

        def fake_apply(messages: list[dict[str, str]], **kwargs: object) -> str:
            if kwargs.get("add_generation_prompt"):
                return user_with_gen
            if len(messages) == 2 and messages[1]["role"] == "assistant":
                return (
                    "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n"
                    "<|im_start|>assistant\n" + messages[1]["content"] + "<|im_end|>"
                )
            return user_only

        tokenizer.apply_chat_template = MagicMock(side_effect=fake_apply)
        tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        tokenizer.pad_token_id = 0

        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens",
            corpus_fraction=0.0, corpus_subset=None, max_examples=2,
        )
        marker = handler.get_answer_start_token(tokenizer)

        with patch(
            "information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset",
            return_value=_mmlu_pro_dataset(2),
        ):
            handler.get_train_dataset(tokenizer)

        rendered = [
            fake_apply(call.args[0], **call.kwargs)
            for call in tokenizer.apply_chat_template.call_args_list
            if len(call.args[0]) == 2
        ]
        assert rendered, "expected at least one assistant-turn render"
        for text in rendered:
            assert marker in text, (
                f"marker {marker!r} not in rendered training text {text!r}"
            )


class TestGetTrainDatasetMixHalf:
    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_cyber_mix_half_calls_both_loaders(self, mock_load: MagicMock) -> None:
        def side_effect(*args: Any, **kwargs: Any) -> Any:
            if args and args[0] == "TIGER-Lab/MMLU-Pro":
                return _mmlu_pro_dataset(64)
            return _corpus_dataset(64)

        mock_load.side_effect = side_effect

        handler = WMDPHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            corpus_fraction=0.5,
            corpus_subset="cyber",
            max_examples=128,
        )
        tokenizer = _make_tokenizer()
        train_ds = handler.get_train_dataset(tokenizer)

        assert len(train_ds) == 128

        mmlu_calls = [
            c for c in mock_load.call_args_list
            if c.args and c.args[0] == "TIGER-Lab/MMLU-Pro"
        ]
        corpus_calls = [
            c for c in mock_load.call_args_list
            if c.args and c.args[0] != "TIGER-Lab/MMLU-Pro"
        ]
        assert len(mmlu_calls) == 1
        assert len(corpus_calls) == 1
        assert corpus_calls[0].args[0] == "cais/wmdp-corpora"
        assert corpus_calls[0].args[1] == "cyber-forget-corpus"

    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_bio_mix_half_uses_bio_corpus_repo(self, mock_load: MagicMock) -> None:
        def side_effect(*args: Any, **kwargs: Any) -> Any:
            if args and args[0] == "TIGER-Lab/MMLU-Pro":
                return _mmlu_pro_dataset(64)
            return _corpus_dataset(64)

        mock_load.side_effect = side_effect

        handler = WMDPHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            corpus_fraction=0.5,
            corpus_subset="bio",
            max_examples=128,
        )
        tokenizer = _make_tokenizer()
        handler.get_train_dataset(tokenizer)

        corpus_calls = [
            c for c in mock_load.call_args_list
            if c.args and c.args[0] != "TIGER-Lab/MMLU-Pro"
        ]
        assert len(corpus_calls) == 1
        assert corpus_calls[0].args[0] == "cais/wmdp-bio-forget-corpus"

    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_mix_half_splits_examples_evenly(self, mock_load: MagicMock) -> None:
        def side_effect(*args: Any, **kwargs: Any) -> Any:
            if args and args[0] == "TIGER-Lab/MMLU-Pro":
                return _mmlu_pro_dataset(64)
            return _corpus_dataset(64)

        mock_load.side_effect = side_effect

        handler = WMDPHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            corpus_fraction=0.5,
            corpus_subset="cyber",
            max_examples=64,
        )
        tokenizer = _make_tokenizer()
        train_ds = handler.get_train_dataset(tokenizer)

        assert len(train_ds) == 64

    def test_mismatched_fraction_with_subset_raises(self) -> None:
        handler = WMDPHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            corpus_fraction=0.0,
            corpus_subset="bio",
            max_examples=64,
        )
        tokenizer = _make_tokenizer()
        with pytest.raises(ValueError):
            handler.get_train_dataset(tokenizer)

    def test_mismatched_fraction_without_subset_raises(self) -> None:
        handler = WMDPHandler(
            max_length=500,
            batch_size=8,
            generations_dir="/tmp/gens",
            corpus_fraction=0.5,
            corpus_subset=None,
            max_examples=64,
        )
        tokenizer = _make_tokenizer()
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
    def test_max_examples_does_not_limit_val_dataset(self, mock_load: MagicMock) -> None:
        mock_load.side_effect = [_make_mock_subset(s, n=3) for s in SUBJECTS]

        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens", max_examples=4
        )
        tokenizer = _make_tokenizer()
        val_ds = cast(GenerationValDataset, handler.get_val_dataset(tokenizer))

        assert len(val_ds) == 9


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
        tokenizer.batch_decode = MagicMock(return_value=["Answer: A"])

        batch = self._make_batch(1)
        batch["labels"] = [_make_label("A", "wmdp-bio")]

        correct, total = handler.validate_batch(
            model, tokenizer, batch, FakeStrategy({}, label_key="question"),
        )
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
        tokenizer.batch_decode = MagicMock(return_value=["Answer: B"])

        batch = self._make_batch(1)
        batch["labels"] = [_make_label("A", "wmdp-bio")]

        correct, total = handler.validate_batch(
            model, tokenizer, batch, FakeStrategy({}, label_key="question"),
        )
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

        correct, total = handler.validate_batch(
            model, tokenizer, batch, FakeStrategy({}, label_key="question"),
        )
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
            return_value=["Answer: A", "Answer: C", "I refuse to answer"]
        )

        batch = self._make_batch(3)

        correct, total = handler.validate_batch(
            model, tokenizer, batch, FakeStrategy({}, label_key="question"),
        )
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
        tokenizer.batch_decode = MagicMock(return_value=["Answer: A"])

        batch = self._make_batch(1)
        batch["labels"] = [_make_label("A", "wmdp-bio")]

        handler.validate_batch(
            model, tokenizer, batch, FakeStrategy({}, label_key="question"),
        )

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
        tokenizer.batch_decode = MagicMock(return_value=["Answer: A"])

        batch = self._make_batch(1)
        batch["labels"] = [_make_label("A", "wmdp-bio")]

        handler.validate_batch(
            model, tokenizer, batch, FakeStrategy({}, label_key="question"),
        )
        assert len(handler._completions) == 1

    def test_validate_batch_records_per_example_eval_flops(self) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens"
        )
        model = MagicMock()
        model.generate = MagicMock(
            return_value=torch.tensor([
                [1, 2, 3, 4, 5, 7, 8, 0],
                [1, 2, 3, 4, 5, 9, 0, 0],
                [1, 2, 3, 4, 5, 6, 7, 8],
            ], dtype=torch.long)
        )
        model.num_parameters = MagicMock(return_value=1_000_000)
        tokenizer = _make_tokenizer()
        tokenizer.batch_decode = MagicMock(return_value=["A", "B", "C"])

        batch = {
            "input_ids": torch.zeros(3, 5, dtype=torch.long),
            "attention_mask": torch.tensor([
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1],
            ], dtype=torch.long),
            "labels": [
                _make_label("A", "wmdp-bio"),
                _make_label("B", "wmdp-chem"),
                _make_label("C", "wmdp-cyber"),
            ],
        }

        handler.validate_batch(
            model, tokenizer, batch, FakeStrategy({}, label_key="question"),
        )

        model.num_parameters.assert_called_once_with(exclude_embeddings=True)
        n_input_expected = [5, 4, 4]
        n_output_expected = [2, 1, 3]
        for i, completion in enumerate(handler._completions):
            expected = 2 * 1_000_000 * (n_input_expected[i] + n_output_expected[i])
            assert completion["eval_flops"] == expected
            assert isinstance(completion["eval_flops"], int)


def _wmdp_completion(
    *,
    question: str = "q1",
    correct_answer: str = "A",
    subject: str = "wmdp-bio",
    response: str = "A",
    predicted_answer: str | None = "A",
    correct: bool = True,
    eval_flops: int = 0,
    attack_flops: int = 0,
    pareto_step_idx: int = 0,
) -> dict[str, Any]:
    return {
        "question": question,
        "choices": ["c0", "c1", "c2", "c3"],
        "correct_answer": correct_answer,
        "subject": subject,
        "response": response,
        "predicted_answer": predicted_answer,
        "correct": correct,
        "eval_flops": eval_flops,
        "attack_flops": attack_flops,
        "pareto_step_idx": pareto_step_idx,
    }


class TestSaveCompletions:
    def test_writes_input_data_and_responses_jsonl(self, tmp_path: Path) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir=str(tmp_path)
        )
        handler._completions = [_wmdp_completion()]

        handler.save_completions("test-run-456")

        run_dir = tmp_path / "test-run-456"
        assert (run_dir / "input_data.jsonl").exists()
        assert (run_dir / "responses.jsonl").exists()

    def test_input_data_has_correct_keys(self, tmp_path: Path) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir=str(tmp_path)
        )
        handler._completions = [_wmdp_completion()]

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
        handler._completions = [_wmdp_completion()]

        handler.save_completions("test-run-resp")

        run_dir = tmp_path / "test-run-resp"
        with open(run_dir / "responses.jsonl") as f:
            rows = [json.loads(line) for line in f]

        assert len(rows) == 1
        assert set(rows[0].keys()) == {
            "question", "response", "predicted_answer", "correct", "dependent_flops",
            "attack_flops", "pareto_step_idx",
        }

    def test_save_completions_writes_dependent_flops(self, tmp_path: Path) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir=str(tmp_path)
        )
        handler._completions = [
            _wmdp_completion(eval_flops=1234, attack_flops=100),
            _wmdp_completion(
                question="q2", correct_answer="B", subject="wmdp-chem",
                response="C", predicted_answer="C", correct=False,
                eval_flops=5678,
            ),
        ]

        handler.save_completions("test-run-flops")

        run_dir = tmp_path / "test-run-flops"
        with open(run_dir / "responses.jsonl") as f:
            rows = [json.loads(line) for line in f]

        assert len(rows) == 2
        assert rows[0]["dependent_flops"] == 1334
        assert rows[1]["dependent_flops"] == 5678
        assert isinstance(rows[0]["dependent_flops"], int)

    def test_null_predicted_answer_for_refusal(self, tmp_path: Path) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir=str(tmp_path)
        )
        handler._completions = [
            _wmdp_completion(
                response="I cannot answer", predicted_answer=None, correct=False,
            )
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
        handler._completions = [_wmdp_completion()]

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


class TestDeterministicShuffleMmluPro:
    def _seen_questions(self, handler: WMDPHandler, tokenizer: MagicMock) -> list[str]:
        handler.get_train_dataset(tokenizer)
        return [
            call.args[0][0]["content"]
            for call in tokenizer.apply_chat_template.call_args_list
        ]

    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_same_max_examples_picks_same_rows_across_runs(
        self, mock_load: MagicMock
    ) -> None:
        mock_load.return_value = _mmlu_pro_dataset(64)
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens",
            corpus_fraction=0.0, corpus_subset=None, max_examples=16,
        )
        tok_a, tok_b = _make_tokenizer(), _make_tokenizer()
        seen_a = self._seen_questions(handler, tok_a)
        seen_b = self._seen_questions(handler, tok_b)
        assert seen_a == seen_b, "same handler config must select the same rows"

    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_shuffle_changes_order_from_raw(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _mmlu_pro_dataset(64)
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens",
            corpus_fraction=0.0, corpus_subset=None, max_examples=16,
        )
        tokenizer = _make_tokenizer()
        seen = self._seen_questions(handler, tokenizer)
        seen_qids = re.findall(r"Question: (q\d+)", "\n".join(seen))
        first_16_raw = [f"q{i}" for i in range(16)]
        assert sorted(seen_qids) != sorted(first_16_raw), (
            "shuffle must produce a different selection than raw category-sorted order"
        )

    @patch("information_safety.algorithms.dataset_handlers.wmdp.datasets.load_dataset")
    def test_small_max_examples_spans_multiple_categories(
        self, mock_load: MagicMock
    ) -> None:
        n = 99
        letters = "ABCDEFGHIJ"
        raw = datasets.Dataset.from_dict({
            "question": [f"q{i}" for i in range(n)],
            "options": [[f"opt_{i}_{j}" for j in range(10)] for i in range(n)],
            "cot_content": [f"cot_{i}" for i in range(n)],
            "answer": [letters[i % 10] for i in range(n)],
            "category": [f"cat_{i % 3}" for i in range(n)],
        })
        mock_load.return_value = raw

        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens",
            corpus_fraction=0.0, corpus_subset=None, max_examples=16,
        )
        train_ds = handler.get_train_dataset(_make_tokenizer())
        assert len(train_ds) == 16


class TestResultRowExtrasPerSubject:
    def test_per_subject_keys_present_after_completions_recorded(self) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens",
            corpus_fraction=0.0, corpus_subset=None, max_examples=4,
        )
        handler._completions = [
            _wmdp_completion(subject="wmdp-bio", correct=True),
            _wmdp_completion(subject="wmdp-bio", correct=False),
            _wmdp_completion(subject="wmdp-chem", correct=True),
            _wmdp_completion(subject="wmdp-chem", correct=True),
            _wmdp_completion(subject="wmdp-cyber", correct=False),
            _wmdp_completion(subject="wmdp-cyber", correct=False),
        ]
        extras = handler.result_row_extras()
        assert extras["bio_performance"] == pytest.approx(0.5)
        assert extras["chem_performance"] == pytest.approx(1.0)
        assert extras["cyber_performance"] == pytest.approx(0.0)

    def test_per_subject_keys_are_none_when_no_completions(self) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens",
            corpus_fraction=0.0, corpus_subset=None, max_examples=4,
        )
        extras = handler.result_row_extras()
        assert extras["bio_performance"] is None
        assert extras["chem_performance"] is None
        assert extras["cyber_performance"] is None

    def test_per_subject_keys_none_for_missing_subject(self) -> None:
        handler = WMDPHandler(
            max_length=500, batch_size=8, generations_dir="/tmp/gens",
            corpus_fraction=0.0, corpus_subset=None, max_examples=4,
        )
        handler._completions = [_wmdp_completion(subject="wmdp-bio", correct=True)]
        extras = handler.result_row_extras()
        assert extras["bio_performance"] == pytest.approx(1.0)
        assert extras["chem_performance"] is None
        assert extras["cyber_performance"] is None
