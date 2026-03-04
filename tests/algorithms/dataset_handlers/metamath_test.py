"""Tests for MetaMath dataset handler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from information_safety.algorithms.dataset_handlers.metamath import (
    MetaMathHandler,
    extract_numerical_answer,
    format_example,
)


class TestFormatExample:
    def test_formats_query_and_response(self) -> None:
        example = {"query": "What is 2+2?", "response": "The answer is 4."}
        result = format_example(example)
        assert "What is 2+2?" in result
        assert "The answer is 4." in result

    def test_has_question_and_answer_markers(self) -> None:
        example = {"query": "Q", "response": "A"}
        result = format_example(example)
        assert "### Question: Q" in result
        assert "### Answer: A" in result


class TestExtractNumericalAnswer:
    def test_extracts_boxed_answer(self) -> None:
        text = "The answer is #### 42"
        assert extract_numerical_answer(text) == "42"

    def test_extracts_decimal(self) -> None:
        text = "The answer is #### 3.14"
        assert extract_numerical_answer(text) == "3.14"

    def test_extracts_negative(self) -> None:
        text = "The answer is #### -7"
        assert extract_numerical_answer(text) == "-7"

    def test_returns_none_for_no_match(self) -> None:
        text = "No numerical answer here"
        assert extract_numerical_answer(text) is None

    def test_extracts_from_model_output(self) -> None:
        text = "Let me calculate... The answer is #### 100"
        assert extract_numerical_answer(text) == "100"


class TestMetaMathHandler:
    @patch("information_safety.algorithms.dataset_handlers.metamath.datasets.load_dataset")
    def test_get_train_dataset(self, mock_load: MagicMock) -> None:
        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(
            side_effect=lambda key: {"query": "Q?", "response": "A. #### 1"}
        )
        mock_ds.__len__ = MagicMock(return_value=10)
        mock_ds.map = MagicMock(return_value=mock_ds)
        mock_load.return_value = mock_ds

        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        tokenizer.eos_token = "</s>"
        tokenizer.eos_token_id = 2
        tokenizer.pad_token = "<pad>"
        tokenizer.pad_token_id = 0

        handler = MetaMathHandler(max_length=500, batch_size=8)
        handler.get_train_dataset(tokenizer)
        mock_load.assert_called_once()

    @patch("information_safety.algorithms.dataset_handlers.metamath.datasets.load_dataset")
    def test_get_val_dataset(self, mock_load: MagicMock) -> None:
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(
            return_value=iter([{"question": "What is 1+1?", "answer": "#### 2"}])
        )
        mock_ds.__len__ = MagicMock(return_value=1)
        mock_load.return_value = mock_ds

        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        tokenizer.eos_token = "</s>"
        tokenizer.eos_token_id = 2
        tokenizer.pad_token = "<pad>"
        tokenizer.pad_token_id = 0

        handler = MetaMathHandler(max_length=500, batch_size=8)
        handler.get_val_dataset(tokenizer)
        mock_load.assert_called_once()

    def test_validate_batch_extracts_answers(self) -> None:
        handler = MetaMathHandler(max_length=500, batch_size=8)

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.batch_decode = MagicMock(
            return_value=["The answer is #### 42", "The answer is #### 7"]
        )
        tokenizer.pad_token_id = 0

        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "labels": ["42", "7"],
        }

        generated_ids = torch.tensor([[1, 2, 3, 7, 8], [4, 5, 6, 9, 10]])
        model.generate = MagicMock(return_value=generated_ids)

        correct, total = handler.validate_batch(model, tokenizer, batch)
        assert total == 2
        assert correct == 2
