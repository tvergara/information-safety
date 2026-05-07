"""Tests for SafetyPairHandler dataset handler."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import torch

from information_safety.algorithms.dataset_handlers.safety_pair import SafetyPairHandler

MOCK_REFUSAL_DATA = [
    {"prompt": f"harmful_prompt_{i}", "llama3_output": f"refusal_{i}", "output": f"bad_{i}"}
    for i in range(10)
]

MOCK_SMOLTALK_EXAMPLES = [
    {"messages": [{"role": "user", "content": f"q_{i}"}, {"role": "assistant", "content": f"a_{i}"}]}
    for i in range(100)
]

MOCK_HARMBENCH_BEHAVIORS = {
    "name": "test",
    "description": "test",
    "data": [
        {"behavior": f"behavior_{i}", "default_target": "", "messages": []}
        for i in range(5)
    ],
}


def _make_handler(
    include_refusals: bool = False,
    num_benign_examples: int = 20,
    seed: int = 42,
) -> SafetyPairHandler:
    return SafetyPairHandler(
        include_refusals=include_refusals,
        num_benign_examples=num_benign_examples,
        refusal_data_path="/fake/refusal.json",
        harmbench_test_path="/fake/harmbench.json",
        max_length=512,
        batch_size=8,
        generations_dir="/tmp/gens",
        seed=seed,
    )


def _mock_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock(return_value="<formatted>")
    tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    tokenizer.pad_token_id = 0
    return tokenizer


def _mock_benign_dataset(size: int = 100) -> MagicMock:
    mock_ds = MagicMock()
    mock_ds.__len__ = MagicMock(return_value=size)
    mock_ds.filter = MagicMock(return_value=mock_ds)
    mock_ds.select = MagicMock(return_value=mock_ds)
    mock_ds.column_names = ["messages"]
    mock_ds.map = MagicMock(return_value=mock_ds)
    return mock_ds


class TestTrainDatasetWithoutRefusals:
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.datasets.load_dataset")
    @patch("builtins.open", mock_open(read_data=json.dumps(MOCK_REFUSAL_DATA)))
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.json.load")
    def test_contains_only_benign_examples(
        self, mock_json_load: MagicMock, mock_load_dataset: MagicMock
    ) -> None:
        mock_json_load.return_value = MOCK_REFUSAL_DATA
        mock_ds = _mock_benign_dataset()
        mock_load_dataset.return_value = mock_ds

        handler = _make_handler(include_refusals=False, num_benign_examples=20)
        tokenizer = _mock_tokenizer()
        handler.get_train_dataset(tokenizer)

        mock_ds.select.assert_called_once()
        select_arg = mock_ds.select.call_args[0][0]
        assert len(select_arg) == 20



class TestTrainDatasetWithRefusals:
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.datasets.concatenate_datasets")
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.datasets.Dataset.from_list")
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.datasets.load_dataset")
    @patch("builtins.open", mock_open(read_data=json.dumps(MOCK_REFUSAL_DATA)))
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.json.load")
    def test_total_length_equals_num_benign(
        self,
        mock_json_load: MagicMock,
        mock_load_dataset: MagicMock,
        mock_from_list: MagicMock,
        mock_concat: MagicMock,
    ) -> None:
        mock_json_load.return_value = MOCK_REFUSAL_DATA
        num_refusals = len(MOCK_REFUSAL_DATA)
        num_benign = 20

        mock_ds = _mock_benign_dataset()
        mock_load_dataset.return_value = mock_ds

        mock_refusal_ds = MagicMock()
        mock_refusal_ds.column_names = ["prompt", "llama3_output"]
        mock_refusal_ds.map = MagicMock(return_value=mock_refusal_ds)
        mock_from_list.return_value = mock_refusal_ds

        handler = _make_handler(include_refusals=True, num_benign_examples=num_benign)
        tokenizer = _mock_tokenizer()
        handler.get_train_dataset(tokenizer)

        select_arg = mock_ds.select.call_args[0][0]
        assert len(select_arg) == num_benign - num_refusals



class TestMatchedDatasetSizes:
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.datasets.concatenate_datasets")
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.datasets.Dataset.from_list")
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.datasets.load_dataset")
    @patch("builtins.open", mock_open(read_data=json.dumps(MOCK_REFUSAL_DATA)))
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.json.load")
    def test_both_modes_produce_same_total(
        self,
        mock_json_load: MagicMock,
        mock_load_dataset: MagicMock,
        mock_from_list: MagicMock,
        mock_concat: MagicMock,
    ) -> None:
        """Both include_refusals=True and False produce N total training examples."""
        mock_json_load.return_value = MOCK_REFUSAL_DATA
        num_refusals = len(MOCK_REFUSAL_DATA)
        num_benign = 20

        benign_ds = _mock_benign_dataset()
        mock_load_dataset.return_value = benign_ds

        mock_refusal_ds = MagicMock()
        mock_refusal_ds.column_names = ["prompt", "llama3_output"]
        mock_refusal_ds.map = MagicMock(return_value=mock_refusal_ds)
        mock_from_list.return_value = mock_refusal_ds

        tokenizer = _mock_tokenizer()

        handler_u = _make_handler(include_refusals=False, num_benign_examples=num_benign)
        handler_u.get_train_dataset(tokenizer)
        select_u = benign_ds.select.call_args[0][0]
        total_u = len(select_u)

        benign_ds.select.reset_mock()

        handler_s = _make_handler(include_refusals=True, num_benign_examples=num_benign)
        handler_s.get_train_dataset(tokenizer)
        select_s = benign_ds.select.call_args[0][0]
        total_s = len(select_s) + num_refusals

        assert total_u == total_s == num_benign


class TestChatFormatting:
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.datasets.load_dataset")
    @patch("builtins.open", mock_open(read_data=json.dumps(MOCK_REFUSAL_DATA)))
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.json.load")
    def test_benign_examples_use_chat_template(
        self, mock_json_load: MagicMock, mock_load_dataset: MagicMock
    ) -> None:
        mock_json_load.return_value = MOCK_REFUSAL_DATA

        captured_fn = None

        mock_ds = _mock_benign_dataset()

        def capture_map(fn: object, **kwargs: object) -> MagicMock:
            nonlocal captured_fn
            captured_fn = fn
            return MagicMock()

        mock_ds.map = capture_map
        mock_load_dataset.return_value = mock_ds

        handler = _make_handler(include_refusals=False)
        tokenizer = _mock_tokenizer()
        handler.get_train_dataset(tokenizer)

        assert captured_fn is not None
        example = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
        captured_fn(example)

        tokenizer.apply_chat_template.assert_called_once_with(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            tokenize=False,
        )


class TestValidationDatasetLoads:
    @patch("builtins.open", mock_open(read_data=json.dumps(MOCK_HARMBENCH_BEHAVIORS)))
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.json.load")
    def test_loads_correct_number_of_behaviors(self, mock_json_load: MagicMock) -> None:
        mock_json_load.return_value = MOCK_HARMBENCH_BEHAVIORS

        handler = _make_handler()
        tokenizer = _mock_tokenizer()
        val_ds = handler.get_val_dataset(tokenizer)

        assert len(val_ds) == len(MOCK_HARMBENCH_BEHAVIORS["data"])  # type: ignore[arg-type]


class TestRefusalDataFieldExtraction:
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.datasets.concatenate_datasets")
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.datasets.Dataset.from_list")
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.datasets.load_dataset")
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.json.load")
    def test_uses_prompt_and_llama3_output_fields(
        self,
        mock_json_load: MagicMock,
        mock_load_dataset: MagicMock,
        mock_from_list: MagicMock,
        mock_concat: MagicMock,
    ) -> None:
        refusal_data = [
            {"prompt": "harmful_q", "llama3_output": "I cannot help", "output": "bad_answer"}
        ]
        mock_json_load.return_value = refusal_data

        mock_ds = _mock_benign_dataset()
        mock_load_dataset.return_value = mock_ds

        mock_refusal_ds = MagicMock()
        mock_refusal_ds.column_names = ["prompt", "llama3_output"]

        captured_refusal_fn = None

        def capture_refusal_map(fn: object, **kwargs: object) -> MagicMock:
            nonlocal captured_refusal_fn
            captured_refusal_fn = fn
            return MagicMock()

        mock_refusal_ds.map = capture_refusal_map
        mock_from_list.return_value = mock_refusal_ds

        handler = _make_handler(include_refusals=True, num_benign_examples=20)
        tokenizer = _mock_tokenizer()

        with patch("builtins.open", mock_open()):
            handler.get_train_dataset(tokenizer)

        assert captured_refusal_fn is not None
        captured_refusal_fn({"prompt": "harmful_q", "llama3_output": "I cannot help"})

        call_args = tokenizer.apply_chat_template.call_args_list
        found = False
        for call in call_args:
            messages = call[0][0]
            if len(messages) == 2:
                if (
                    messages[0]["content"] == "harmful_q"
                    and messages[1]["content"] == "I cannot help"
                ):
                    found = True
                    break
        assert found, "Refusal data should use 'prompt' and 'llama3_output' fields"


class TestBenignDataSamplingIsSeeded:
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.datasets.load_dataset")
    @patch("builtins.open", mock_open(read_data=json.dumps(MOCK_REFUSAL_DATA)))
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.json.load")
    def test_same_seed_produces_same_indices(
        self, mock_json_load: MagicMock, mock_load_dataset: MagicMock
    ) -> None:
        mock_json_load.return_value = MOCK_REFUSAL_DATA

        mock_ds = _mock_benign_dataset()
        mock_load_dataset.return_value = mock_ds

        tokenizer = _mock_tokenizer()

        handler1 = _make_handler(include_refusals=False, num_benign_examples=20, seed=42)
        handler1.get_train_dataset(tokenizer)
        indices1 = list(mock_ds.select.call_args[0][0])

        mock_ds.select.reset_mock()

        handler2 = _make_handler(include_refusals=False, num_benign_examples=20, seed=42)
        handler2.get_train_dataset(tokenizer)
        indices2 = list(mock_ds.select.call_args[0][0])

        assert indices1 == indices2

    @patch("information_safety.algorithms.dataset_handlers.safety_pair.datasets.load_dataset")
    @patch("builtins.open", mock_open(read_data=json.dumps(MOCK_REFUSAL_DATA)))
    @patch("information_safety.algorithms.dataset_handlers.safety_pair.json.load")
    def test_different_seed_produces_different_indices(
        self, mock_json_load: MagicMock, mock_load_dataset: MagicMock
    ) -> None:
        mock_json_load.return_value = MOCK_REFUSAL_DATA

        mock_ds = _mock_benign_dataset()
        mock_load_dataset.return_value = mock_ds

        tokenizer = _mock_tokenizer()

        handler1 = _make_handler(include_refusals=False, num_benign_examples=20, seed=42)
        handler1.get_train_dataset(tokenizer)
        indices1 = list(mock_ds.select.call_args[0][0])

        mock_ds.select.reset_mock()

        handler2 = _make_handler(include_refusals=False, num_benign_examples=20, seed=99)
        handler2.get_train_dataset(tokenizer)
        indices2 = list(mock_ds.select.call_args[0][0])

        assert indices1 != indices2


class TestValidateBatch:
    def test_returns_zero_correct_and_batch_total(self) -> None:
        handler = _make_handler()

        model = MagicMock()
        model.generate = MagicMock(
            return_value=torch.tensor([[1, 2, 3, 7, 8], [4, 5, 6, 9, 10]])
        )

        tokenizer = _mock_tokenizer()
        tokenizer.batch_decode = MagicMock(return_value=["response1", "response2"])

        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "labels": [
                json.dumps({"behavior": "Do X"}),
                json.dumps({"behavior": "Do Y"}),
            ],
        }

        correct, total = handler.validate_batch(model, tokenizer, batch)
        assert correct == 0
        assert total == 2

    def test_accumulates_completions(self) -> None:
        handler = _make_handler()

        prefix_out = torch.tensor([[1, 2, 3, 7, 8]])
        continuation_out = torch.tensor([[1, 2, 3, 7, 8, 9, 10]])
        model = MagicMock()
        model.generate = MagicMock(side_effect=[prefix_out, continuation_out])

        tokenizer = _mock_tokenizer()
        tokenizer.batch_decode = MagicMock(return_value=["generated text"])

        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": [json.dumps({"behavior": "Do X"})],
        }

        handler.validate_batch(model, tokenizer, batch)

        assert len(handler._completions) == 1
        assert handler._completions[0]["behavior"] == "Do X"

    def test_two_stage_generation(self) -> None:
        handler = _make_handler()

        prompt_ids = torch.tensor([[10, 20, 30]])
        prefix_out = torch.tensor([[10, 20, 30, 40, 50]])
        continuation_out = torch.tensor([[10, 20, 30, 40, 50, 60, 70]])

        model = MagicMock()
        model.generate = MagicMock(side_effect=[prefix_out, continuation_out])

        tokenizer = _mock_tokenizer()
        tokenizer.batch_decode = MagicMock(return_value=["full response"])

        batch = {
            "input_ids": prompt_ids,
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": [json.dumps({"behavior": "Do X"})],
        }

        handler.validate_batch(model, tokenizer, batch)

        stage2_input = model.generate.call_args_list[1][1]["input_ids"]
        assert stage2_input.tolist() == prefix_out.tolist()
        model.disable_adapter_layers.assert_called_once()
        model.enable_adapter_layers.assert_called_once()


class TestSaveCompletions:
    def test_writes_jsonl_files(self, tmp_path: Path) -> None:
        handler = _make_handler()
        handler.generations_dir = str(tmp_path)
        handler._completions = [
            {"behavior": "Do X", "response": "Sure..."},
            {"behavior": "Do Y", "response": "Ok..."},
        ]

        handler.save_completions("test-run")

        run_dir = tmp_path / "test-run"
        assert (run_dir / "input_data.jsonl").exists()
        assert (run_dir / "responses.jsonl").exists()

        with open(run_dir / "input_data.jsonl") as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 2

    def test_resets_completions_after_save(self, tmp_path: Path) -> None:
        handler = _make_handler()
        handler.generations_dir = str(tmp_path)
        handler._completions = [
            {"behavior": "Do X", "response": "Sure..."},
        ]

        handler.save_completions("run-456")
        assert handler._completions == []


class TestGetAnswerStartToken:
    def test_llama3_marker_excludes_user_content(self) -> None:
        tokenizer = MagicMock()
        user_only = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "Q<|eot_id|>"
        )
        user_with_gen = user_only + "<|start_header_id|>assistant<|end_header_id|>\n\n"

        def fake_apply(messages: list[dict[str, str]], **kwargs: object) -> str:
            return user_with_gen if kwargs.get("add_generation_prompt") else user_only

        tokenizer.apply_chat_template = MagicMock(side_effect=fake_apply)

        handler = _make_handler()
        token = handler.get_answer_start_token(tokenizer)
        assert token == "<|start_header_id|>assistant<|end_header_id|>"
        assert "Q" not in token
        assert "<|eot_id|>" not in token

    def test_qwen_marker(self) -> None:
        tokenizer = MagicMock()
        user_only = "<|im_start|>user\nQ<|im_end|>\n"
        user_with_gen = user_only + "<|im_start|>assistant\n"

        def fake_apply(messages: list[dict[str, str]], **kwargs: object) -> str:
            return user_with_gen if kwargs.get("add_generation_prompt") else user_only

        tokenizer.apply_chat_template = MagicMock(side_effect=fake_apply)

        handler = _make_handler()
        token = handler.get_answer_start_token(tokenizer)
        assert token == "<|im_start|>assistant"

    def test_raises_when_template_has_no_assistant_suffix(self) -> None:
        import pytest

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="<|im_start|>user\nQ<|im_end|>")

        handler = _make_handler()
        with pytest.raises(ValueError, match="Empty assistant marker"):
            handler.get_answer_start_token(tokenizer)
