"""Tests for the Refusal SFT defense adapter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from information_safety.defenses.refusal_sft import (
    SFTHParams,
    _build_chat_text,
    _load_jsonl,
    train_refusal_sft,
)


def _write_refusals(path: Path, n: int) -> None:
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({
                "prompt": f"p_{i}",
                "refusal": "I can't help with that.",
                "source": "mmlu-pro-biology",
                "id": f"x-{i}",
            }) + "\n")


def _write_retain(path: Path, n: int) -> None:
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({
                "prompt": f"u_{i}",
                "response": f"a_{i}",
                "source": "smoltalk",
                "id": f"smoltalk-{i}",
            }) + "\n")


def _mock_tokenizer() -> MagicMock:
    tok = MagicMock()

    def _apply_chat_template(messages: list[dict[str, str]], **_kw: object) -> str:
        parts = []
        for m in messages:
            parts.append(f"<|{m['role']}|>\n{m['content']}<|end|>")
        return "\n".join(parts)

    tok.apply_chat_template = MagicMock(side_effect=_apply_chat_template)
    tok.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    tok.pad_token_id = 0
    tok.pad_token = "<pad>"
    tok.eos_token = "<eos>"
    return tok


class TestLoadJsonl:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        _write_refusals(path, 5)
        rows = _load_jsonl(path)
        assert len(rows) == 5
        for row in rows:
            assert set(row.keys()) == {"prompt", "refusal", "source", "id"}


class TestBuildChatText:
    def test_calls_apply_chat_template_with_two_messages(self) -> None:
        tok = _mock_tokenizer()
        _build_chat_text(tok, user="u", assistant="a")
        tok.apply_chat_template.assert_called_once()
        msgs = tok.apply_chat_template.call_args[0][0]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"


class TestTrainRefusalSft:
    @patch("information_safety.defenses.refusal_sft.AutoTokenizer.from_pretrained")
    @patch("information_safety.defenses.refusal_sft.AutoModelForCausalLM.from_pretrained")
    @patch("information_safety.defenses.refusal_sft.Trainer")
    def test_loads_base_model_and_runs_trainer(
        self,
        mock_trainer_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tok_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        refusals_path = tmp_path / "refusals.jsonl"
        retain_path = tmp_path / "retain.jsonl"
        out_dir = tmp_path / "out"
        _write_refusals(refusals_path, 3)
        _write_retain(retain_path, 3)

        mock_tok_cls.return_value = _mock_tokenizer()
        mock_model_cls.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        train_refusal_sft(
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            refusals_path=refusals_path,
            retain_path=retain_path,
            output_dir=out_dir,
            hparams=SFTHParams(max_steps=2, per_device_train_batch_size=1, bf16=False),
        )

        mock_model_cls.assert_called_once()
        assert mock_model_cls.call_args[0][0] == "meta-llama/Llama-3.1-8B-Instruct"
        mock_trainer.train.assert_called_once()
        mock_trainer.save_model.assert_called_once()

    @patch("information_safety.defenses.refusal_sft.AutoTokenizer.from_pretrained")
    @patch("information_safety.defenses.refusal_sft.AutoModelForCausalLM.from_pretrained")
    @patch("information_safety.defenses.refusal_sft.Trainer")
    def test_dry_run_does_not_train(
        self,
        mock_trainer_cls: MagicMock,
        mock_model_cls: MagicMock,
        mock_tok_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        refusals_path = tmp_path / "refusals.jsonl"
        retain_path = tmp_path / "retain.jsonl"
        out_dir = tmp_path / "out"
        _write_refusals(refusals_path, 2)
        _write_retain(retain_path, 2)

        mock_tok_cls.return_value = _mock_tokenizer()
        mock_model_cls.return_value = MagicMock()

        train_refusal_sft(
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            refusals_path=refusals_path,
            retain_path=retain_path,
            output_dir=out_dir,
            hparams=SFTHParams(),
            dry_run=True,
        )

        mock_trainer_cls.assert_not_called()

    def test_missing_files_raise(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            train_refusal_sft(
                base_model="x",
                refusals_path=tmp_path / "missing.jsonl",
                retain_path=tmp_path / "missing.jsonl",
                output_dir=tmp_path / "out",
                hparams=SFTHParams(),
            )
