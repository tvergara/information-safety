"""Tests for jailbreak benchmark runner utilities."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import torch

from information_safety.algorithms.jailbreak.runner import (
    generate_attack_responses,
    write_generation_artifacts,
)


def _mock_model_tokenizer() -> tuple[MagicMock, MagicMock]:
    seq_len = 3
    vocab_size = 100

    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate.side_effect = lambda input_ids, **kw: torch.cat(
        [input_ids, torch.full((input_ids.shape[0], 2), 99)], dim=1,
    )
    model.return_value = MagicMock(logits=torch.randn(1, seq_len, vocab_size))

    tokenizer = MagicMock()
    tokenizer.side_effect = lambda text, **kw: {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    tokenizer.decode.return_value = "mock-response"
    tokenizer.pad_token_id = 0
    return model, tokenizer


def test_generate_attack_responses_adds_response_and_bits() -> None:
    model, tokenizer = _mock_model_tokenizer()
    rows = [
        {
            "attack_method": "zero_shot",
            "attack_prompt": "p1",
            "attack_template": "t1",
            "behavior": "b1",
            "category": "c1",
            "bits": 12,
        }
    ]
    out = generate_attack_responses(
        rows,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=4,
    )
    assert out[0]["response"] == "mock-response"
    assert isinstance(out[0]["bits"], float)
    assert out[0]["bits"] > 0


def test_write_generation_artifacts_writes_expected_files(tmp_path: Path) -> None:
    rows = [
        {
            "attack_method": "zero_shot",
            "attack_prompt": "p1",
            "behavior": "b1",
            "category": "c1",
            "response": "r1",
            "bits": 12,
            "query_budget": 1,
        }
    ]
    write_generation_artifacts(str(tmp_path), rows)

    assert (tmp_path / "input_data.jsonl").exists()
    assert (tmp_path / "responses.jsonl").exists()
    assert (tmp_path / "attack_results.jsonl").exists()

    with open(tmp_path / "attack_results.jsonl") as f:
        payload = [json.loads(line) for line in f]
    assert payload[0]["attack_method"] == "zero_shot"
