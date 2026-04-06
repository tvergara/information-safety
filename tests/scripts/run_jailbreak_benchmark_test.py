"""Tests for jailbreak benchmark runner script."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from information_safety.algorithms.jailbreak.prompt_methods import PromptAttackMethod
from scripts.run_jailbreak_benchmark import run_method


def test_run_method_writes_artifacts_and_result_row(tmp_path: Path) -> None:
    generations_dir = tmp_path / "generations"
    results_file = tmp_path / "results.jsonl"

    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate.side_effect = lambda input_ids, **kw: torch.cat(
        [input_ids, torch.full((input_ids.shape[0], 2), 99)], dim=1,
    )

    tokenizer = MagicMock()
    tokenizer.side_effect = lambda text, **kw: {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    tokenizer.decode.return_value = "mock-response"
    tokenizer.pad_token_id = 0

    behaviors = [
        {"behavior": "b1", "category": "c1"},
        {"behavior": "b2", "category": "c2"},
    ]

    with patch(
        "scripts.run_jailbreak_benchmark.compute_cross_entropy_bits",
        return_value=42.0,
    ):
        run_method(
            method=PromptAttackMethod.ZERO_SHOT,
            behaviors=behaviors,
            model=model,
            tokenizer=tokenizer,
            model_name="test-model",
            max_new_tokens=4,
            seed=42,
            results_file=results_file,
            generations_dir=generations_dir,
        )

    assert results_file.exists()
    rows = [json.loads(line) for line in results_file.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["experiment_name"] == "zero_shot"
    assert rows[0]["model_name"] == "test-model"
    assert rows[0]["bits"] == 42.0
    assert rows[0]["asr"] is None

    eval_run_id = rows[0]["eval_run_id"]
    run_dir = generations_dir / eval_run_id
    assert (run_dir / "input_data.jsonl").exists()
    assert (run_dir / "responses.jsonl").exists()

    responses = [json.loads(line) for line in (run_dir / "responses.jsonl").read_text().splitlines()]
    assert len(responses) == 2
    assert responses[0]["response"] == "mock-response"
