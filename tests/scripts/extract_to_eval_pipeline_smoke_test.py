"""Smoke test for the GCG Pareto extract -> merge -> strategy -> evaluator pipeline.

No real model is loaded; the strategy's handler is mocked, the evaluator's classifier is mocked.
Verifies the data plumbing only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from information_safety.algorithms.strategies.precomputed_adversarial_prompt import (
    PrecomputedGCGStrategy,
)
from scripts.evaluate_harmbench import evaluate_results_file
from scripts.extract_adversariallm_attacks import convert_run_dir
from scripts.rebuild_merged_jsonl import rebuild_merged_jsonl

_GCG_CONFIG: dict[str, object] = {
    "num_steps": 10,
    "topk": 8,
    "optim_str_init": "x x x x",
}


def _gcg_step(idx: int, loss: float, flops: int, prompt: str, completion: str) -> dict:
    return {
        "step": idx,
        "loss": loss,
        "flops": flops,
        "model_input": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},
        ],
        "model_completions": [completion],
    }


def _write_run_json(path: Path, behavior: str, steps: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": {
            "attack_params": dict(_GCG_CONFIG),
            "model_params": {"id": "fake/model"},
        },
        "runs": [
            {
                "original_prompt": [
                    {"role": "user", "content": behavior},
                    {"role": "assistant", "content": "tgt"},
                ],
                "steps": steps,
                "total_time": 0.0,
            }
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def test_extract_to_eval_pipeline_smoke(tmp_path: Path) -> None:
    adv_dir = tmp_path / "adv_out"
    _write_run_json(
        adv_dir / "0" / "run.json",
        behavior="beh_A",
        steps=[
            _gcg_step(0, 2.0, 10, "p0", "c0"),
            _gcg_step(1, 0.5, 10, "p1", "c1"),
        ],
    )

    shard = tmp_path / "shard.jsonl"
    convert_run_dir(str(adv_dir), str(shard))

    merged = tmp_path / "merged.jsonl"
    rebuild_merged_jsonl(attack="gcg", shard_paths=[shard], output_path=merged)

    merged_rows = [json.loads(line) for line in merged.read_text().splitlines()]
    assert len(merged_rows) == 2

    strategy = PrecomputedGCGStrategy(suffix_file=str(merged))

    model = MagicMock()
    tokenizer = MagicMock()
    pl_module = MagicMock()
    pl_module.tokenizer = tokenizer

    handler_setup = MagicMock()
    handler_setup.get_val_dataset.return_value = [
        {"labels": json.dumps({"behavior": "beh_A", "category": "c"})}
    ]
    strategy.setup(model, handler_setup, pl_module)

    handler = MagicMock()
    handler.max_length = 128
    handler.record_precomputed_completions.return_value = (0, 2)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": [json.dumps({"behavior": "beh_A", "category": "c"})],
    }
    correct, total = strategy.validation_step(model, tokenizer, batch, handler)
    assert total == 2

    recorded_labels, recorded_completions = (
        handler.record_precomputed_completions.call_args[0]
    )
    assert len(recorded_labels) == 2
    parsed = sorted(
        (
            (json.loads(label), comp)
            for label, comp in zip(recorded_labels, recorded_completions)
        ),
        key=lambda x: x[0]["pareto_step_idx"],
    )
    assert parsed[0][0]["pareto_step_idx"] == 0
    assert parsed[1][0]["pareto_step_idx"] == 1

    gen_base = tmp_path / "generations"
    run_dir = gen_base / "smoke-run"
    run_dir.mkdir(parents=True)

    input_data = [
        {"behavior": "beh_A", "category": "c", "pareto_step_idx": 0},
        {"behavior": "beh_A", "category": "c", "pareto_step_idx": 1},
    ]
    response_data = [
        {"behavior": "beh_A", "response": "c1", "pareto_step_idx": 0,
         "dependent_flops": 20, "attack_flops": 20},
        {"behavior": "beh_A", "response": "c0", "pareto_step_idx": 1,
         "dependent_flops": 10, "attack_flops": 10},
    ]
    with open(run_dir / "input_data.jsonl", "w") as f:
        for row in input_data:
            f.write(json.dumps(row) + "\n")
    with open(run_dir / "responses.jsonl", "w") as f:
        for row in response_data:
            f.write(json.dumps(row) + "\n")

    results_file = tmp_path / "results.jsonl"
    with open(results_file, "w") as f:
        f.write(
            json.dumps(
                {
                    "eval_run_id": "smoke-run",
                    "dataset_name": "advbench_harmbench",
                    "asr": None,
                }
            )
            + "\n"
        )

    classifier_model = MagicMock()
    classifier_model.device = torch.device("cpu")
    classifier_model.generate.side_effect = lambda input_ids, **kw: torch.cat(
        [input_ids, torch.full((input_ids.shape[0], 1), 99)], dim=1
    )
    classifier_tok = MagicMock()
    classifier_tok.side_effect = lambda prompts, **kw: {
        "input_ids": torch.tensor([[1, 2, 3]] * len(prompts)),
        "attention_mask": torch.tensor([[1, 1, 1]] * len(prompts)),
    }
    classifier_tok.decode.side_effect = ["Yes", "No"]

    with patch(
        "scripts.evaluate_harmbench.load_classifier",
        return_value=(classifier_model, classifier_tok),
    ):
        evaluate_results_file(str(results_file), generations_base=str(gen_base))

    with open(results_file) as f:
        updated = [json.loads(line) for line in f]
    assert len(updated) == 1
    assert updated[0]["asr"] == 1.0
