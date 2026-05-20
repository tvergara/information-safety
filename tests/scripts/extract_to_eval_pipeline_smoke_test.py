"""Smoke test for the extract -> merge -> strategy -> evaluator pipeline.

Covers all three iterative attacks (GCG, AutoDAN, PAIR). No real model is loaded; the strategy's
handler is mocked, the evaluator's classifier is mocked. Verifies the data plumbing only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from information_safety.algorithms.strategies.precomputed_adversarial_prompt import (
    PrecomputedAutoDANStrategy,
    PrecomputedGCGStrategy,
    PrecomputedPAIRStrategy,
)
from scripts import extract_adversariallm_attacks
from scripts.evaluate_harmbench import evaluate_results_file
from scripts.extract_adversariallm_attacks import convert_run_dir
from scripts.rebuild_merged_jsonl import rebuild_merged_jsonl


@pytest.fixture(autouse=True)
def _relax_min_attack_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extract_adversariallm_attacks, "MIN_ATTACK_STEPS", 1)

_GCG_CONFIG: dict[str, object] = {
    "num_steps": 10,
    "topk": 8,
    "optim_str_init": "x x x x",
}
_AUTODAN_CONFIG: dict[str, object] = {
    "num_steps": 10,
    "batch_size": 4,
    "mutation": 0.25,
}
_PAIR_CONFIG: dict[str, object] = {
    "num_steps": 25,
    "num_streams": 2,
    "attack_model": {
        "max_new_tokens": 8,
        "max_attempts": 4,
        "id": "fake/attacker",
        "num_params": 1_000_000,
    },
}


def _gcg_step(
    idx: int, loss: float | None, flops: int, prompt: str, completion: str
) -> dict:
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


def _write_run_json(
    path: Path,
    behavior: str,
    steps: list[dict],
    attack_params: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": {
            "attack_params": attack_params,
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


def _run_extract_strategy_eval(
    *,
    tmp_path: Path,
    attack: str,
    attack_params: dict[str, object],
    steps: list[dict],
    strategy_cls: type,
    pair_attacked_model: str | None,
    expected_num_rows: int,
    classifier_labels: list[str],
    expected_asr: float,
) -> None:
    adv_dir = tmp_path / "adv_out"
    _write_run_json(
        adv_dir / "0" / "run.json",
        behavior="beh_A",
        steps=steps,
        attack_params=attack_params,
    )

    shard = tmp_path / "shard.jsonl"
    convert_run_dir(str(adv_dir), str(shard))

    merged = tmp_path / "merged.jsonl"
    rebuild_merged_jsonl(
        attack=attack,
        shard_paths=[shard],
        output_path=merged,
        pair_attacked_model=pair_attacked_model,
    )

    merged_rows = [json.loads(line) for line in merged.read_text().splitlines()]
    assert len(merged_rows) == expected_num_rows

    strategy = strategy_cls(suffix_file=str(merged))

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
    handler.record_precomputed_completions.return_value = (0, expected_num_rows)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": [json.dumps({"behavior": "beh_A", "category": "c"})],
    }
    correct, total = strategy.validation_step(model, tokenizer, batch, handler)
    assert total == expected_num_rows

    recorded_labels, recorded_completions, _ = (
        handler.record_precomputed_completions.call_args[0]
    )
    assert len(recorded_labels) == expected_num_rows
    parsed = sorted(
        (
            (json.loads(label), comp)
            for label, comp in zip(recorded_labels, recorded_completions)
        ),
        key=lambda x: x[0]["pareto_step_idx"],
    )
    pareto_idxs = [p[0]["pareto_step_idx"] for p in parsed]
    assert pareto_idxs == sorted(set(pareto_idxs))
    assert pareto_idxs[0] == 0

    gen_base = tmp_path / "generations"
    run_dir = gen_base / "smoke-run"
    run_dir.mkdir(parents=True)

    input_data = [
        {"behavior": "beh_A", "category": "c", "pareto_step_idx": idx}
        for idx in pareto_idxs
    ]
    response_data = [
        {
            "behavior": "beh_A",
            "response": parsed[i][1],
            "pareto_step_idx": pareto_idxs[i],
            "dependent_flops": 0,
            "attack_flops": 0,
        }
        for i in range(expected_num_rows)
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
    classifier_tok.decode.side_effect = classifier_labels

    with patch(
        "scripts.evaluate_harmbench.load_classifier",
        return_value=(classifier_model, classifier_tok),
    ):
        evaluate_results_file(str(results_file), generations_base=str(gen_base))

    with open(results_file) as f:
        updated = [json.loads(line) for line in f]
    assert len(updated) == 1
    assert updated[0]["asr"] == expected_asr


def test_extract_to_eval_pipeline_smoke_gcg(tmp_path: Path) -> None:
    """GCG trajectory: 2 Pareto rows; canonical (idx=0) classifier label drives ASR."""
    steps = [
        _gcg_step(0, 2.0, 10, "p0", "c0"),
        _gcg_step(1, 0.5, 10, "p1", "c1"),
    ]
    _run_extract_strategy_eval(
        tmp_path=tmp_path,
        attack="gcg",
        attack_params=dict(_GCG_CONFIG),
        steps=steps,
        strategy_cls=PrecomputedGCGStrategy,
        pair_attacked_model=None,
        expected_num_rows=2,
        classifier_labels=["Yes", "No"],
        expected_asr=1.0,
    )


def test_extract_to_eval_pipeline_smoke_autodan(tmp_path: Path) -> None:
    """AutoDAN trajectory: 3 Pareto rows; canonical (idx=0) classifier label drives ASR."""
    steps = [
        _gcg_step(0, 3.0, 10, "p0", "c0"),
        _gcg_step(1, 1.8, 30, "p1", "c1"),
        _gcg_step(2, 1.0, 60, "p2", "c2"),
    ]
    _run_extract_strategy_eval(
        tmp_path=tmp_path,
        attack="autodan",
        attack_params=dict(_AUTODAN_CONFIG),
        steps=steps,
        strategy_cls=PrecomputedAutoDANStrategy,
        pair_attacked_model=None,
        expected_num_rows=3,
        classifier_labels=["Yes", "No", "No"],
        expected_asr=1.0,
    )


def test_extract_to_eval_pipeline_smoke_pair(tmp_path: Path) -> None:
    """PAIR trajectory: stride-sampled rows; canonical (idx=0) drives ASR."""
    n = 25
    steps = [_gcg_step(i, None, 1, f"p{i}", f"c{i}") for i in range(n)]
    _run_extract_strategy_eval(
        tmp_path=tmp_path,
        attack="pair",
        attack_params=dict(_PAIR_CONFIG),
        steps=steps,
        strategy_cls=PrecomputedPAIRStrategy,
        pair_attacked_model="fake/attacked",
        expected_num_rows=3,
        classifier_labels=["Yes", "No", "No"],
        expected_asr=1.0,
    )
