"""Tests for the AdversariaLLM GCG attack-output converter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.extract_adversariallm_attacks import convert_run_dir


def _write_run_json(path: Path, behavior: str, steps: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"attack_params": {"name": "gcg"}},
        "runs": [
            {
                "original_prompt": [
                    {"role": "user", "content": behavior},
                    {"role": "assistant", "content": "target_A"},
                ],
                "steps": steps,
                "total_time": 0.0,
            }
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def test_convert_run_dir_emits_one_row_per_behavior(tmp_path: Path) -> None:
    run_dir = tmp_path / "adv_out"
    _write_run_json(
        run_dir / "0" / "run.json",
        behavior="beh_A",
        steps=[
            {
                "step": 0,
                "loss": 2.0,
                "model_input": [
                    {"role": "user", "content": "beh_A suffix_init"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
            {
                "step": 1,
                "loss": 0.5,
                "model_input": [
                    {"role": "user", "content": "beh_A suffix_opt"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
        ],
    )
    _write_run_json(
        run_dir / "1" / "run.json",
        behavior="beh_B",
        steps=[
            {
                "step": 0,
                "loss": 1.5,
                "model_input": [
                    {"role": "user", "content": "beh_B suffix_B"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
        ],
    )

    out_file = tmp_path / "out.jsonl"
    convert_run_dir(str(run_dir), str(out_file))

    with open(out_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 2
    behaviors = {r["behavior"] for r in rows}
    assert behaviors == {"beh_A", "beh_B"}
    for row in rows:
        assert isinstance(row["adversarial_prompt"], str)
        assert "behavior" in row and "adversarial_prompt" in row


def test_convert_run_dir_selects_lowest_loss_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "adv_out"
    _write_run_json(
        run_dir / "0" / "run.json",
        behavior="beh_A",
        steps=[
            {
                "step": 0,
                "loss": 3.0,
                "model_input": [
                    {"role": "user", "content": "beh_A worst"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
            {
                "step": 1,
                "loss": 0.1,
                "model_input": [
                    {"role": "user", "content": "beh_A best"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
            {
                "step": 2,
                "loss": 0.5,
                "model_input": [
                    {"role": "user", "content": "beh_A middle"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
        ],
    )

    out_file = tmp_path / "out.jsonl"
    convert_run_dir(str(run_dir), str(out_file))

    with open(out_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
    assert rows[0]["adversarial_prompt"] == "beh_A best"


def test_convert_run_dir_raises_on_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "adv_out"
    run_dir.mkdir()

    out_file = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="No run.json"):
        convert_run_dir(str(run_dir), str(out_file))


def test_convert_run_dir_raises_on_duplicate_behavior(tmp_path: Path) -> None:
    run_dir = tmp_path / "adv_out"
    for subdir in ("0", "1"):
        _write_run_json(
            run_dir / subdir / "run.json",
            behavior="beh_A",
            steps=[
                {
                    "step": 0,
                    "loss": 1.0,
                    "model_input": [
                        {"role": "user", "content": "beh_A suffix"},
                        {"role": "assistant", "content": ""},
                    ],
                    "model_completions": ["resp"],
                },
            ],
        )

    out_file = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="duplicate behavior"):
        convert_run_dir(str(run_dir), str(out_file))
