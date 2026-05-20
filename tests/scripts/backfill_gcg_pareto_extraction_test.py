"""Tests for scripts/backfill_gcg_pareto_extraction.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import extract_adversariallm_attacks
from scripts.backfill_gcg_pareto_extraction import backfill_root


@pytest.fixture(autouse=True)
def _relax_min_attack_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extract_adversariallm_attacks, "MIN_ATTACK_STEPS", 1)

_GCG_CONFIG: dict[str, object] = {
    "num_steps": 10,
    "topk": 8,
    "optim_str_init": "x x x x",
}
_AUTODAN_CONFIG: dict[str, object] = {
    "num_steps": 5,
    "batch_size": 4,
    "mutation": 0.25,
}


def _gcg_step(idx: int, loss: float, flops: int, prompt: str) -> dict:
    return {
        "step": idx,
        "loss": loss,
        "flops": flops,
        "model_input": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},
        ],
        "model_completions": [f"c_{prompt}"],
    }


def _write_run_json(
    path: Path,
    behavior: str,
    steps: list[dict],
    attack_params: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": {
            "attack_params": attack_params if attack_params else dict(_GCG_CONFIG),
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_gcg_layout(adv_root: Path, base_name: str) -> Path:
    base = adv_root / base_name
    run_dir = base / "2026-01-15" / "12-30-45"
    _write_run_json(
        run_dir / "0" / "run.json",
        behavior="beh_A",
        steps=[
            _gcg_step(0, 2.0, 10, "p0"),
            _gcg_step(1, 0.5, 20, "p1"),
            _gcg_step(2, 0.2, 30, "p2"),
        ],
    )
    return base


def _make_autodan_layout(adv_root: Path, base_name: str) -> Path:
    base = adv_root / base_name
    run_dir = base / "2026-01-15" / "12-30-45"
    _write_run_json(
        run_dir / "0" / "run.json",
        behavior="beh_A",
        steps=[
            _gcg_step(0, 2.0, 10, "p0"),
            _gcg_step(1, 0.5, 30, "p1"),
            _gcg_step(2, 0.2, 60, "p2"),
        ],
        attack_params=dict(_AUTODAN_CONFIG),
    )
    return base


def test_backfill_writes_gcg_jsonl_to_attacks_dir(tmp_path: Path) -> None:
    adv_root = tmp_path / "adversariallm-outputs"
    attacks_dir = tmp_path / "attacks"
    _make_gcg_layout(adv_root, "gcg-model-data")

    n_touched = backfill_root(
        adv_save_dir_root=adv_root, attacks_dir=attacks_dir, dry_run=False
    )
    assert n_touched == 1

    jsonl_path = attacks_dir / "gcg-model-data.jsonl"
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert len(rows) == 3
    by_idx = {r["pareto_step_idx"]: r for r in rows}
    assert set(by_idx) == {0, 1, 2}
    zero = by_idx[0]
    assert zero["adversarial_prompt"] == "p2"
    assert zero["attack_flops"] == 60


def test_backfill_is_idempotent(tmp_path: Path) -> None:
    adv_root = tmp_path / "adversariallm-outputs"
    attacks_dir = tmp_path / "attacks"
    _make_gcg_layout(adv_root, "gcg-model-data")

    backfill_root(adv_save_dir_root=adv_root, attacks_dir=attacks_dir, dry_run=False)
    first = (attacks_dir / "gcg-model-data.jsonl").read_text()
    backfill_root(adv_save_dir_root=adv_root, attacks_dir=attacks_dir, dry_run=False)
    second = (attacks_dir / "gcg-model-data.jsonl").read_text()
    assert first == second


def test_backfill_emits_autodan_trajectory_rows(tmp_path: Path) -> None:
    adv_root = tmp_path / "adversariallm-outputs"
    attacks_dir = tmp_path / "attacks"
    _make_autodan_layout(adv_root, "autodan-model-data")

    backfill_root(adv_save_dir_root=adv_root, attacks_dir=attacks_dir, dry_run=False)

    jsonl_path = attacks_dir / "autodan-model-data.jsonl"
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert len(rows) == 3
    by_idx = {r["pareto_step_idx"]: r for r in rows}
    assert set(by_idx) == {0, 1, 2}
    zero = by_idx[0]
    assert zero["adversarial_prompt"] == "p2"
    assert zero["attack_flops"] == 60


def test_backfill_dry_run_writes_nothing(tmp_path: Path) -> None:
    adv_root = tmp_path / "adversariallm-outputs"
    attacks_dir = tmp_path / "attacks"
    _make_gcg_layout(adv_root, "gcg-model-data")
    existing_jsonl = attacks_dir / "gcg-model-data.jsonl"
    _write_jsonl(existing_jsonl, [{"behavior": "old", "x": 1}])

    before = existing_jsonl.read_text()
    backfill_root(adv_save_dir_root=adv_root, attacks_dir=attacks_dir, dry_run=True)
    after = existing_jsonl.read_text()
    assert before == after


def test_backfill_does_not_touch_final_results_or_generations(tmp_path: Path) -> None:
    adv_root = tmp_path / "adversariallm-outputs"
    attacks_dir = tmp_path / "attacks"
    _make_gcg_layout(adv_root, "gcg-model-data")
    results_file = tmp_path / "final-results.jsonl"
    results_file.write_text(json.dumps({"asr": 0.5, "eval_run_id": "abc"}) + "\n")
    gens_dir = tmp_path / "generations"
    gens_dir.mkdir()
    (gens_dir / "abc").mkdir()
    (gens_dir / "abc" / "responses.jsonl").write_text("placeholder\n")

    results_before = results_file.read_text()
    gens_before = (gens_dir / "abc" / "responses.jsonl").read_text()

    backfill_root(adv_save_dir_root=adv_root, attacks_dir=attacks_dir, dry_run=False)

    assert results_file.read_text() == results_before
    assert (gens_dir / "abc" / "responses.jsonl").read_text() == gens_before


def test_backfill_discovers_hydra_timestamped_layout(tmp_path: Path) -> None:
    adv_root = tmp_path / "adversariallm-outputs"
    attacks_dir = tmp_path / "attacks"
    base = adv_root / "gcg-model-data"
    _write_run_json(
        base / "2026-01-15" / "12-30-45" / "0" / "run.json",
        behavior="beh_A",
        steps=[_gcg_step(0, 2.0, 10, "p0"), _gcg_step(1, 0.2, 30, "p2")],
    )
    _write_run_json(
        base / "2026-01-15" / "13-00-00" / "0" / "run.json",
        behavior="beh_B",
        steps=[_gcg_step(0, 1.5, 10, "q0"), _gcg_step(1, 0.3, 30, "q1")],
    )

    n_touched = backfill_root(
        adv_save_dir_root=adv_root, attacks_dir=attacks_dir, dry_run=False
    )
    assert n_touched == 1

    rows = [
        json.loads(line)
        for line in (attacks_dir / "gcg-model-data.jsonl").read_text().splitlines()
    ]
    behaviors = {r["behavior"] for r in rows}
    assert behaviors == {"beh_A", "beh_B"}
    assert len(rows) == 4


def test_backfill_overwrites_pre_existing_attacks_jsonl(tmp_path: Path) -> None:
    adv_root = tmp_path / "adversariallm-outputs"
    attacks_dir = tmp_path / "attacks"
    _make_gcg_layout(adv_root, "gcg-model-data")
    stale_jsonl = attacks_dir / "gcg-model-data.jsonl"
    _write_jsonl(stale_jsonl, [{"behavior": "beh_A", "pareto_step_idx": 0, "stale": True}])

    backfill_root(adv_save_dir_root=adv_root, attacks_dir=attacks_dir, dry_run=False)

    rows = [json.loads(line) for line in stale_jsonl.read_text().splitlines()]
    assert len(rows) == 3
    assert all("stale" not in r for r in rows)
