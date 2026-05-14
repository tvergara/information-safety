"""Tests for invalidate_results.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.invalidate_results import invalidate_rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _row(eval_run_id: str, **kwargs: Any) -> dict[str, Any]:
    return {
        "eval_run_id": eval_run_id,
        "dataset_name": "wmdp",
        "experiment_name": "DataStrategy",
        "max_examples": 16,
        "asr": 0.25,
        **kwargs,
    }


def test_flags_rows_matching_predicate(tmp_path: Path) -> None:
    results_file = tmp_path / "results.jsonl"
    _write_jsonl(results_file, [
        _row("a"),
        _row("b", dataset_name="advbench_harmbench"),
        _row("c", max_examples=None),
    ])

    summary = invalidate_rows(
        results_path=results_file,
        predicate=lambda r: r.get("dataset_name") in {"wmdp", "evilmath"}
        and r.get("max_examples") is not None,
        reason="wmdp_evilmath_eval_truncated_to_max_examples",
    )

    assert summary.flagged == 1
    rows_by_id = {r["eval_run_id"]: r for r in _read_jsonl(results_file)}
    assert rows_by_id["a"]["invalidated_reason"] == "wmdp_evilmath_eval_truncated_to_max_examples"
    assert "invalidated_reason" not in rows_by_id["b"]
    assert "invalidated_reason" not in rows_by_id["c"]


def test_skips_already_flagged_rows(tmp_path: Path) -> None:
    results_file = tmp_path / "results.jsonl"
    _write_jsonl(results_file, [
        _row("a", invalidated_reason="prior_reason"),
        _row("b"),
    ])

    summary = invalidate_rows(
        results_path=results_file,
        predicate=lambda r: True,
        reason="new_reason",
    )

    assert summary.flagged == 1
    assert summary.already_flagged == 1
    rows_by_id = {r["eval_run_id"]: r for r in _read_jsonl(results_file)}
    assert rows_by_id["a"]["invalidated_reason"] == "prior_reason"
    assert rows_by_id["b"]["invalidated_reason"] == "new_reason"


def test_dry_run_does_not_modify_file(tmp_path: Path) -> None:
    results_file = tmp_path / "results.jsonl"
    original = [_row("a"), _row("b")]
    _write_jsonl(results_file, original)

    summary = invalidate_rows(
        results_path=results_file,
        predicate=lambda r: True,
        reason="some_reason",
        dry_run=True,
    )

    assert summary.flagged == 2
    assert _read_jsonl(results_file) == original


def test_writes_backup_before_rewriting(tmp_path: Path) -> None:
    results_file = tmp_path / "results.jsonl"
    _write_jsonl(results_file, [_row("a")])

    invalidate_rows(
        results_path=results_file,
        predicate=lambda r: True,
        reason="some_reason",
    )

    assert (tmp_path / "results.jsonl.bak").exists()


def test_no_matches_does_not_rewrite(tmp_path: Path) -> None:
    results_file = tmp_path / "results.jsonl"
    _write_jsonl(results_file, [_row("a")])
    mtime_before = results_file.stat().st_mtime_ns

    summary = invalidate_rows(
        results_path=results_file,
        predicate=lambda r: False,
        reason="some_reason",
    )

    assert summary.flagged == 0
    assert results_file.stat().st_mtime_ns == mtime_before
    assert not (tmp_path / "results.jsonl.bak").exists()
