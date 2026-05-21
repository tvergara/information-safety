"""Tests for null_asr.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.null_asr import null_asr_rows


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
        "asr": 0.25,
        **kwargs,
    }


def test_nulls_asr_on_matching_rows(tmp_path: Path) -> None:
    results_file = tmp_path / "results.jsonl"
    _write_jsonl(results_file, [
        _row("a"),
        _row("b", dataset_name="advbench_harmbench"),
        _row("c"),
    ])

    summary = null_asr_rows(
        results_path=results_file,
        predicate=lambda r: r.get("dataset_name") == "wmdp",
    )

    assert summary.nulled == 2
    rows_by_id = {r["eval_run_id"]: r for r in _read_jsonl(results_file)}
    assert rows_by_id["a"]["asr"] is None
    assert rows_by_id["b"]["asr"] == 0.25
    assert rows_by_id["c"]["asr"] is None


def test_skips_already_null_rows(tmp_path: Path) -> None:
    results_file = tmp_path / "results.jsonl"
    _write_jsonl(results_file, [
        _row("a", asr=None),
        _row("b"),
    ])

    summary = null_asr_rows(
        results_path=results_file,
        predicate=lambda r: True,
    )

    assert summary.nulled == 1
    assert summary.already_null == 1


def test_skips_invalidated_rows(tmp_path: Path) -> None:
    results_file = tmp_path / "results.jsonl"
    _write_jsonl(results_file, [
        _row("a", invalidated_reason="prior_reason"),
        _row("b"),
    ])

    summary = null_asr_rows(
        results_path=results_file,
        predicate=lambda r: True,
    )

    assert summary.nulled == 1
    assert summary.skipped_invalidated == 1
    rows_by_id = {r["eval_run_id"]: r for r in _read_jsonl(results_file)}
    assert rows_by_id["a"]["asr"] == 0.25
    assert rows_by_id["b"]["asr"] is None


def test_dry_run_does_not_modify_file(tmp_path: Path) -> None:
    results_file = tmp_path / "results.jsonl"
    original = [_row("a"), _row("b")]
    _write_jsonl(results_file, original)

    summary = null_asr_rows(
        results_path=results_file,
        predicate=lambda r: True,
        dry_run=True,
    )

    assert summary.nulled == 2
    assert _read_jsonl(results_file) == original


def test_writes_backup_before_rewriting(tmp_path: Path) -> None:
    results_file = tmp_path / "results.jsonl"
    _write_jsonl(results_file, [_row("a")])

    null_asr_rows(
        results_path=results_file,
        predicate=lambda r: True,
    )

    assert (tmp_path / "results.jsonl.bak").exists()


def test_no_matches_does_not_rewrite(tmp_path: Path) -> None:
    results_file = tmp_path / "results.jsonl"
    _write_jsonl(results_file, [_row("a")])
    mtime_before = results_file.stat().st_mtime_ns

    summary = null_asr_rows(
        results_path=results_file,
        predicate=lambda r: False,
    )

    assert summary.nulled == 0
    assert results_file.stat().st_mtime_ns == mtime_before
    assert not (tmp_path / "results.jsonl.bak").exists()
