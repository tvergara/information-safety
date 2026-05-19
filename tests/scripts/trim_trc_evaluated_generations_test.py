"""Tests for scripts/trim_trc_evaluated_generations.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.trim_trc_evaluated_generations import (
    TRC_GENERATIONS_DIR,
    collect_trimmable_eval_run_ids,
    main,
)


def _write_results(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_collects_eval_run_ids_with_non_null_asr(tmp_path: Path) -> None:
    results = tmp_path / "final-results.jsonl"
    _write_results(results, [
        {"eval_run_id": "run-1", "asr": 0.42},
        {"eval_run_id": "run-2", "asr": None},
        {"eval_run_id": "run-3", "asr": 0.0},
    ])
    assert collect_trimmable_eval_run_ids(results) == {"run-1", "run-3"}


def test_skips_invalidated_rows(tmp_path: Path) -> None:
    results = tmp_path / "final-results.jsonl"
    _write_results(results, [
        {"eval_run_id": "run-good", "asr": 0.5},
        {
            "eval_run_id": "run-bad", "asr": 0.5,
            "invalidated_reason": "wrong-classifier",
        },
    ])
    assert collect_trimmable_eval_run_ids(results) == {"run-good"}


def test_dedups_eval_run_ids(tmp_path: Path) -> None:
    """Deferred-eval epochs share an eval_run_id across rows."""
    results = tmp_path / "final-results.jsonl"
    _write_results(results, [
        {"eval_run_id": "run-1", "asr": 0.1},
        {"eval_run_id": "run-1", "asr": 0.2},
    ])
    assert collect_trimmable_eval_run_ids(results) == {"run-1"}


def test_rejects_eval_run_id_with_unsafe_chars(tmp_path: Path) -> None:
    """Path-escaping ids must never reach the ssh rm command."""
    results = tmp_path / "final-results.jsonl"
    _write_results(results, [{"eval_run_id": "../escape", "asr": 0.5}])
    with pytest.raises(ValueError, match="unsafe"):
        collect_trimmable_eval_run_ids(results)


def test_dry_run_does_not_invoke_ssh(tmp_path: Path) -> None:
    results = tmp_path / "final-results.jsonl"
    _write_results(results, [{"eval_run_id": "run-1", "asr": 0.5}])
    with patch("scripts.trim_trc_evaluated_generations.subprocess.run") as run:
        main(["--results-file", str(results)])
    assert run.call_count == 0


def test_apply_invokes_single_batched_ssh_rm(tmp_path: Path) -> None:
    results = tmp_path / "final-results.jsonl"
    _write_results(results, [
        {"eval_run_id": f"run-{i}", "asr": 0.5} for i in range(5)
    ])
    with patch("scripts.trim_trc_evaluated_generations.subprocess.run") as run:
        main(["--results-file", str(results), "--apply"])
    assert run.call_count == 1
    argv = run.call_args[0][0]
    assert argv[:2] == ["ssh", "trc"]
    assert argv[2].startswith("rm -rf ")
    for i in range(5):
        assert f"{TRC_GENERATIONS_DIR}/run-{i}" in argv[2]
