"""Tests for scripts/trim_trc_evaluated_generations.py."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from scripts._trc_common import TRC_DATA_MOUNT
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


def _fake_run_factory(
    submit_stdout: str = "Submitted 7dd2cddd-a165-4d2a-a5c4-b77509932509\n",
    info_state: str = "SUCCEEDED",
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "eai job submit" in argv[2]:
            return subprocess.CompletedProcess(argv, 0, stdout=submit_stdout, stderr="")
        if "eai job info" in argv[2]:
            return subprocess.CompletedProcess(
                argv, 0, stdout=f"state: {info_state}\n", stderr=""
            )
        raise AssertionError(f"unexpected argv: {argv!r}")
    return fake_run


def test_apply_submits_eai_job_with_work_mount(tmp_path: Path) -> None:
    """Rm must run inside an eai job (where /work is mounted), not via ssh login."""
    results = tmp_path / "final-results.jsonl"
    _write_results(results, [
        {"eval_run_id": f"run-{i}", "asr": 0.5} for i in range(3)
    ])
    with patch(
        "scripts.trim_trc_evaluated_generations.subprocess.run",
        side_effect=_fake_run_factory(),
    ) as run, patch("scripts.trim_trc_evaluated_generations.time.sleep"):
        main(["--results-file", str(results), "--apply"])

    submit_argv = run.call_args_list[0][0][0]
    assert submit_argv[:2] == ["ssh", "trc"]
    assert "eai job submit" in submit_argv[2]
    assert f"--data {TRC_DATA_MOUNT}" in submit_argv[2]
    assert "rm -rf" in submit_argv[2]
    for i in range(3):
        assert f"{TRC_GENERATIONS_DIR}/run-{i}" in submit_argv[2]

    info_argv = run.call_args_list[1][0][0]
    assert info_argv[:2] == ["ssh", "trc"]
    assert "eai job info 7dd2cddd-a165-4d2a-a5c4-b77509932509" in info_argv[2]


def test_apply_batches_into_multiple_eai_jobs(tmp_path: Path) -> None:
    """Ssh exec hits E2BIG for one giant rm; submit-and-wait batches around it."""
    results = tmp_path / "final-results.jsonl"
    _write_results(results, [
        {"eval_run_id": f"run-{i:04d}", "asr": 0.5} for i in range(5)
    ])
    submit_counter = iter(range(100))

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "eai job submit" in argv[2]:
            i = next(submit_counter)
            return subprocess.CompletedProcess(
                argv, 0,
                stdout=f"Submitted 00000000-0000-0000-0000-{i:012d}\n",
                stderr="",
            )
        return subprocess.CompletedProcess(argv, 0, stdout="state: SUCCEEDED\n", stderr="")

    with patch("scripts.trim_trc_evaluated_generations._BATCH_SIZE", 2), \
         patch("scripts.trim_trc_evaluated_generations.subprocess.run",
               side_effect=fake_run) as run, \
         patch("scripts.trim_trc_evaluated_generations.time.sleep"):
        main(["--results-file", str(results), "--apply"])

    submit_calls = [c for c in run.call_args_list if "eai job submit" in c[0][0][2]]
    assert len(submit_calls) == 3


def test_apply_raises_when_eai_job_fails(tmp_path: Path) -> None:
    results = tmp_path / "final-results.jsonl"
    _write_results(results, [{"eval_run_id": "run-1", "asr": 0.5}])
    with patch(
        "scripts.trim_trc_evaluated_generations.subprocess.run",
        side_effect=_fake_run_factory(info_state="FAILED"),
    ), patch("scripts.trim_trc_evaluated_generations.time.sleep"):
        with pytest.raises(RuntimeError, match="FAILED"):
            main(["--results-file", str(results), "--apply"])
