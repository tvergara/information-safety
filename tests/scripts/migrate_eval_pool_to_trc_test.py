"""Tests for scripts/migrate_eval_pool_to_trc.py — TRC eval-pool migrator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts import migrate_eval_pool_to_trc


def _write_pending_spec(
    pending_dir: Path, *, spec_id: str = "abc1234567890def", epoch: int = 0,
) -> None:
    pending_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_id": spec_id,
        "epoch": epoch,
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "adapter_path": f"/work/adapters/{spec_id}/epoch_{epoch}",
        "eval_meta": {"spec_id": spec_id, "epoch": epoch},
    }
    (pending_dir / f"{spec_id}_ep{epoch}.json").write_text(json.dumps(payload))


def test_eval_job_name_format() -> None:
    assert migrate_eval_pool_to_trc.eval_job_name("abc", epoch=2) == "is_eval_abc_2"


def test_dry_run_emits_one_line_per_spec_and_does_not_call_ssh(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    queue = tmp_path / "eval-pool"
    _write_pending_spec(queue / "pending", spec_id="aa", epoch=0)
    _write_pending_spec(queue / "pending", spec_id="bb", epoch=1)
    with patch("subprocess.run") as run:
        migrate_eval_pool_to_trc.main([
            "--queue-root", str(queue),
            "--count", "10",
            "--state-file", str(tmp_path / "state.jsonl"),
            "--dry-run",
        ])
    run.assert_not_called()
    out = capsys.readouterr().out
    assert out.count("eai job submit") == 2


def test_count_caps_submissions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    queue = tmp_path / "eval-pool"
    for i in range(5):
        _write_pending_spec(queue / "pending", spec_id=f"id{i}", epoch=0)
    migrate_eval_pool_to_trc.main([
        "--queue-root", str(queue),
        "--count", "2",
        "--state-file", str(tmp_path / "state.jsonl"),
        "--dry-run",
    ])
    err = capsys.readouterr().err
    assert "Submitted: 2" in err


def test_submit_appends_to_state_file(tmp_path: Path) -> None:
    queue = tmp_path / "eval-pool"
    _write_pending_spec(queue / "pending", spec_id="abcd1234abcd1234", epoch=3)
    state = tmp_path / "state.jsonl"
    fake = MagicMock()
    fake.stdout = "uuid 12345678-1234-1234-1234-1234567890ab\n"
    with patch("subprocess.run", return_value=fake):
        migrate_eval_pool_to_trc.main([
            "--queue-root", str(queue),
            "--count", "10",
            "--state-file", str(state),
        ])
    rows = [json.loads(line) for line in state.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["spec_id"] == "abcd1234abcd1234"
    assert rows[0]["epoch"] == 3
    assert rows[0]["job_id"] == "is_eval_abcd1234abcd1234_3"
    assert rows[0]["eai_uuid"] == "12345678-1234-1234-1234-1234567890ab"


def test_no_pending_specs_submits_nothing(tmp_path: Path) -> None:
    queue = tmp_path / "eval-pool"
    queue.mkdir()
    with patch("subprocess.run") as run:
        migrate_eval_pool_to_trc.main([
            "--queue-root", str(queue),
            "--count", "10",
            "--state-file", str(tmp_path / "state.jsonl"),
        ])
    run.assert_not_called()
