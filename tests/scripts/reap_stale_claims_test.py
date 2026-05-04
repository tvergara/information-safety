from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from scripts.reap_stale_claims import main, reap


def _make_queue(tmp_path: Path) -> Path:
    queue_root = tmp_path / "queue"
    for sub in ("pending", "claimed", "done", "failed", "logs"):
        (queue_root / sub).mkdir(parents=True, exist_ok=True)
    return queue_root


def _seed_claim(
    queue_root: Path,
    *,
    job_id: str,
    pool_id: str,
    worker_index: int,
    attempts: int = 0,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "id": job_id,
        "command": ["python", "-c", "pass"],
        "config": {"k": "v"},
        "attempts": attempts,
    }
    if extra:
        payload.update(extra)
    path = queue_root / "claimed" / f"{job_id}.{pool_id}.{worker_index}.json"
    path.write_text(json.dumps(payload))
    return path


def _stub_squeue(live_ids: set[str]) -> subprocess.CompletedProcess[str]:
    stdout = "\n".join(sorted(live_ids)) + ("\n" if live_ids else "")
    return subprocess.CompletedProcess(
        args=["squeue"], returncode=0, stdout=stdout, stderr=""
    )


def test_reaps_dead_pool_claim_to_pending(tmp_path: Path) -> None:
    queue_root = _make_queue(tmp_path)
    claim = _seed_claim(queue_root, job_id="abc", pool_id="123", worker_index=0)

    with mock.patch(
        "scripts.reap_stale_claims.subprocess.run",
        return_value=_stub_squeue(set()),
    ):
        reap(queue_root=queue_root, max_attempts=3)

    assert not claim.exists()
    pending_path = queue_root / "pending" / "abc.json"
    assert pending_path.exists()
    payload = json.loads(pending_path.read_text())
    assert payload["attempts"] == 1
    assert payload["id"] == "abc"


def test_skips_live_pool_claim(tmp_path: Path) -> None:
    queue_root = _make_queue(tmp_path)
    claim = _seed_claim(queue_root, job_id="abc", pool_id="123", worker_index=0)

    with mock.patch(
        "scripts.reap_stale_claims.subprocess.run",
        return_value=_stub_squeue({"123"}),
    ):
        reap(queue_root=queue_root, max_attempts=3)

    assert claim.exists()
    assert list((queue_root / "pending").iterdir()) == []
    assert list((queue_root / "failed").iterdir()) == []


def test_increments_attempts_each_reap(tmp_path: Path) -> None:
    queue_root = _make_queue(tmp_path)
    _seed_claim(
        queue_root, job_id="abc", pool_id="123", worker_index=0, attempts=1
    )

    with mock.patch(
        "scripts.reap_stale_claims.subprocess.run",
        return_value=_stub_squeue(set()),
    ):
        reap(queue_root=queue_root, max_attempts=3)

    payload = json.loads((queue_root / "pending" / "abc.json").read_text())
    assert payload["attempts"] == 2


def test_moves_to_failed_after_max_attempts(tmp_path: Path) -> None:
    queue_root = _make_queue(tmp_path)
    _seed_claim(
        queue_root, job_id="abc", pool_id="123", worker_index=0, attempts=2
    )

    with mock.patch(
        "scripts.reap_stale_claims.subprocess.run",
        return_value=_stub_squeue(set()),
    ):
        reap(queue_root=queue_root, max_attempts=3)

    assert list((queue_root / "pending").iterdir()) == []
    assert list((queue_root / "claimed").iterdir()) == []
    failed_path = queue_root / "failed" / "abc.json"
    assert failed_path.exists()
    payload = json.loads(failed_path.read_text())
    assert payload["attempts"] == 3
    assert payload["failure_reason"] == "exceeded retries"


def test_handles_mixed_live_and_dead_claims(tmp_path: Path) -> None:
    queue_root = _make_queue(tmp_path)
    _seed_claim(queue_root, job_id="aaa", pool_id="111", worker_index=0)
    _seed_claim(queue_root, job_id="bbb", pool_id="222", worker_index=0)
    _seed_claim(queue_root, job_id="ccc", pool_id="222", worker_index=1)

    with mock.patch(
        "scripts.reap_stale_claims.subprocess.run",
        return_value=_stub_squeue({"111"}),
    ):
        reap(queue_root=queue_root, max_attempts=3)

    claimed_remaining = sorted(p.name for p in (queue_root / "claimed").iterdir())
    assert claimed_remaining == ["aaa.111.0.json"]
    pending_files = sorted(p.name for p in (queue_root / "pending").iterdir())
    assert pending_files == ["bbb.json", "ccc.json"]


def test_treats_local_pool_id_as_alive(tmp_path: Path) -> None:
    queue_root = _make_queue(tmp_path)
    claim = _seed_claim(
        queue_root, job_id="abc", pool_id="local-9999", worker_index=0
    )

    with mock.patch(
        "scripts.reap_stale_claims.subprocess.run",
        return_value=_stub_squeue(set()),
    ):
        reap(queue_root=queue_root, max_attempts=3)

    assert claim.exists()
    assert list((queue_root / "pending").iterdir()) == []
    assert list((queue_root / "failed").iterdir()) == []


def test_atomic_write_on_failure(tmp_path: Path) -> None:
    queue_root = _make_queue(tmp_path)
    claim = _seed_claim(queue_root, job_id="abc", pool_id="123", worker_index=0)

    with mock.patch(
        "scripts.reap_stale_claims.subprocess.run",
        return_value=_stub_squeue(set()),
    ), mock.patch(
        "scripts.reap_stale_claims.os.replace",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(OSError):
            reap(queue_root=queue_root, max_attempts=3)

    assert claim.exists()
    pending_dest = queue_root / "pending" / "abc.json"
    assert not pending_dest.exists()
    failed_dest = queue_root / "failed" / "abc.json"
    assert not failed_dest.exists()


def test_squeue_failure_propagates_without_touching_claims(tmp_path: Path) -> None:
    queue_root = _make_queue(tmp_path)
    claim = _seed_claim(queue_root, job_id="abc", pool_id="123", worker_index=0)

    with mock.patch(
        "scripts.reap_stale_claims.subprocess.run",
        side_effect=FileNotFoundError("squeue not found"),
    ):
        with pytest.raises(FileNotFoundError):
            main(["--queue-root", str(queue_root)])

    assert claim.exists()


def test_idempotent_when_pool_recovers(tmp_path: Path) -> None:
    queue_root = _make_queue(tmp_path)
    claim = _seed_claim(queue_root, job_id="abc", pool_id="123", worker_index=0)

    with mock.patch(
        "scripts.reap_stale_claims.subprocess.run",
        return_value=_stub_squeue({"123"}),
    ):
        reap(queue_root=queue_root, max_attempts=3)
        reap(queue_root=queue_root, max_attempts=3)

    assert claim.exists()
    assert list((queue_root / "pending").iterdir()) == []
    assert list((queue_root / "failed").iterdir()) == []


def test_summary_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    queue_root = _make_queue(tmp_path)
    _seed_claim(queue_root, job_id="aaa", pool_id="111", worker_index=0, attempts=0)
    _seed_claim(queue_root, job_id="bbb", pool_id="222", worker_index=0, attempts=2)
    _seed_claim(queue_root, job_id="ccc", pool_id="333", worker_index=0, attempts=0)

    with mock.patch(
        "scripts.reap_stale_claims.subprocess.run",
        return_value=_stub_squeue({"333"}),
    ):
        reap(queue_root=queue_root, max_attempts=3)

    captured = capsys.readouterr()
    assert "1 reaped to pending" in captured.out
    assert "1 moved to failed" in captured.out
    assert "1 skipped (live)" in captured.out
