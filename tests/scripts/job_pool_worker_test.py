"""Tests for the cluster job-pool worker."""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import IO, cast
from unittest import mock

from scripts.job_pool_worker import run_worker


def _seed_pending(queue_root: Path, n_jobs: int) -> list[str]:
    pending = queue_root / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    ids = []
    for i in range(n_jobs):
        job_id = f"job{i:03d}"
        ids.append(job_id)
        payload = {
            "id": job_id,
            "command": ["python", "-c", "pass"],
            "config": {"i": i},
        }
        (pending / f"{job_id}.json").write_text(json.dumps(payload))
    return ids


class _FakeCompleted:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def _success_run(*args: object, **kwargs: object) -> _FakeCompleted:
    cast(IO[str], kwargs["stdout"]).write("ok\n")
    return _FakeCompleted(0)


def _fail_run(*args: object, **kwargs: object) -> _FakeCompleted:
    cast(IO[str], kwargs["stdout"]).write("boom\n")
    return _FakeCompleted(1)


def test_worker_processes_all_pending_to_done(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    _seed_pending(queue_root, n_jobs=3)
    with mock.patch("subprocess.run", side_effect=_success_run):
        run_worker(queue_root=queue_root, worker_index=0)
    assert sorted((queue_root / "done").iterdir()) != []
    assert len(list((queue_root / "pending").iterdir())) == 0
    assert len(list((queue_root / "done").iterdir())) == 3


def test_worker_routes_failed_jobs_to_failed(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    _seed_pending(queue_root, n_jobs=2)
    with mock.patch("subprocess.run", side_effect=_fail_run):
        run_worker(queue_root=queue_root, worker_index=0)
    assert len(list((queue_root / "failed").iterdir())) == 2
    assert len(list((queue_root / "done").iterdir())) == 0


def test_worker_preserves_logs(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    _seed_pending(queue_root, n_jobs=1)
    with mock.patch("subprocess.run", side_effect=_fail_run):
        run_worker(queue_root=queue_root, worker_index=2)
    log_files = list((queue_root / "logs").iterdir())
    assert len(log_files) == 1
    assert ".2.log" in log_files[0].name
    contents = log_files[0].read_text()
    assert "boom" in contents


def test_worker_sets_cuda_visible_devices(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    _seed_pending(queue_root, n_jobs=1)
    captured: dict[str, object] = {}

    def _capture_run(*args: object, **kwargs: object) -> _FakeCompleted:
        captured["env"] = kwargs.get("env")
        return _FakeCompleted(0)

    with mock.patch("subprocess.run", side_effect=_capture_run):
        run_worker(queue_root=queue_root, worker_index=3)
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["CUDA_VISIBLE_DEVICES"] == "3"


def _worker_entrypoint(queue_root: str, worker_index: int) -> None:
    """Top-level so it's picklable for multiprocessing."""
    with mock.patch("subprocess.run", side_effect=_success_run):
        run_worker(queue_root=Path(queue_root), worker_index=worker_index)


def test_no_double_claims_under_concurrency(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    n_jobs = 32
    _seed_pending(queue_root, n_jobs=n_jobs)
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker_entrypoint, args=(str(queue_root), i)) for i in range(4)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    done = list((queue_root / "done").iterdir())
    failed = list((queue_root / "failed").iterdir())
    pending_left = list((queue_root / "pending").iterdir())
    assert len(pending_left) == 0
    assert len(done) + len(failed) == n_jobs
    done_ids = sorted(p.name.split(".")[0] for p in done)
    assert len(done_ids) == len(set(done_ids))


def test_subprocess_run_called_with_command_from_payload(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    _seed_pending(queue_root, n_jobs=1)
    captured: dict[str, object] = {}

    def _capture(*args: object, **kwargs: object) -> _FakeCompleted:
        captured["cmd"] = args[0] if args else kwargs.get("args")
        return _FakeCompleted(0)

    with mock.patch("subprocess.run", side_effect=_capture):
        run_worker(queue_root=queue_root, worker_index=0)
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd == ["python", "-c", "pass"]
