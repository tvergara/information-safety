"""Worker for the cluster job pool.

One worker per GPU. Each instance atomically claims a job file from
``<queue_root>/pending/`` via ``os.rename`` and runs the command in a
subprocess pinned to ``CUDA_VISIBLE_DEVICES=$WORKER_INDEX``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def _ensure_dirs(queue_root: Path) -> None:
    for sub in ("pending", "claimed", "done", "failed", "logs"):
        (queue_root / sub).mkdir(parents=True, exist_ok=True)


def _try_claim(
    pending_path: Path, claimed_dir: Path, worker_index: int, pool_id: str
) -> Path | None:
    job_id = pending_path.stem
    target = claimed_dir / f"{job_id}.{pool_id}.{worker_index}.json"
    try:
        os.rename(pending_path, target)
    except FileNotFoundError:
        return None
    return target


def _build_env(worker_index: int) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(worker_index)
    return env


def _run_one(
    claimed_path: Path,
    *,
    queue_root: Path,
    worker_index: int,
) -> int:
    payload = json.loads(claimed_path.read_text())
    job_id = payload["id"]
    command = payload["command"]
    log_path = queue_root / "logs" / f"{job_id}.{worker_index}.log"
    with open(log_path, "w") as log_fp:
        result = subprocess.run(
            command,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            env=_build_env(worker_index),
            check=False,
        )
    target_dir = queue_root / ("done" if result.returncode == 0 else "failed")
    final_path = target_dir / f"{job_id}.json"
    os.rename(claimed_path, final_path)
    return result.returncode


def run_worker(*, queue_root: Path, worker_index: int) -> None:
    """Drain the pending queue, claiming and running one job at a time."""
    _ensure_dirs(queue_root)
    pending_dir = queue_root / "pending"
    claimed_dir = queue_root / "claimed"
    pool_id = os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"

    while True:
        candidates = sorted(pending_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            return
        claimed: Path | None = None
        for cand in candidates:
            claimed = _try_claim(cand, claimed_dir, worker_index, pool_id)
            if claimed is not None:
                break
        if claimed is None:
            time.sleep(0.05)
            continue
        _run_one(claimed, queue_root=queue_root, worker_index=worker_index)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cluster job-pool worker")
    parser.add_argument("--queue-root", type=Path, default=None)
    parser.add_argument("--worker-index", type=int, default=None)
    args = parser.parse_args(argv)

    queue_root = args.queue_root or Path(os.environ["QUEUE_ROOT"])
    worker_index = (
        args.worker_index
        if args.worker_index is not None
        else int(os.environ["WORKER_INDEX"])
    )
    run_worker(queue_root=queue_root, worker_index=worker_index)


if __name__ == "__main__":
    main()
