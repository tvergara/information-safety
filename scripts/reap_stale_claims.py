"""Reap stale cluster job-pool claims back to pending."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_MAX_ATTEMPTS = 3


def _live_pool_ids() -> set[str]:
    user = os.environ["USER"]
    result = subprocess.run(
        ["squeue", "-u", user, "-h", "-O", "JobID", "--noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _atomic_write(dest: Path, payload: dict[str, Any]) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=str))
    os.replace(tmp, dest)


def reap(*, queue_root: Path, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> dict[str, int]:
    claimed_dir = queue_root / "claimed"
    pending_dir = queue_root / "pending"
    failed_dir = queue_root / "failed"
    for d in (pending_dir, failed_dir):
        d.mkdir(parents=True, exist_ok=True)

    live = _live_pool_ids()

    counts = {"reaped": 0, "failed": 0, "skipped": 0}
    for claim in sorted(claimed_dir.glob("*.json")):
        job_id, pool_id, _ = claim.stem.split(".")
        if pool_id.startswith("local-") or pool_id in live:
            counts["skipped"] += 1
            continue
        payload = json.loads(claim.read_text())
        payload["attempts"] = payload["attempts"] + 1
        if payload["attempts"] >= max_attempts:
            payload["failure_reason"] = "exceeded retries"
            _atomic_write(failed_dir / f"{job_id}.json", payload)
            counts["failed"] += 1
        else:
            _atomic_write(pending_dir / f"{job_id}.json", payload)
            counts["reaped"] += 1
        os.remove(claim)

    print(
        f"{counts['reaped']} reaped to pending | "
        f"{counts['failed']} moved to failed | "
        f"{counts['skipped']} skipped (live)"
    )
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reap stale cluster job-pool claims")
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    args = parser.parse_args(argv)
    reap(queue_root=args.queue_root, max_attempts=args.max_attempts)


if __name__ == "__main__":
    main()
