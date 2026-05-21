"""Submit pending DataStrategyDeferredEval evaluations as TRC eai jobs.

Mirrors ``migrate_pool_to_trc.py`` but for the vLLM eval pool: each pending
eval spec under ``/work/eval-pool/<queue-root>/pending/`` is submitted as a
single-GPU eai job that runs ``scripts/run_eval_pool.py`` against that one
spec. Spec naming is ``is_eval_<spec_id>_<epoch>``.

Idempotent via the per-spec ``.json`` file: only specs still present in
``pending/`` get submitted on each call, and the ``--count`` cap matches the
upstream pool migrator so we do not flood TRC.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scripts._trc_common import (
    append_state_row,
    build_eai_submit_remote,
    build_eval_pool_container_cmd,
    default_state_file,
    extract_eai_uuid,
)

__all__ = ["eval_job_name", "main"]

DEFAULT_STATE_FILE = default_state_file("trc-eval-pool-submitted.jsonl")


def eval_job_name(spec_id: str, *, epoch: int) -> str:
    return f"is_eval_{spec_id}_{epoch}"


def _load_pending_specs(queue_root: Path) -> list[dict[str, Any]]:
    pending_dir = queue_root / "pending"
    if not pending_dir.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(pending_dir.glob("*.json"))]


def _build_eai_submit_argv(spec: dict[str, Any], *, queue_root: str) -> list[str]:
    remote = build_eai_submit_remote(
        name=eval_job_name(spec["spec_id"], epoch=spec["epoch"]),
        container_cmd=build_eval_pool_container_cmd(
            base_model=spec["base_model"], queue_root=queue_root,
        ),
        gpu=1,
        cpu=8,
        mem=64,
        max_run_time=21600,
        preemptable=False,
    )
    return ["ssh", "trc", remote]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue-root", required=True,
        help="Eval-pool queue root on TRC, e.g. /work/eval-pool/run-1",
    )
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    queue_root_path = Path(args.queue_root)
    specs = _load_pending_specs(queue_root_path)

    submitted = 0
    for spec in specs:
        if submitted >= args.count:
            break
        job_name = eval_job_name(spec["spec_id"], epoch=spec["epoch"])
        cli_argv = _build_eai_submit_argv(spec, queue_root=args.queue_root)
        if args.dry_run:
            print(" ".join(cli_argv))
            submitted += 1
            continue

        result = subprocess.run(cli_argv, check=True, capture_output=True, text=True)
        eai_uuid = extract_eai_uuid(result.stdout)
        append_state_row(args.state_file, {
            "job_id": job_name,
            "spec_id": spec["spec_id"],
            "epoch": spec["epoch"],
            "eai_uuid": eai_uuid,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        print(f"submitted {job_name} -> {eai_uuid}")
        submitted += 1

    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}Eval-pool pending: {len(specs)} | Submitted: {submitted}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
