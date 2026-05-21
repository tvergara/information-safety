"""Submit eval-pool drain workers to TRC, one or more per base_model.

Each worker is a single-GPU eai job that runs ``scripts/run_eval_pool.py``
against the canonical TRC eval-pool queue, filtered to a specific
``--base-model``. Workers are non-preemptable with a 6h cap; re-submit to
resume.

Usage:
    python scripts/submit_trc_eval_workers.py \\
        --queue-root /work/eval-pool/run-canonical-deferred \\
        --workers-per-model 2 \\
        --base-models meta-llama/Meta-Llama-3-8B-Instruct,Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from scripts._trc_common import (
    append_state_row,
    build_eai_submit_remote,
    build_eval_pool_container_cmd,
    default_state_file,
    eai_name_slug,
    ensure_trc_synced,
    extract_eai_uuid,
)

DEFAULT_STATE_FILE = default_state_file("trc-eval-worker-pool-submitted.jsonl")


def worker_job_name(model: str, worker_index: int, submit_ts: int) -> str:
    return f"is_evalpool_{eai_name_slug(model)}_w{worker_index}_{submit_ts}"


def build_submit_argv(model: str, worker_index: int, submit_ts: int, queue_root: str) -> list[str]:
    remote = build_eai_submit_remote(
        name=worker_job_name(model, worker_index, submit_ts),
        container_cmd=build_eval_pool_container_cmd(base_model=model, queue_root=queue_root),
        gpu=1,
        cpu=8,
        mem=64,
        max_run_time=21600,
        preemptable=False,
    )
    return ["ssh", "trc", remote]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--workers-per-model", type=int, required=True)
    parser.add_argument(
        "--base-models",
        required=True,
        help="Comma-separated list of base_model identifiers (HF names).",
    )
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.dry_run:
        ensure_trc_synced()

    models = args.base_models.split(",")
    submit_ts = int(time.time())
    submitted = 0
    for model in models:
        for worker_index in range(args.workers_per_model):
            cli_argv = build_submit_argv(model, worker_index, submit_ts, args.queue_root)
            if args.dry_run:
                print(" ".join(cli_argv))
                submitted += 1
                continue
            result = subprocess.run(cli_argv, check=True, capture_output=True, text=True)
            eai_uuid = extract_eai_uuid(result.stdout)
            append_state_row(args.state_file, {
                "job_id": worker_job_name(model, worker_index, submit_ts),
                "base_model": model,
                "worker_index": worker_index,
                "eai_uuid": eai_uuid,
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            print(f"submitted {worker_job_name(model, worker_index, submit_ts)} -> {eai_uuid}")
            submitted += 1
    prefix = "[dry-run] " if args.dry_run else ""
    print(f"{prefix}Submitted {submitted} workers across {len(models)} base_models", file=sys.stderr)


if __name__ == "__main__":
    main()
