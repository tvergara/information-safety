"""Migrate HF-hosted pending specs from Tamia's job pool to Nibi single-GPU sbatch jobs.

Unlike Tamia (whole-node 4xH100 pool) or TRC (eai), Nibi exposes the
``gpubase_bygpu_b2`` partition which takes single-GPU requests directly. Each
pending spec is sbatch'd individually — no pool, no claim state machine.
SLURM's queue is the queue; ``final-results.jsonl`` is the source of truth
for completion.

Pulls pending spec files from Tamia (via robot.tamia), filters to HF-hosted
models, then submits each as one sbatch on Nibi (via robot.nibi). The Tamia
pending file is deleted only after the sbatch succeeds, so a re-run picks
up exactly the specs that still need to ship.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scripts._trc_common import append_state_row, default_state_file
from scripts.migrate_pool_to_trc import (
    LOCAL_PENDING_CACHE,
    TAMIA_QUEUE_ROOT_BASE,
    TAMIA_ROBOT,
    is_hf_hosted_spec,
    rewrite_defense_paths,
)

__all__ = [
    "DEFAULT_STATE_FILE",
    "NIBI_GENERATIONS_DIR",
    "NIBI_REPO_DIR",
    "NIBI_RESULTS_FILE",
    "NIBI_ROBOT",
    "NIBI_RUN_SCRIPT",
    "extract_slurm_job_id",
    "main",
]

NIBI_ROBOT = "robot.nibi.alliancecan.ca"
NIBI_REPO_DIR = "/home/tvergara/information-safety"
NIBI_RUN_SCRIPT = f"{NIBI_REPO_DIR}/slurm/run-nibi-single.sh"
NIBI_RESULTS_FILE = "/scratch/tvergara/information-safety/results/final-results.jsonl"
NIBI_GENERATIONS_DIR = "/scratch/tvergara/information-safety/generations"

DEFAULT_STATE_FILE = default_state_file("nibi-pool-submitted.jsonl")

_SBATCH_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")


def extract_slurm_job_id(stdout: str) -> str:
    m = _SBATCH_JOB_ID_RE.search(stdout)
    if m is None:
        raise ValueError(f"no 'Submitted batch job N' line found in sbatch output: {stdout!r}")
    return m.group(1)


def _rsync_pending_to_local(*, queue_root: str) -> Path:
    target = LOCAL_PENDING_CACHE / queue_root
    target.mkdir(parents=True, exist_ok=True)
    remote = f"{TAMIA_ROBOT}:{TAMIA_QUEUE_ROOT_BASE}/{queue_root}/pending/"
    subprocess.run(
        ["rsync", "-a", "--delete", remote, f"{target}/"],
        check=True,
        capture_output=True,
        text=True,
    )
    return target


def _delete_tamia_pending(spec_id: str, queue_root: str) -> None:
    remote_path = f"{TAMIA_QUEUE_ROOT_BASE}/{queue_root}/pending/{spec_id}.json"
    subprocess.run(
        ["ssh", TAMIA_ROBOT, f"rm {remote_path}"],
        check=True,
        capture_output=True,
        text=True,
    )


def _load_pending_specs(local: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(local.glob("*.json"))]


def _build_sbatch_argv(spec: dict[str, Any]) -> list[str]:
    for token in spec["command"]:
        if any(ch.isspace() for ch in token):
            raise ValueError(
                f"spec command token contains whitespace, will be re-split by ssh: "
                f"{token!r}"
            )
    job_name = f"nibi-pool-{spec['id']}"
    return [
        "ssh", NIBI_ROBOT,
        "sbatch",
        f"--job-name={job_name}",
        NIBI_RUN_SCRIPT,
        *spec["command"],
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    local = _rsync_pending_to_local(queue_root=args.queue_root)
    pending_specs = [rewrite_defense_paths(s) for s in _load_pending_specs(local)]
    hf_specs = [s for s in pending_specs if is_hf_hosted_spec(s)]

    submitted = 0
    for spec in hf_specs:
        if submitted >= args.count:
            break
        spec_id = spec["id"]

        cli_argv = _build_sbatch_argv(spec)
        if args.dry_run:
            print(" ".join(cli_argv))
            submitted += 1
            continue

        result = subprocess.run(cli_argv, check=True, capture_output=True, text=True)
        slurm_job_id = extract_slurm_job_id(result.stdout)
        append_state_row(args.state_file, {
            "spec_id": spec_id,
            "slurm_job_id": slurm_job_id,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        _delete_tamia_pending(spec_id, args.queue_root)
        print(f"submitted {spec_id} -> nibi sbatch {slurm_job_id}")
        submitted += 1

    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}HF-hosted pending: {len(hf_specs)} | Submitted: {submitted}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
