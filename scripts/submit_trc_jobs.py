"""Submit pending information-safety jobs to trc via ``eai job submit``."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scripts._trc_common import (
    TRC_BASE_DIR,
    TRC_DATA_MOUNT,
    TRC_EAI_IMAGE,
    TRC_HF_HOME,
    TRC_REPO_DIR,
    extract_eai_uuid,
    load_submitted_state,
    shell_quote,
)
from scripts.build_job_queue import (
    build_command,
    default_results_file,
    iter_missing_configs,
)
from scripts.check_results import (
    EVILMATH_CONFIGS,
    HARMBENCH_CONFIGS,
    STRONGREJECT_CONFIGS,
    WMDP_CONFIGS,
)

TRC_DEFENSE_HF_NAMESPACE = "tvergara"

DEFAULT_STATE_FILE = Path(
    os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
) / "information-safety" / "trc-submitted.jsonl"


def deterministic_job_name(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return "is_" + hashlib.sha256(payload).hexdigest()[:16]


def record_submission(
    *,
    state_file: Path,
    config: dict[str, Any],
    job_id: str,
    eai_uuid: str,
) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "job_id": job_id,
        "eai_uuid": eai_uuid,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config,
    }
    with open(state_file, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def iter_pending_configs(
    *,
    configs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    state_file: Path,
) -> list[dict[str, Any]]:
    missing = iter_missing_configs(configs, rows)
    submitted = load_submitted_state(state_file)
    return [c for c in missing if deterministic_job_name(c) not in submitted]


def build_eai_submit_argv(
    *, config: dict[str, Any], preemptable: bool = True
) -> list[str]:
    hydra_cmd = build_command(
        config,
        attacks_dir=Path(f"{TRC_BASE_DIR}/attacks"),
        base_dir=Path(TRC_BASE_DIR),
        defense_hf_namespace=TRC_DEFENSE_HF_NAMESPACE,
    )
    hydra_str = " ".join(shell_quote(tok) for tok in hydra_cmd)
    container_cmd = " && ".join([
        f"cd {TRC_REPO_DIR}",
        "source .venv/bin/activate",
        f"export RESULTS_FILE={TRC_BASE_DIR}/results/final-results.jsonl",
        f"export GENERATIONS_DIR={TRC_BASE_DIR}/generations",
        f"export HF_HOME={TRC_HF_HOME}",
        f"export SCRATCH={TRC_BASE_DIR}",
        "export HYDRA_FULL_ERROR=1",
        hydra_str,
    ])
    job_name = deterministic_job_name(config)
    preempt_flag = "--preemptable" if preemptable else "--non-preemptable"
    remote = (
        f"eai job submit"
        f" --name {job_name}"
        f" --image {TRC_EAI_IMAGE}"
        f" --data {TRC_DATA_MOUNT}"
        f" --gpu 1 --mem 64 --cpu 8 --max-run-time 43200"
        f" {preempt_flag}"
        f" --enforce-name"
        f" -- bash -lc {shell_quote(container_cmd)}"
    )
    return ["ssh", "trc", remote]


def _submit_one(
    *,
    config: dict[str, Any],
    state_file: Path,
    dry_run: bool,
    preemptable: bool,
) -> None:
    argv = build_eai_submit_argv(config=config, preemptable=preemptable)
    if dry_run:
        print(f"{argv[0]} {argv[1]} {shell_quote(argv[2])}")
        return
    result = subprocess.run(argv, check=True, capture_output=True, text=True)
    eai_uuid = extract_eai_uuid(result.stdout)
    record_submission(
        state_file=state_file,
        config=config,
        job_id=deterministic_job_name(config),
        eai_uuid=eai_uuid,
    )
    print(f"submitted {deterministic_job_name(config)} -> {eai_uuid}")


def main(
    argv: list[str] | None = None,
    *,
    configs: list[dict[str, Any]] | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-file", type=Path, default=default_results_file())
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset", action="store_true")
    preempt_group = parser.add_mutually_exclusive_group()
    preempt_group.add_argument(
        "--preemptable", dest="preemptable", action="store_true", default=True
    )
    preempt_group.add_argument(
        "--non-preemptable", dest="preemptable", action="store_false"
    )
    args = parser.parse_args(argv)

    state_file = args.state_file if args.state_file is not None else DEFAULT_STATE_FILE
    if args.reset and state_file.exists():
        state_file.unlink()

    if configs is None:
        configs = (
            HARMBENCH_CONFIGS + WMDP_CONFIGS + EVILMATH_CONFIGS + STRONGREJECT_CONFIGS
        )

    rows = [
        json.loads(line)
        for line in args.results_file.read_text().splitlines()
        if line.strip()
    ]
    pending = iter_pending_configs(
        configs=configs,
        rows=rows,
        state_file=state_file,
    )

    submitted = 0
    skipped = 0
    for config in pending:
        try:
            _submit_one(
                config=config,
                state_file=state_file,
                dry_run=args.dry_run,
                preemptable=args.preemptable,
            )
            submitted += 1
        except FileNotFoundError as exc:
            print(f"WARNING: skipping config — {exc}")
            skipped += 1

    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}Total configs: {len(configs)} | "
        f"Pending: {len(pending)} | "
        f"Submitted: {submitted} | "
        f"Skipped: {skipped}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
