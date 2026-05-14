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

from scripts.build_job_queue import (
    build_command,
    default_attacks_dir,
    default_results_file,
    iter_missing_configs,
)
from scripts.check_results import (
    EVILMATH_CONFIGS,
    HARMBENCH_CONFIGS,
    STRONGREJECT_CONFIGS,
    WMDP_CONFIGS,
)

TRC_BASE_DIR = "/work/information-safety-results"
TRC_REPO_DIR = "/work/information-safety"
TRC_HF_HOME = "/work/.hf-cache"
TRC_EAI_IMAGE = (
    "registry.toolkit-sp.yul201.service-now.com/snow.research.mmteb/mteb-lite:v1"
)
TRC_DATA_MOUNT = "snow.research.mmteb.safety:/work:rw"

DEFAULT_STATE_FILE = Path(
    os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
) / "information-safety" / "trc-submitted.jsonl"


def deterministic_job_name(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return "is_" + hashlib.sha256(payload).hexdigest()[:16]


def load_submitted_state(state_file: Path) -> set[str]:
    if not state_file.exists():
        return set()
    return {
        json.loads(line)["job_id"]
        for line in state_file.read_text().splitlines()
        if line.strip()
    }


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


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def build_eai_submit_argv(*, config: dict[str, Any]) -> list[str]:
    hydra_cmd = build_command(
        config,
        attacks_dir=Path(f"{TRC_BASE_DIR}/attacks"),
        base_dir=Path(TRC_BASE_DIR),
        existence_check_attacks_dir=default_attacks_dir(),
    )
    hydra_str = " ".join(_shell_quote(tok) for tok in hydra_cmd)
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
    remote = (
        f"eai job submit"
        f" --name {job_name}"
        f" --image {TRC_EAI_IMAGE}"
        f" --data {TRC_DATA_MOUNT}"
        f" --gpu 1 --mem 64 --cpu 8 --max-run-time 43200"
        f" --non-preemptable"
        f" --enforce-name"
        f" -- bash -lc {_shell_quote(container_cmd)}"
    )
    return ["ssh", "trc", remote]


def _submit_one(
    *,
    config: dict[str, Any],
    state_file: Path,
    dry_run: bool,
) -> None:
    argv = build_eai_submit_argv(config=config)
    if dry_run:
        print(f"{argv[0]} {argv[1]} {_shell_quote(argv[2])}")
        return
    result = subprocess.run(argv, check=True, capture_output=True, text=True)
    eai_uuid = result.stdout.strip().splitlines()[-1]
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
            _submit_one(config=config, state_file=state_file, dry_run=args.dry_run)
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
