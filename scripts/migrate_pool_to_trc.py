"""Migrate HF-hosted DataStrategy specs from Tamia's job pool to TRC eai jobs.

Pulls pending spec files from Tamia (``run-wmdp-rerun/pending/``), filters to
HF-hosted models (skipping any spec whose model path starts with ``/``), and
submits each one as an ``eai job submit`` on TRC. After a successful submit,
the spec is *deleted* from Tamia's pending dir so its workers can no longer
pick it up. Idempotent via a local JSONL state file keyed on
``is_pool_<spec_id>``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scripts._trc_common import (
    TRC_BASE_DIR,
    TRC_HF_HOME,
    TRC_REPO_DIR,
    append_state_row,
    build_eai_submit_remote,
    default_state_file,
    extract_eai_uuid,
    load_submitted_state,
)

__all__ = [
    "DEFAULT_STATE_FILE",
    "is_hf_hosted_spec",
    "main",
    "pool_job_name",
    "rewrite_defense_paths",
]

TRC_INFORMATION_SAFETY_VENV = "/work/envs/information-safety/.venv"
DEFENSE_LOCAL_PREFIX = "/scratch/t/tvergara/information-safety/defenses/"
DEFENSE_HF_NAMESPACE = "tvergara"
DATASET_HANDLER_PREFIX = "algorithm/dataset_handler="
TRC_RESULTS_FILE = f"{TRC_BASE_DIR}/main/final-results.jsonl"
TRC_GENERATIONS_DIR = f"{TRC_BASE_DIR}/main/generations"

TAMIA_ROBOT = "robot.tamia.ecpia.ca"
TAMIA_QUEUE_ROOT_BASE = "/scratch/t/tvergara/information-safety/job-pool"

LOCAL_PENDING_CACHE = (
    Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    / "information-safety"
    / "trc-pool-pending"
)

DEFAULT_STATE_FILE = default_state_file("trc-pool-submitted.jsonl")

_HF_MODEL_PREFIX = "algorithm.model.pretrained_model_name_or_path="


def pool_job_name(spec_id: str) -> str:
    return f"is_pool_{spec_id}"


def rewrite_defense_paths(spec: dict[str, Any]) -> dict[str, Any]:
    new_command: list[str] = []
    for token in spec["command"]:
        if token.startswith(_HF_MODEL_PREFIX):
            value = token[len(_HF_MODEL_PREFIX):]
            if value.startswith(DEFENSE_LOCAL_PREFIX):
                defense_name = value[len(DEFENSE_LOCAL_PREFIX):].rstrip("/")
                token = f"{_HF_MODEL_PREFIX}{DEFENSE_HF_NAMESPACE}/{defense_name}"
        new_command.append(token)
    return {**spec, "command": new_command}


def is_hf_hosted_spec(spec: dict[str, Any]) -> bool:
    saw_model_override = False
    for token in spec["command"]:
        if token.startswith(_HF_MODEL_PREFIX):
            saw_model_override = True
            if token[len(_HF_MODEL_PREFIX):].startswith("/"):
                return False
        elif "=/scratch/" in token or "=/network/" in token:
            return False
    if not saw_model_override:
        raise ValueError(
            f"spec {spec.get('id', '?')} has no {_HF_MODEL_PREFIX} override"
        )
    return True


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


def _build_container_cmd(spec: dict[str, Any]) -> str:
    for token in spec["command"]:
        if any(ch.isspace() for ch in token):
            raise ValueError(
                f"spec command token contains whitespace, will be re-split by bash: "
                f"{token!r}"
            )
    spec_cmd = shlex.join(spec["command"])
    return " && ".join([
        f"cd {TRC_REPO_DIR}",
        f"export SCRATCH={TRC_BASE_DIR}",
        f"export RESULTS_FILE={TRC_RESULTS_FILE}",
        f"export GENERATIONS_DIR={TRC_GENERATIONS_DIR}",
        f"export HF_HOME={TRC_HF_HOME}",
        f"export HF_HUB_CACHE={TRC_HF_HOME}/hub",
        f"source {TRC_INFORMATION_SAFETY_VENV}/bin/activate",
        spec_cmd,
    ])


def _build_eai_submit_argv(spec: dict[str, Any]) -> list[str]:
    remote = build_eai_submit_remote(
        name=pool_job_name(spec["id"]),
        container_cmd=_build_container_cmd(spec),
        gpu=1,
        cpu=8,
        mem=64,
        max_run_time=43200,
        preemptable=False,
    )
    return ["ssh", "trc", remote]


def _load_pending_specs(local: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(local.glob("*.json"))]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-dataset", default=None)
    parser.add_argument("--exclude-model", action="append", default=[])
    args = parser.parse_args(argv)

    local = _rsync_pending_to_local(queue_root=args.queue_root)
    pending_specs = [rewrite_defense_paths(s) for s in _load_pending_specs(local)]
    if args.include_dataset is not None:
        wanted = f"{DATASET_HANDLER_PREFIX}{args.include_dataset}"
        pending_specs = [s for s in pending_specs if wanted in s["command"]]
    if args.exclude_model:
        pending_specs = [
            s for s in pending_specs
            if not any(
                t.startswith(_HF_MODEL_PREFIX)
                and t[len(_HF_MODEL_PREFIX):] in args.exclude_model
                for t in s["command"]
            )
        ]
    hf_specs = [s for s in pending_specs if is_hf_hosted_spec(s)]
    submitted_state = load_submitted_state(args.state_file)

    submitted = 0
    for spec in hf_specs:
        if submitted >= args.count:
            break
        spec_id = spec["id"]
        job_id = pool_job_name(spec_id)
        if job_id in submitted_state:
            continue

        cli_argv = _build_eai_submit_argv(spec)
        if args.dry_run:
            print(" ".join(cli_argv))
            submitted += 1
            continue

        result = subprocess.run(cli_argv, check=True, capture_output=True, text=True)
        eai_uuid = extract_eai_uuid(result.stdout)
        append_state_row(args.state_file, {
            "job_id": job_id,
            "spec_id": spec_id,
            "eai_uuid": eai_uuid,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        _delete_tamia_pending(spec_id, args.queue_root)
        print(f"submitted {job_id} -> {eai_uuid}")
        submitted += 1

    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}HF-hosted pending: {len(hf_specs)} | Submitted: {submitted}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
