"""Migrate HF-hosted DataStrategy specs from a Mila-local pending dir to TRC eai jobs.

Reads spec JSONs from a directory on the Mila filesystem (passed via
``--pending-dir``), filters to HF-hosted models (skipping any spec whose model
path starts with ``/``), and submits each as an ``eai job submit`` on TRC. Job
names are ``is_pool_<spec_id>_<epoch>`` so re-runs never collide with TRC's
``--enforce-name`` on already-terminated jobs of the same spec.

Idempotent via the pending dir: spec files are deleted after a successful
submit, so re-runs only see specs that still need to ship. The local JSONL
state file is an append-only audit log (``job_id``, ``spec_id``, ``eai_uuid``,
``submitted_at``) and is no longer consulted for dedup.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scripts._trc_common import (
    TRC_ADAPTER_ROOT,
    TRC_BASE_DIR,
    TRC_EVAL_QUEUE_BASE,
    TRC_HF_HOME,
    TRC_REPO_DIR,
    append_state_row,
    build_eai_submit_remote,
    current_git_sha,
    default_state_file,
    extract_eai_uuid,
    verify_sha_pushed,
)

__all__ = [
    "DEFAULT_STATE_FILE",
    "is_hf_hosted_spec",
    "main",
    "pool_job_name",
    "rewrite_defense_paths",
]

TRC_INFORMATION_SAFETY_VENV = "/work/envs/information-safety/.venv"
DEFENSE_LOCAL_PREFIXES = (
    "/scratch/t/tvergara/information-safety/defenses/",
    "/network/scratch/b/brownet/information-safety/defenses/",
)
DEFENSE_HF_NAMESPACE = "tvergara"
DATASET_HANDLER_PREFIX = "algorithm/dataset_handler="
TRC_RESULTS_FILE = f"{TRC_BASE_DIR}/main/final-results.jsonl"
TRC_GENERATIONS_DIR = f"{TRC_BASE_DIR}/main/generations"

DEFAULT_STATE_FILE = default_state_file("trc-pool-submitted.jsonl")
TRC_SYNC_STATE_FILE = default_state_file("trc-synced-sha")

_HF_MODEL_PREFIX = "algorithm.model.pretrained_model_name_or_path="

_SYNC_POLL_SECONDS = 10
_SYNC_TIMEOUT_SECONDS = 1800


def pool_job_name(spec_id: str, *, epoch: int) -> str:
    return f"is_pool_{spec_id}_{epoch}"


def rewrite_defense_paths(spec: dict[str, Any]) -> dict[str, Any]:
    new_command: list[str] = []
    for token in spec["command"]:
        if token.startswith(_HF_MODEL_PREFIX):
            value = token[len(_HF_MODEL_PREFIX):]
            for prefix in DEFENSE_LOCAL_PREFIXES:
                if value.startswith(prefix):
                    defense_name = value[len(prefix):].rstrip("/")
                    token = f"{_HF_MODEL_PREFIX}{DEFENSE_HF_NAMESPACE}/{defense_name}"
                    break
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
        return False
    return True


def _build_container_cmd(spec: dict[str, Any], *, eval_queue_root: str) -> str:
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
        "export HF_HUB_OFFLINE=1",
        "export HF_DATASETS_OFFLINE=1",
        f"export ADAPTER_ROOT={TRC_ADAPTER_ROOT}",
        f"export EVAL_QUEUE_ROOT={eval_queue_root}",
        f"source {TRC_INFORMATION_SAFETY_VENV}/bin/activate",
        spec_cmd,
    ])


def _build_eai_submit_argv(spec: dict[str, Any], *, epoch: int, eval_queue_root: str) -> list[str]:
    remote = build_eai_submit_remote(
        name=pool_job_name(spec["id"], epoch=epoch),
        container_cmd=_build_container_cmd(spec, eval_queue_root=eval_queue_root),
        gpu=1,
        cpu=8,
        mem=64,
        max_run_time=43200,
        preemptable=False,
    )
    return ["ssh", "trc", remote]


def _load_pending_specs(local: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(local.glob("*.json"))]


def _build_sync_container_cmd(*, sha: str) -> str:
    return " && ".join([
        'export PATH="$HOME/.local/bin:$PATH"',
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        f"cd {TRC_REPO_DIR}",
        "git fetch origin --quiet",
        f"git reset --hard {sha}",
        "git submodule update --init --recursive",
        f"(source {TRC_INFORMATION_SAFETY_VENV}/bin/activate"
        " && uv pip install --quiet --index-strategy unsafe-best-match"
        " --extra-index-url https://download.pytorch.org/whl/cu128 -e .)",
    ])


def _wait_for_eai_job(uuid: str) -> None:
    deadline = time.time() + _SYNC_TIMEOUT_SECONDS
    while True:
        result = subprocess.run(
            ["ssh", "trc", f"eai job info {uuid}"],
            check=True, capture_output=True, text=True,
        )
        state = next(
            (
                line.split(":", 1)[1].strip()
                for line in result.stdout.splitlines()
                if line.startswith("state:")
            ),
            None,
        )
        if state is None:
            raise RuntimeError(
                f"no 'state:' line in eai job info {uuid} output"
            )
        if state == "SUCCEEDED":
            return
        if state in {"FAILED", "CANCELLED", "INTERRUPTED"}:
            raise RuntimeError(f"sync job {uuid} ended in state {state}")
        if time.time() >= deadline:
            raise TimeoutError(
                f"sync job {uuid} did not finish within {_SYNC_TIMEOUT_SECONDS}s"
            )
        time.sleep(_SYNC_POLL_SECONDS)


def _ensure_trc_synced(*, state_file: Path = TRC_SYNC_STATE_FILE) -> None:
    sha = current_git_sha()
    if state_file.exists() and state_file.read_text().strip() == sha:
        return
    verify_sha_pushed(sha)
    print(f"Syncing TRC /work to {sha[:8]}...", file=sys.stderr)
    remote = build_eai_submit_remote(
        name=f"is_pool_sync_{sha[:8]}",
        container_cmd=_build_sync_container_cmd(sha=sha),
        cpu=4, mem=16, max_run_time=1800, preemptable=False,
    )
    result = subprocess.run(
        ["ssh", "trc", remote], check=True, capture_output=True, text=True,
    )
    uuid = extract_eai_uuid(result.stdout)
    _wait_for_eai_job(uuid)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(sha)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue-root",
        required=True,
        help="TRC eval-queue subdir name (suffixed with '-deferred').",
    )
    parser.add_argument(
        "--pending-dir",
        type=Path,
        required=True,
        help="Mila-local directory containing pending spec JSONs.",
    )
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-dataset", default=None)
    parser.add_argument("--exclude-model", action="append", default=[])
    args = parser.parse_args(argv)

    if not args.dry_run:
        _ensure_trc_synced()

    pending_specs = [rewrite_defense_paths(s) for s in _load_pending_specs(args.pending_dir)]
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
    epoch = int(time.time())
    eval_queue_root = f"{TRC_EVAL_QUEUE_BASE}/{args.queue_root}-deferred"

    submitted = 0
    for spec in hf_specs:
        if submitted >= args.count:
            break
        spec_id = spec["id"]
        job_id = pool_job_name(spec_id, epoch=epoch)

        cli_argv = _build_eai_submit_argv(
            spec, epoch=epoch, eval_queue_root=eval_queue_root,
        )
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
        (args.pending_dir / f"{spec_id}.json").unlink()
        print(f"submitted {job_id} -> {eai_uuid}")
        submitted += 1

    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}HF-hosted pending: {len(hf_specs)} | Submitted: {submitted}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
