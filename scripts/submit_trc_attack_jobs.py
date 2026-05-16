"""Submit pending AdversariaLLM attack shards to trc via ``eai job submit``."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scripts._trc_common import (
    TRC_BASE_DIR,
    TRC_EAI_IMAGE,
    TRC_HF_HOME,
    TRC_REPO_DIR,
    append_state_row,
    build_eai_submit_remote,
    compute_shards,
    default_state_file,
    extract_eai_uuid,
    load_submitted_state,
    model_slug,
    shell_quote,
    trc_behaviors_csv,
    trc_targets_json,
)

__all__ = [
    "DATASET_TOTALS_DEFAULT",
    "TRC_ADVERSARIALLM_VENV",
    "TRC_BASE_DIR",
    "TRC_EAI_IMAGE",
    "TRC_REPO_DIR",
    "build_eai_submit_argv",
    "deterministic_job_name",
    "iter_pending_attack_shards",
    "load_submitted_state",
    "main",
    "record_submission",
]

TRC_ADVERSARIALLM_VENV = "/work/envs/adversariallm/.venv"

DATASET_TOTALS_DEFAULT: dict[str, int] = {
    "harmbench": 200,
    "strongreject": 313,
    "evilmath": 298,
}

DEFAULT_ATTACKS = ["gcg", "autodan", "pair"]
DEFAULT_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen3-4B",
    "allenai/Olmo-3-7B-Instruct",
]
DEFAULT_DATASETS = ["harmbench", "strongreject"]

DEFAULT_STATE_FILE = default_state_file("trc-attacks-submitted.jsonl")


def deterministic_job_name(spec: dict[str, Any]) -> str:
    identity = {
        "attack": spec["attack"],
        "model": spec["model"],
        "dataset": spec["dataset"],
        "shard": spec["shard"],
    }
    payload = json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    return "is_attack_" + hashlib.sha256(payload).hexdigest()[:16]


def record_submission(
    *,
    state_file: Path,
    spec: dict[str, Any],
    job_id: str,
    eai_uuid: str,
) -> None:
    append_state_row(state_file, {
        "job_id": job_id,
        "eai_uuid": eai_uuid,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec": spec,
    })


def iter_pending_attack_shards(
    *,
    attacks: list[str],
    models: list[str],
    datasets: list[str],
    dataset_totals: dict[str, int],
    num_shards: int,
    merged_attacks: set[tuple[str, str, str]],
    submitted_state: set[str],
) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for attack in attacks:
        for model in models:
            slug = model_slug(model)
            for dataset in datasets:
                if (attack, slug, dataset) in merged_attacks:
                    continue
                total = dataset_totals[dataset]
                for shard_idx, (start, end) in enumerate(
                    compute_shards(total=total, num_shards=num_shards)
                ):
                    spec = {
                        "attack": attack,
                        "model": model,
                        "dataset": dataset,
                        "shard": shard_idx,
                        "start": start,
                        "end": end,
                    }
                    if deterministic_job_name(spec) in submitted_state:
                        continue
                    pending.append(spec)
    return pending


def build_eai_submit_argv(
    *, spec: dict[str, Any], preemptable: bool = True
) -> list[str]:
    attack = spec["attack"]
    model = spec["model"]
    dataset = spec["dataset"]
    shard_idx = spec["shard"]
    start = spec["start"]
    end = spec["end"]
    slug = model_slug(model)

    behaviors_csv = trc_behaviors_csv(dataset)
    targets_json = trc_targets_json(dataset)
    base_name = f"{attack}-{slug}-{dataset}-shard{shard_idx}"
    output_jsonl = f"{TRC_BASE_DIR}/attacks/{base_name}.jsonl"
    adv_save_dir = f"{TRC_BASE_DIR}/adversariallm-outputs/{base_name}"

    run_one_attack = " ".join([
        "bash scripts/run_one_attack.sh",
        f"--attack {attack}",
        f"--model {model}",
        f"--shard-start {start}",
        f"--shard-end {end}",
        f"--behaviors-csv {behaviors_csv}",
        f"--targets-json {targets_json}",
        f"--output-jsonl {output_jsonl}",
        f"--adv-save-dir {adv_save_dir}",
        f"--adversariallm-venv {TRC_ADVERSARIALLM_VENV}",
        f"--repo-root {TRC_REPO_DIR}",
    ])
    container_cmd = " && ".join([
        f"cd {TRC_REPO_DIR}",
        "unset TRANSFORMERS_CACHE",
        f"export HF_HOME={TRC_HF_HOME}",
        f"export HF_HUB_CACHE={TRC_HF_HOME}/hub",
        f"export HUGGING_FACE_HUB_TOKEN=$(cat {TRC_HF_HOME}/token)",
        run_one_attack,
    ])
    remote = build_eai_submit_remote(
        name=deterministic_job_name(spec),
        container_cmd=container_cmd,
        gpu=1,
        cpu=8,
        mem=64,
        max_run_time=43200,
        preemptable=preemptable,
    )
    return ["ssh", "trc", remote]


def _submit_one(
    *,
    spec: dict[str, Any],
    state_file: Path,
    dry_run: bool,
    preemptable: bool,
) -> None:
    argv = build_eai_submit_argv(spec=spec, preemptable=preemptable)
    if dry_run:
        print(f"{argv[0]} {argv[1]} {shell_quote(argv[2])}")
        return
    result = subprocess.run(argv, check=True, capture_output=True, text=True)
    eai_uuid = extract_eai_uuid(result.stdout)
    job_id = deterministic_job_name(spec)
    record_submission(
        state_file=state_file,
        spec=spec,
        job_id=job_id,
        eai_uuid=eai_uuid,
    )
    print(f"submitted {job_id} -> {eai_uuid}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attacks", nargs="+", default=DEFAULT_ATTACKS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--max-submissions", type=int, default=None)
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

    submitted_state = load_submitted_state(state_file)
    pending = iter_pending_attack_shards(
        attacks=args.attacks,
        models=args.models,
        datasets=args.datasets,
        dataset_totals=DATASET_TOTALS_DEFAULT,
        num_shards=args.num_shards,
        merged_attacks=set(),
        submitted_state=submitted_state,
    )

    submitted = 0
    for spec in pending:
        if args.max_submissions is not None and submitted >= args.max_submissions:
            break
        _submit_one(
            spec=spec,
            state_file=state_file,
            dry_run=args.dry_run,
            preemptable=args.preemptable,
        )
        submitted += 1

    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}Pending: {len(pending)} | Submitted: {submitted}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
