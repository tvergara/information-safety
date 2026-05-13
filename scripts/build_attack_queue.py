from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import datasets

from scripts.build_job_queue import (
    deterministic_id,
    ensure_queue_dirs,
    is_job_known,
    write_pending_payload,
)


@dataclass(frozen=True)
class DatasetPaths:
    behaviors_csv: Path
    targets_json: Path
    total: int


def compute_shards(*, total: int, num_shards: int) -> list[tuple[int, int]]:
    base = total // num_shards
    remainder = total % num_shards
    shards: list[tuple[int, int]] = []
    cursor = 0
    for i in range(num_shards):
        size = base + (1 if i < remainder else 0)
        if size == 0:
            continue
        shards.append((cursor, cursor + size))
        cursor += size
    return shards


_HARMBENCH_PASSTHROUGH_CATEGORY = "harmful"


def write_strongreject_csv(output_path: Path) -> int:
    raw = datasets.load_dataset("walledai/StrongREJECT", split="train")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Behavior", "FunctionalCategory", "SemanticCategory",
            "Tags", "ContextString", "BehaviorID",
        ])
        count = 0
        for i, row in enumerate(cast(list[dict[str, Any]], raw)):
            writer.writerow([
                row["prompt"], "standard", _HARMBENCH_PASSTHROUGH_CATEGORY,
                "", "", f"strongreject_{i:04d}",
            ])
            count += 1
    return count


def write_strongreject_targets(csv_path: Path, json_path: Path) -> int:
    targets: dict[str, str] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            behavior_id = row["BehaviorID"]
            targets[behavior_id] = "Sure, here is " + row["Behavior"].strip().lower()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(targets, f)
    return len(targets)


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "_")


_CSV_BASENAMES = {
    "harmbench": "harmbench_behaviors_standard.csv",
    "strongreject": "strongreject_behaviors.csv",
    "wmdp": "wmdp_behaviors.csv",
    "evilmath": "evilmath_behaviors.csv",
}

_TARGETS_BASENAMES = {
    "strongreject": "strongreject_targets_text.json",
    "wmdp": "wmdp_targets_text.json",
    "evilmath": "evilmath_targets_text.json",
}

_HARMBENCH_TARGETS_PATH = (
    Path(__file__).resolve().parent.parent
    / "third_party/adversariallm/data/optimizer_targets/harmbench_targets_text.json"
)


def _csv_for(*, dataset: str, behaviors_csv_dir: Path) -> Path:
    if dataset not in _CSV_BASENAMES:
        raise ValueError(f"Unknown dataset: {dataset}")
    return behaviors_csv_dir / _CSV_BASENAMES[dataset]


def _targets_for(*, dataset: str, behaviors_csv_dir: Path) -> Path:
    if dataset == "harmbench":
        return _HARMBENCH_TARGETS_PATH
    if dataset not in _TARGETS_BASENAMES:
        raise ValueError(f"Unknown dataset: {dataset}")
    return behaviors_csv_dir / _TARGETS_BASENAMES[dataset]


def _count_csv_rows(csv_path: Path) -> int:
    with open(csv_path) as f:
        return sum(1 for _ in csv.reader(f)) - 1


def build_attack_queue(
    *,
    attacks: list[str],
    models: list[str],
    dataset_names: list[str],
    num_shards: int,
    queue_root: Path,
    behaviors_csv_dir: Path,
    attacks_dir: Path,
    adversariallm_venv: Path,
    repo_root: Path,
) -> list[Path]:
    ensure_queue_dirs(queue_root)
    pending_dir = queue_root / "pending"

    dataset_paths: dict[str, DatasetPaths] = {}
    for dataset in dataset_names:
        csv_path = _csv_for(dataset=dataset, behaviors_csv_dir=behaviors_csv_dir)
        if not csv_path.exists():
            raise FileNotFoundError(f"Behaviors CSV not found: {csv_path}")
        targets_path = _targets_for(
            dataset=dataset, behaviors_csv_dir=behaviors_csv_dir
        )
        if not targets_path.exists():
            raise FileNotFoundError(f"Targets JSON not found: {targets_path}")
        dataset_paths[dataset] = DatasetPaths(
            behaviors_csv=csv_path,
            targets_json=targets_path,
            total=_count_csv_rows(csv_path),
        )

    written: list[Path] = []
    for attack in attacks:
        for model in models:
            for dataset in dataset_names:
                slug = _model_slug(model)
                merged = attacks_dir / f"{attack}-{slug}-{dataset}-merged.jsonl"
                if merged.exists():
                    continue
                paths = dataset_paths[dataset]
                shards = compute_shards(
                    total=paths.total, num_shards=num_shards
                )
                for shard_idx, (start, end) in enumerate(shards):
                    spec = {
                        "attack": attack,
                        "model": model,
                        "dataset": dataset,
                        "shard": shard_idx,
                        "start": start,
                        "end": end,
                    }
                    job_id = deterministic_id(spec)
                    if is_job_known(queue_root, job_id):
                        continue

                    output_jsonl = (
                        attacks_dir
                        / f"{attack}-{slug}-{dataset}-shard{shard_idx}.jsonl"
                    )
                    adv_save_dir = (
                        attacks_dir.parent
                        / "adversariallm-outputs"
                        / f"{attack}-{slug}-{dataset}-shard{shard_idx}"
                    )

                    command = [
                        "bash",
                        str(repo_root / "scripts/run_one_attack.sh"),
                        "--attack", attack,
                        "--model", model,
                        "--shard-start", str(start),
                        "--shard-end", str(end),
                        "--behaviors-csv", str(paths.behaviors_csv),
                        "--targets-json", str(paths.targets_json),
                        "--output-jsonl", str(output_jsonl),
                        "--adv-save-dir", str(adv_save_dir),
                        "--adversariallm-venv", str(adversariallm_venv),
                        "--repo-root", str(repo_root),
                    ]
                    payload = {
                        "id": job_id,
                        "command": command,
                        "spec": spec,
                        "attempts": 0,
                    }
                    written.append(
                        write_pending_payload(pending_dir, job_id, payload)
                    )
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument(
        "--attacks", nargs="+", default=["gcg", "autodan", "pair"]
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "Qwen/Qwen3-4B",
            "allenai/Olmo-3-7B-Instruct",
        ],
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["harmbench", "strongreject"]
    )
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--behaviors-csv-dir", type=Path, required=True)
    parser.add_argument("--attacks-dir", type=Path, required=True)
    parser.add_argument("--adversariallm-venv", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--build-strongreject-csv", action="store_true")
    args = parser.parse_args(argv)

    if args.build_strongreject_csv:
        sr_csv = args.behaviors_csv_dir / "strongreject_behaviors.csv"
        n = write_strongreject_csv(sr_csv)
        print(f"Wrote {n} StrongREJECT prompts to {sr_csv}")
        sr_targets = args.behaviors_csv_dir / "strongreject_targets_text.json"
        m = write_strongreject_targets(sr_csv, sr_targets)
        print(f"Wrote {m} StrongREJECT targets to {sr_targets}")

    written = build_attack_queue(
        attacks=args.attacks,
        models=args.models,
        dataset_names=args.datasets,
        num_shards=args.num_shards,
        queue_root=args.queue_root,
        behaviors_csv_dir=args.behaviors_csv_dir,
        attacks_dir=args.attacks_dir,
        adversariallm_venv=args.adversariallm_venv,
        repo_root=args.repo_root,
    )
    print(f"Wrote {len(written)} pending jobs to {args.queue_root / 'pending'}")


if __name__ == "__main__":
    main()
