"""Merge cluster-synced results into the canonical Mila results file."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_RESULTS_DIR = Path("/network/scratch/b/brownet/information-safety/results")
DEFAULT_GENERATIONS_DIR = Path("/network/scratch/b/brownet/information-safety/generations")


@dataclass
class MergeSummary:
    new_rows: int
    asr_preserved_skips: int
    new_symlinks: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".merging")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _should_create_symlink(target: Path, link: Path) -> bool:
    return target.exists() and not link.is_symlink() and not link.exists()


def merge_results(
    *,
    cluster: str,
    results_dir: Path,
    generations_dir: Path,
    dry_run: bool = False,
) -> MergeSummary:
    canonical_path = results_dir / "final-results.jsonl"
    synced_path = results_dir / f"final-results.jsonl.{cluster}"

    canonical_rows = _read_jsonl(canonical_path) if canonical_path.exists() else []
    synced_rows = _read_jsonl(synced_path)

    by_id: dict[str, dict[str, Any]] = {r["eval_run_id"]: r for r in canonical_rows}
    cluster_gen_dir = generations_dir / f"from-{cluster}"

    new_rows = 0
    asr_preserved_skips = 0
    new_symlinks = 0
    for synced in synced_rows:
        run_id = synced["eval_run_id"]

        existing = by_id.get(run_id)
        if existing is None:
            by_id[run_id] = synced
            new_rows += 1
        elif existing["asr"] is not None:
            asr_preserved_skips += 1
        else:
            by_id[run_id] = synced

        if _should_create_symlink(cluster_gen_dir / run_id, generations_dir / run_id):
            new_symlinks += 1

    summary = MergeSummary(
        new_rows=new_rows,
        asr_preserved_skips=asr_preserved_skips,
        new_symlinks=new_symlinks,
    )

    if dry_run:
        return summary

    if canonical_path.exists():
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        shutil.copyfile(canonical_path, results_dir / f"final-results.jsonl.bak.{timestamp}")

    _atomic_write_jsonl(canonical_path, list(by_id.values()))

    for synced in synced_rows:
        run_id = synced["eval_run_id"]
        target = cluster_gen_dir / run_id
        link = generations_dir / run_id
        if _should_create_symlink(target, link):
            link.symlink_to(target)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", required=True, choices=["tamia", "nibi"])
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--generations-dir", type=Path, default=DEFAULT_GENERATIONS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    summary = merge_results(
        cluster=args.cluster,
        results_dir=args.results_dir,
        generations_dir=args.generations_dir,
        dry_run=args.dry_run,
    )
    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}{summary.new_rows} new rows | "
        f"{summary.asr_preserved_skips} ASR-preserved skips | "
        f"{summary.new_symlinks} new symlinks"
    )


if __name__ == "__main__":
    main()
