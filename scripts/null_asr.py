"""Null the ``asr`` field on rows in final-results.jsonl so they re-enter eval.

Useful after a parser fix: setting ``asr=None`` makes the row visible to the
WMDP backfill (``evaluate_wmdp.py --results-file``) and the HarmBench eval
sweep, which both look for ``asr is None``.

Skips rows already invalidated (``invalidated_reason`` set) and rows that are
already ``asr=None``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.check_results import default_results_file


@dataclass
class NullSummary:
    nulled: int
    already_null: int
    skipped_invalidated: int


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".nulling")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def null_asr_rows(
    *,
    results_path: Path,
    predicate: Callable[[dict[str, Any]], bool],
    dry_run: bool = False,
) -> NullSummary:
    with open(results_path) as f:
        rows = [json.loads(line) for line in f]

    nulled = 0
    already_null = 0
    skipped_invalidated = 0
    for row in rows:
        if not predicate(row):
            continue
        if row.get("invalidated_reason") is not None:
            skipped_invalidated += 1
            continue
        if row.get("asr") is None:
            already_null += 1
            continue
        if not dry_run:
            row["asr"] = None
        nulled += 1

    summary = NullSummary(
        nulled=nulled, already_null=already_null, skipped_invalidated=skipped_invalidated,
    )

    if dry_run or nulled == 0:
        return summary

    backup = results_path.with_suffix(results_path.suffix + ".bak")
    shutil.copyfile(results_path, backup)
    _atomic_write_jsonl(results_path, rows)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-file", type=Path, default=default_results_file())
    parser.add_argument("--dataset", help="Comma-separated dataset_name values to match")
    parser.add_argument("--experiment", help="Comma-separated experiment_name values to match")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    datasets = set(args.dataset.split(",")) if args.dataset else None
    experiments = set(args.experiment.split(",")) if args.experiment else None

    def predicate(row: dict[str, Any]) -> bool:
        if datasets is not None and row.get("dataset_name") not in datasets:
            return False
        if experiments is not None and row.get("experiment_name") not in experiments:
            return False
        return True

    summary = null_asr_rows(
        results_path=args.results_file, predicate=predicate, dry_run=args.dry_run,
    )
    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}{summary.nulled} nulled | "
        f"{summary.already_null} already null | "
        f"{summary.skipped_invalidated} skipped (invalidated)"
    )


if __name__ == "__main__":
    main()
