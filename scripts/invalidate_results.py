"""Flag rows in final-results.jsonl as invalidated.

Sets `invalidated_reason` on matching rows. Consumers that filter on this
field (evaluators, plots, queue builder) skip flagged rows.
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
class InvalidateSummary:
    flagged: int
    already_flagged: int


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".invalidating")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def invalidate_rows(
    *,
    results_path: Path,
    predicate: Callable[[dict[str, Any]], bool],
    reason: str,
    dry_run: bool = False,
) -> InvalidateSummary:
    with open(results_path) as f:
        rows = [json.loads(line) for line in f]

    flagged = 0
    already_flagged = 0
    for row in rows:
        if row.get("invalidated_reason") is not None:
            already_flagged += 1
            continue
        if predicate(row):
            if not dry_run:
                row["invalidated_reason"] = reason
            flagged += 1

    summary = InvalidateSummary(flagged=flagged, already_flagged=already_flagged)

    if dry_run or flagged == 0:
        return summary

    backup = results_path.with_suffix(results_path.suffix + ".bak")
    shutil.copyfile(results_path, backup)
    _atomic_write_jsonl(results_path, rows)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-file", type=Path, default=default_results_file())
    parser.add_argument("--reason", required=True, help="String stored in invalidated_reason")
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

    summary = invalidate_rows(
        results_path=args.results_file,
        predicate=predicate,
        reason=args.reason,
        dry_run=args.dry_run,
    )
    prefix = "[dry-run] " if args.dry_run else ""
    print(f"{prefix}{summary.flagged} flagged | {summary.already_flagged} already flagged")


if __name__ == "__main__":
    main()
