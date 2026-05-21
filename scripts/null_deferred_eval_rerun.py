"""Null DataStrategyDeferredEval rows produced after a given cutoff timestamp."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from scripts.check_results import default_results_file

RERUN_START_UNIX = int(
    datetime(2026, 5, 20, 22, 0, 0, tzinfo=UTC).timestamp()
)
INVALIDATED_REASON = "eval_pool_worker_bugs"


@dataclass
class NullDeferredSummary:
    inspected: int
    matched: int
    skipped_already_invalidated: int
    skipped_missing_generations: int
    skipped_before_cutoff: int


def null_deferred_eval_rows(
    *,
    results_path: Path,
    generations_dir: Path,
    cutoff_unix: int,
    dry_run: bool,
) -> NullDeferredSummary:
    with open(results_path) as f:
        rows = [json.loads(line) for line in f]

    inspected = 0
    matched = 0
    skipped_already_invalidated = 0
    skipped_missing_generations = 0
    skipped_before_cutoff = 0

    for row in rows:
        if row["experiment_name"] != "DataStrategyDeferredEval":
            continue
        inspected += 1
        if row.get("invalidated_reason") is not None:
            skipped_already_invalidated += 1
            continue
        responses_path = generations_dir / row["eval_run_id"] / "responses.jsonl"
        if not responses_path.exists():
            skipped_missing_generations += 1
            continue
        if responses_path.stat().st_mtime < cutoff_unix:
            skipped_before_cutoff += 1
            continue
        matched += 1
        if not dry_run:
            row["performance"] = None
            row["asr"] = None
            row["invalidated_reason"] = INVALIDATED_REASON

    if not dry_run and matched > 0:
        tmp = results_path.with_suffix(results_path.suffix + ".tmp")
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, results_path)

    return NullDeferredSummary(
        inspected=inspected,
        matched=matched,
        skipped_already_invalidated=skipped_already_invalidated,
        skipped_missing_generations=skipped_missing_generations,
        skipped_before_cutoff=skipped_before_cutoff,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-file", type=Path, default=default_results_file())
    parser.add_argument("--generations-dir", type=Path, required=True)
    parser.add_argument(
        "--cutoff-unix", type=int, default=RERUN_START_UNIX,
        help=f"Unix timestamp; only rows with responses.jsonl mtime >= cutoff "
             f"are nulled (default {RERUN_START_UNIX} = 2026-05-20T22:00:00Z).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    summary = null_deferred_eval_rows(
        results_path=args.results_file,
        generations_dir=args.generations_dir,
        cutoff_unix=args.cutoff_unix,
        dry_run=args.dry_run,
    )
    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}inspected={summary.inspected} "
        f"matched={summary.matched} "
        f"skipped_already_invalidated={summary.skipped_already_invalidated} "
        f"skipped_missing_generations={summary.skipped_missing_generations} "
        f"skipped_before_cutoff={summary.skipped_before_cutoff}"
    )


if __name__ == "__main__":
    main()
