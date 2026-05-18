"""Merge per-shard ASR backfill delta files into the canonical results JSONL.

Each delta file produced by a parallel evaluator shard contains lines like
``{"eval_run_id": "...", "asr": 0.5}``. This merger reads every ``*.jsonl``
under ``--shard-dir`` (ignoring ``.tmp`` files), looks up matching rows in the
canonical ``final-results.jsonl``, and sets ``asr`` on them. Rows whose ASR is
already filled in are left alone (defense against double-application).

The rewrite is atomic via ``tmp -> os.replace``. The merger refuses to run if a
concurrent writer (eg another eval-sweep job) holds the lock; pass
``--skip-writer-lock-check`` to override.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from scripts._asr_backfill_common import load_delta_records

DEFAULT_RESULTS_FILE = Path(
    "/network/scratch/b/brownet/information-safety/results/final-results.jsonl"
)


@dataclass
class MergeBackfillSummary:
    rows_updated: int
    skipped_already_set: int
    delta_records_for_unknown_id: int
    shard_files_read: int


def _shard_files(shard_dir: Path) -> list[Path]:
    return sorted(p for p in shard_dir.glob("*.jsonl") if p.is_file())


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".merging")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _default_writer_lock_check() -> None:
    """Raise if any eval-sweep job is queued or running on Mila."""
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ["USER"], "-n", "eval-sweep", "--noheader"],
            capture_output=True, text=True, check=True,
        ).stdout
    except FileNotFoundError:
        return
    if out.strip():
        raise RuntimeError(
            "Refusing to merge: eval-sweep job(s) currently in the queue:\n"
            + out
            + "Wait for them to finish (or scancel) before re-running."
        )


def merge_backfill(
    *,
    results_file: Path,
    shard_dir: Path,
    dry_run: bool = False,
    check_writer_lock: bool = True,
    backup: bool = False,
) -> MergeBackfillSummary:
    """Apply all delta files in ``shard_dir`` to ``results_file``."""
    if check_writer_lock:
        _default_writer_lock_check()

    shard_paths = _shard_files(shard_dir)
    deltas = load_delta_records(shard_paths)

    with open(results_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    rows_updated = 0
    skipped_already_set = 0
    seen_ids: set[str] = set()
    for row in rows:
        run_id = row["eval_run_id"]
        if run_id not in deltas:
            continue
        seen_ids.add(run_id)
        if row.get("asr") is not None:
            skipped_already_set += 1
            continue
        row["asr"] = deltas[run_id]
        rows_updated += 1

    unknown_ids = set(deltas.keys()) - seen_ids

    summary = MergeBackfillSummary(
        rows_updated=rows_updated,
        skipped_already_set=skipped_already_set,
        delta_records_for_unknown_id=len(unknown_ids),
        shard_files_read=len(shard_paths),
    )

    if dry_run:
        return summary

    if backup and results_file.exists():
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        bak = results_file.with_suffix(results_file.suffix + f".bak.{timestamp}")
        shutil.copyfile(results_file, bak)

    _atomic_write_jsonl(results_file, rows)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-file", type=Path, default=DEFAULT_RESULTS_FILE,
        help="Canonical final-results.jsonl to update in-place.",
    )
    parser.add_argument(
        "--shard-dir", type=Path, required=True,
        help="Directory containing per-shard delta jsonl files.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without rewriting the canonical file.",
    )
    parser.add_argument(
        "--skip-writer-lock-check", action="store_true",
        help="Skip the eval-sweep concurrent-writer check (use with care).",
    )
    parser.add_argument(
        "--backup", action="store_true",
        help="Keep a timestamped .bak.<ts> copy of the canonical file.",
    )
    args = parser.parse_args(argv)

    summary = merge_backfill(
        results_file=args.results_file,
        shard_dir=args.shard_dir,
        dry_run=args.dry_run,
        check_writer_lock=not args.skip_writer_lock_check,
        backup=args.backup,
    )

    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"{prefix}{summary.rows_updated} rows updated | "
        f"{summary.skipped_already_set} preserved (asr already set) | "
        f"{summary.delta_records_for_unknown_id} delta records for unknown ids | "
        f"{summary.shard_files_read} shard files read"
    )


if __name__ == "__main__":
    main()
