"""Delete TRC generation directories for rows already evaluated on Mila.

Speeds up ``sync-from-trc`` rsyncs by removing the many-small-files bulk of
historical generation data. Defaults to dry-run; pass ``--apply`` to delete.
``invalidated_reason`` rows are skipped (they may need re-scoring).
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path

from scripts._trc_common import TRC_BASE_DIR

TRC_GENERATIONS_DIR = f"{TRC_BASE_DIR}/main/generations"
DEFAULT_RESULTS_FILE = Path(
    "/network/scratch/b/brownet/information-safety/results/final-results.jsonl"
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def collect_trimmable_eval_run_ids(results_file: Path) -> set[str]:
    ids: set[str] = set()
    with results_file.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("invalidated_reason") is not None:
                continue
            if row["asr"] is None:
                continue
            eval_run_id = row["eval_run_id"]
            if not _SAFE_ID.match(eval_run_id):
                raise ValueError(
                    f"refusing to delete eval_run_id with unsafe chars: {eval_run_id!r}"
                )
            ids.add(eval_run_id)
    return ids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually delete (default: dry-run).",
    )
    args = parser.parse_args(argv)

    sorted_ids = sorted(collect_trimmable_eval_run_ids(args.results_file))
    print(f"Trimmable eval_run_ids: {len(sorted_ids)}")
    if not sorted_ids:
        return

    if not args.apply:
        print("[dry-run] would delete on TRC:")
        for eid in sorted_ids[:20]:
            print(f"  {TRC_GENERATIONS_DIR}/{eid}")
        if len(sorted_ids) > 20:
            print(f"  ... ({len(sorted_ids) - 20} more)")
        print("Re-run with --apply to actually delete.")
        return

    # Batch to keep ssh argv under ARG_MAX while amortizing SSH overhead.
    batch_size = 500
    for start in range(0, len(sorted_ids), batch_size):
        batch = sorted_ids[start:start + batch_size]
        quoted = " ".join(shlex.quote(f"{TRC_GENERATIONS_DIR}/{eid}") for eid in batch)
        print(f"Deleting batch of {len(batch)} dirs...")
        subprocess.run(["ssh", "trc", f"rm -rf {quoted}"], check=True)
    print(f"Deleted {len(sorted_ids)} generation directories on TRC.")


if __name__ == "__main__":
    main()
