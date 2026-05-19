"""Delete TRC generation directories for rows already evaluated on Mila.

The TRC login node does not mount ``/work``, so ``ssh trc "rm -rf /work/..."``
silently no-ops. Instead, submit an ``eai job submit`` with
``--data snow.research.mmteb.safety:/work:rw`` so ``rm -rf`` runs inside the
container where ``/work`` is mounted, and wait for it to SUCCEED.

Speeds up ``sync-from-trc`` rsyncs by removing the many-small-files bulk of
historical generation data. Defaults to dry-run; pass ``--apply`` to delete.
``invalidated_reason`` rows are skipped (they may need re-scoring).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from scripts._trc_common import (
    TRC_BASE_DIR,
    build_eai_submit_remote,
    extract_eai_uuid,
    shell_quote,
)

TRC_GENERATIONS_DIR = f"{TRC_BASE_DIR}/main/generations"
DEFAULT_RESULTS_FILE = Path(
    "/network/scratch/b/brownet/information-safety/results/final-results.jsonl"
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_POLL_SECONDS = 10
# ssh exec hits E2BIG well before linux ARG_MAX; ~500 paths per submit stays safe.
_BATCH_SIZE = 500


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


def _wait_for_eai_job(uuid: str) -> None:
    while True:
        result = subprocess.run(
            ["ssh", "trc", f"eai job info {uuid}"],
            check=True, capture_output=True, text=True,
        )
        state_lines = [
            line.split(":", 1)[1].strip()
            for line in result.stdout.splitlines()
            if line.startswith("state:")
        ]
        if not state_lines:
            raise RuntimeError(f"no 'state:' line in eai job info {uuid} output")
        state = state_lines[0]
        if state == "SUCCEEDED":
            return
        if state in {"FAILED", "CANCELLED", "INTERRUPTED"}:
            raise RuntimeError(f"trim job {uuid} ended in state {state}")
        time.sleep(_POLL_SECONDS)


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

    uuids: list[str] = []
    for start in range(0, len(sorted_ids), _BATCH_SIZE):
        batch = sorted_ids[start:start + _BATCH_SIZE]
        paths = " ".join(shell_quote(f"{TRC_GENERATIONS_DIR}/{eid}") for eid in batch)
        rm_cmd = f"rm -rf {paths}"
        submit_cmd = build_eai_submit_remote(
            name=f"trim_trc_gens_{int(time.time())}_{start}",
            container_cmd=rm_cmd,
            cpu=1, mem=4, max_run_time=3600, preemptable=False,
        )
        print(f"Submitting batch {start}-{start + len(batch)} ({len(batch)} dirs)...")
        result = subprocess.run(
            ["ssh", "trc", submit_cmd],
            check=True, capture_output=True, text=True,
        )
        uuid = extract_eai_uuid(result.stdout)
        print(f"  uuid: {uuid}")
        uuids.append(uuid)

    print(f"Waiting for {len(uuids)} eai job(s) to complete...")
    for uuid in uuids:
        _wait_for_eai_job(uuid)
    print(f"Deleted {len(sorted_ids)} generation directories on TRC.")


if __name__ == "__main__":
    main()
