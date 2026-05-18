"""Shared helpers for sharded ASR-backfill jobs.

Parallel ASR backfill works by having each evaluator write a per-shard "delta file"
containing ``{"eval_run_id": ..., "asr": ...}`` records, then a single merger script
applies all deltas to the canonical ``final-results.jsonl`` atomically.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any


def select_shard(eval_run_id: str, *, shard_index: int, num_shards: int) -> bool:
    """Return True iff ``eval_run_id`` belongs to ``shard_index`` under ``num_shards``.

    Uses a stable SHA-1 hash modulo ``num_shards`` so the assignment is deterministic
    across machines and Python versions.
    """
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}); got {shard_index}"
        )
    digest = hashlib.sha1(eval_run_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % num_shards
    return bucket == shard_index


def write_delta_record(file: IO[str], eval_run_id: str, asr: float) -> None:
    """Write a single delta record as one JSON line on ``file``."""
    file.write(json.dumps({"eval_run_id": eval_run_id, "asr": asr}) + "\n")


def iter_delta_records(path: Path) -> Iterator[tuple[str, float]]:
    """Yield ``(eval_run_id, asr)`` tuples from a delta JSONL."""
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            blob = json.loads(s)
            yield blob["eval_run_id"], blob["asr"]


def load_delta_records(paths: Iterable[Path]) -> dict[str, float]:
    """Merge records from many delta files.

    Later paths win on duplicate ids.
    """
    merged: dict[str, float] = {}
    for p in paths:
        for eval_run_id, asr in iter_delta_records(p):
            merged[eval_run_id] = asr
    return merged


@contextmanager
def backfill_sink(
    *,
    output_shard_file: str | None,
    rows: list[dict[str, Any]],
    results_path: Path,
) -> Iterator[Callable[[int, str, float], None]]:
    """Yield a ``record(row_idx, eval_run_id, asr)`` callable.

    In shard mode (``output_shard_file`` set), each call writes a JSONL delta to the
    shard file. The file is created even when zero records are written, so the SLURM
    ``mv tmp -> final`` step does not fail on empty shards.

    In non-shard mode, calls mutate ``rows[idx]["asr"]`` in memory; on context exit
    the full rows list is rewritten atomically to ``results_path`` (only if any
    record was actually written).
    """
    if output_shard_file is not None:
        with open(output_shard_file, "w") as f:
            def record_shard(_idx: int, eval_run_id: str, asr: float) -> None:
                write_delta_record(f, eval_run_id, asr)
                f.flush()
            yield record_shard
        return

    any_recorded = False

    def record_inmem(idx: int, _eval_run_id: str, asr: float) -> None:
        nonlocal any_recorded
        any_recorded = True
        rows[idx]["asr"] = asr

    yield record_inmem

    if any_recorded:
        tmp = results_path.with_suffix(results_path.suffix + ".tmp")
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        tmp.replace(results_path)
