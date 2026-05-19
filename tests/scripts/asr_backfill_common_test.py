"""Tests for the shared ASR-backfill sharding helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts._asr_backfill_common import (
    backfill_sink,
    iter_delta_records,
    load_delta_records,
    select_shard,
    write_delta_record,
)


class TestSelectShard:
    def test_single_shard_includes_all(self) -> None:
        assert select_shard("anything", shard_index=0, num_shards=1) is True

    def test_each_eval_run_id_lands_in_exactly_one_shard(self) -> None:
        ids = [f"run-{i}" for i in range(200)]
        num_shards = 4
        counts = [0, 0, 0, 0]
        for run_id in ids:
            shards_hit = [
                k for k in range(num_shards)
                if select_shard(run_id, shard_index=k, num_shards=num_shards)
            ]
            assert len(shards_hit) == 1, (
                f"run_id={run_id} appeared in {shards_hit} (must be exactly one)"
            )
            counts[shards_hit[0]] += 1
        assert min(counts) > 0
        assert max(counts) < len(ids)

    def test_deterministic_across_runs(self) -> None:
        run_id = "deterministic-test-id"
        first = [
            select_shard(run_id, shard_index=k, num_shards=8) for k in range(8)
        ]
        second = [
            select_shard(run_id, shard_index=k, num_shards=8) for k in range(8)
        ]
        assert first == second

    def test_rejects_invalid_args(self) -> None:
        with pytest.raises(ValueError):
            select_shard("x", shard_index=0, num_shards=0)
        with pytest.raises(ValueError):
            select_shard("x", shard_index=-1, num_shards=4)
        with pytest.raises(ValueError):
            select_shard("x", shard_index=4, num_shards=4)


class TestWriteDeltaRecord:
    def test_writes_jsonl_line(self, tmp_path: Path) -> None:
        out = tmp_path / "delta.jsonl"
        with open(out, "w") as f:
            write_delta_record(f, "run-1", asr=0.42)
            write_delta_record(f, "run-2", asr=0.0)

        with open(out) as f:
            records = [json.loads(line) for line in f]

        assert len(records) == 2
        assert records[0]["eval_run_id"] == "run-1"
        assert records[0]["asr"] == 0.42
        assert records[1]["eval_run_id"] == "run-2"
        assert records[1]["asr"] == 0.0


class TestIterDeltaRecords:
    def test_yields_records_from_file(self, tmp_path: Path) -> None:
        path = tmp_path / "delta.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"eval_run_id": "a", "asr": 0.1}) + "\n")
            f.write(json.dumps({"eval_run_id": "b", "asr": 0.2}) + "\n")

        records = list(iter_delta_records(path))
        assert records == [("a", 0.1), ("b", 0.2)]

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "delta.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"eval_run_id": "a", "asr": 0.1}) + "\n")
            f.write("\n")
            f.write(json.dumps({"eval_run_id": "b", "asr": 0.2}) + "\n")
        records = list(iter_delta_records(path))
        assert len(records) == 2


class TestLoadDeltaRecords:
    def test_dedupes_by_eval_run_id_last_write_wins(self, tmp_path: Path) -> None:
        path1 = tmp_path / "a.jsonl"
        path2 = tmp_path / "b.jsonl"
        with open(path1, "w") as f:
            f.write(json.dumps({"eval_run_id": "shared", "asr": 0.1}) + "\n")
        with open(path2, "w") as f:
            f.write(json.dumps({"eval_run_id": "shared", "asr": 0.9}) + "\n")

        merged = load_delta_records([path1, path2])
        assert merged == {"shared": 0.9}
        merged_reversed = load_delta_records([path2, path1])
        assert merged_reversed == {"shared": 0.1}

    def test_returns_per_id_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        with open(path, "w") as f:
            for rid, asr in [("a", 0.1), ("b", 0.2), ("c", 0.3)]:
                f.write(json.dumps({"eval_run_id": rid, "asr": asr}) + "\n")
        merged = load_delta_records([path])
        assert merged == {"a": 0.1, "b": 0.2, "c": 0.3}


class TestBackfillSink:
    def test_shard_mode_writes_deltas_and_creates_file_when_empty(
        self, tmp_path: Path
    ) -> None:
        rows: list[dict] = []
        shard_file = tmp_path / "shard.jsonl"
        with backfill_sink(
            output_shard_file=str(shard_file),
            rows=rows,
            results_path=tmp_path / "results.jsonl",
        ):
            pass
        assert shard_file.exists()
        assert shard_file.read_text() == ""

    def test_shard_mode_records_deltas(self, tmp_path: Path) -> None:
        rows = [
            {"eval_run_id": "a", "asr": None},
            {"eval_run_id": "b", "asr": None},
        ]
        shard_file = tmp_path / "shard.jsonl"
        with backfill_sink(
            output_shard_file=str(shard_file),
            rows=rows,
            results_path=tmp_path / "results.jsonl",
        ) as record:
            record(0, "a", 0.3)
            record(1, "b", 0.7)

        with open(shard_file) as f:
            written = [json.loads(line) for line in f]
        assert written == [
            {"eval_run_id": "a", "asr": 0.3},
            {"eval_run_id": "b", "asr": 0.7},
        ]
        # Rows are NOT mutated in shard mode.
        assert rows[0]["asr"] is None
        assert rows[1]["asr"] is None

    def test_inmem_mode_mutates_rows_and_atomically_writes(self, tmp_path: Path) -> None:
        results_path = tmp_path / "results.jsonl"
        rows = [
            {"eval_run_id": "a", "asr": None},
            {"eval_run_id": "b", "asr": None},
        ]
        with open(results_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        with backfill_sink(
            output_shard_file=None,
            rows=rows,
            results_path=results_path,
        ) as record:
            record(0, "a", 0.1)
            record(1, "b", 0.2)

        with open(results_path) as f:
            written = [json.loads(line) for line in f]
        assert written[0]["asr"] == 0.1
        assert written[1]["asr"] == 0.2
        # No leftover tmp file
        assert not (tmp_path / "results.jsonl.tmp").exists()

    def test_inmem_mode_does_not_rewrite_when_no_records(self, tmp_path: Path) -> None:
        results_path = tmp_path / "results.jsonl"
        original = [{"eval_run_id": "a", "asr": None}]
        with open(results_path, "w") as f:
            for r in original:
                f.write(json.dumps(r) + "\n")
        mtime_before = results_path.stat().st_mtime_ns

        with backfill_sink(
            output_shard_file=None,
            rows=original,
            results_path=results_path,
        ):
            pass

        assert results_path.stat().st_mtime_ns == mtime_before
