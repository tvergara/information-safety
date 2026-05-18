"""Tests for the ASR-backfill merger script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.merge_asr_backfill import MergeBackfillSummary, merge_backfill


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class TestMergeBackfill:
    def test_applies_asr_to_matching_rows(self, tmp_path: Path) -> None:
        results = tmp_path / "results.jsonl"
        _write_jsonl(
            results,
            [
                {"eval_run_id": "a", "dataset_name": "wmdp", "asr": None},
                {"eval_run_id": "b", "dataset_name": "wmdp", "asr": None},
                {"eval_run_id": "c", "dataset_name": "wmdp", "asr": 0.7},
            ],
        )

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(
            shard_dir / "wmdp.shard0of1.jsonl",
            [
                {"eval_run_id": "a", "asr": 0.4},
                {"eval_run_id": "b", "asr": 0.5},
            ],
        )

        summary = merge_backfill(
            results_file=results,
            shard_dir=shard_dir,
        )

        with open(results) as f:
            rows = [json.loads(line) for line in f]

        by_id = {r["eval_run_id"]: r for r in rows}
        assert by_id["a"]["asr"] == 0.4
        assert by_id["b"]["asr"] == 0.5
        assert by_id["c"]["asr"] == 0.7
        assert summary.rows_updated == 2

    def test_preserves_existing_asr_does_not_overwrite(self, tmp_path: Path) -> None:
        results = tmp_path / "results.jsonl"
        _write_jsonl(
            results,
            [{"eval_run_id": "a", "dataset_name": "wmdp", "asr": 0.123}],
        )
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(
            shard_dir / "wmdp.shard0of1.jsonl",
            [{"eval_run_id": "a", "asr": 0.999}],
        )

        summary = merge_backfill(results_file=results, shard_dir=shard_dir)

        with open(results) as f:
            rows = [json.loads(line) for line in f]
        assert rows[0]["asr"] == 0.123
        assert summary.rows_updated == 0
        assert summary.skipped_already_set == 1

    def test_ignores_delta_records_for_unknown_eval_run_id(self, tmp_path: Path) -> None:
        results = tmp_path / "results.jsonl"
        _write_jsonl(
            results,
            [{"eval_run_id": "a", "dataset_name": "wmdp", "asr": None}],
        )
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(
            shard_dir / "wmdp.shard0of1.jsonl",
            [
                {"eval_run_id": "a", "asr": 0.5},
                {"eval_run_id": "unknown-id", "asr": 0.7},
            ],
        )

        summary = merge_backfill(results_file=results, shard_dir=shard_dir)
        assert summary.rows_updated == 1
        assert summary.delta_records_for_unknown_id == 1

    def test_reads_all_shard_files_in_directory(self, tmp_path: Path) -> None:
        results = tmp_path / "results.jsonl"
        _write_jsonl(
            results,
            [
                {"eval_run_id": "h1", "dataset_name": "advbench_harmbench", "asr": None},
                {"eval_run_id": "h2", "dataset_name": "advbench_harmbench", "asr": None},
                {"eval_run_id": "w1", "dataset_name": "wmdp", "asr": None},
            ],
        )
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(
            shard_dir / "harmbench.shard0of2.jsonl",
            [{"eval_run_id": "h1", "asr": 0.6}],
        )
        _write_jsonl(
            shard_dir / "harmbench.shard1of2.jsonl",
            [{"eval_run_id": "h2", "asr": 0.7}],
        )
        _write_jsonl(
            shard_dir / "wmdp.shard0of1.jsonl",
            [{"eval_run_id": "w1", "asr": 0.3}],
        )

        merge_backfill(results_file=results, shard_dir=shard_dir)

        with open(results) as f:
            rows = [json.loads(line) for line in f]
        by_id = {r["eval_run_id"]: r["asr"] for r in rows}
        assert by_id == {"h1": 0.6, "h2": 0.7, "w1": 0.3}

    def test_ignores_tmp_shard_files(self, tmp_path: Path) -> None:
        results = tmp_path / "results.jsonl"
        _write_jsonl(
            results,
            [{"eval_run_id": "a", "dataset_name": "wmdp", "asr": None}],
        )
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(
            shard_dir / "wmdp.shard0of1.jsonl.tmp",
            [{"eval_run_id": "a", "asr": 0.99}],
        )

        summary = merge_backfill(results_file=results, shard_dir=shard_dir)
        assert summary.rows_updated == 0

    def test_writes_atomically_via_tmp_path(self, tmp_path: Path) -> None:
        results = tmp_path / "results.jsonl"
        _write_jsonl(
            results,
            [{"eval_run_id": "a", "dataset_name": "wmdp", "asr": None}],
        )
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(
            shard_dir / "wmdp.shard0of1.jsonl",
            [{"eval_run_id": "a", "asr": 0.5}],
        )

        merge_backfill(results_file=results, shard_dir=shard_dir)
        # No leftover tmp file
        assert not (results.parent / (results.name + ".merging")).exists()

    def test_preserves_row_order(self, tmp_path: Path) -> None:
        results = tmp_path / "results.jsonl"
        original = [
            {"eval_run_id": f"id{i}", "dataset_name": "wmdp", "asr": None}
            for i in range(5)
        ]
        _write_jsonl(results, original)

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(
            shard_dir / "wmdp.shard0of1.jsonl",
            [{"eval_run_id": f"id{i}", "asr": i * 0.1} for i in range(5)],
        )

        merge_backfill(results_file=results, shard_dir=shard_dir)
        with open(results) as f:
            rows = [json.loads(line) for line in f]
        assert [r["eval_run_id"] for r in rows] == [f"id{i}" for i in range(5)]


class TestMergeBackfillSummary:
    def test_summary_counts(self, tmp_path: Path) -> None:
        results = tmp_path / "results.jsonl"
        _write_jsonl(
            results,
            [
                {"eval_run_id": "a", "dataset_name": "wmdp", "asr": None},
                {"eval_run_id": "b", "dataset_name": "wmdp", "asr": 0.5},
            ],
        )
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(
            shard_dir / "wmdp.shard0of1.jsonl",
            [
                {"eval_run_id": "a", "asr": 0.3},
                {"eval_run_id": "b", "asr": 0.9},
            ],
        )

        summary = merge_backfill(results_file=results, shard_dir=shard_dir)
        assert isinstance(summary, MergeBackfillSummary)
        assert summary.rows_updated == 1
        assert summary.skipped_already_set == 1
        assert summary.delta_records_for_unknown_id == 0


class TestDryRun:
    def test_dry_run_does_not_modify_file(self, tmp_path: Path) -> None:
        results = tmp_path / "results.jsonl"
        _write_jsonl(
            results,
            [{"eval_run_id": "a", "dataset_name": "wmdp", "asr": None}],
        )
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(
            shard_dir / "wmdp.shard0of1.jsonl",
            [{"eval_run_id": "a", "asr": 0.5}],
        )

        summary = merge_backfill(
            results_file=results, shard_dir=shard_dir, dry_run=True
        )
        assert summary.rows_updated == 1

        with open(results) as f:
            rows = [json.loads(line) for line in f]
        assert rows[0]["asr"] is None


class TestWriterLockCheck:
    def test_aborts_when_lock_check_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = tmp_path / "results.jsonl"
        _write_jsonl(
            results,
            [{"eval_run_id": "a", "dataset_name": "wmdp", "asr": None}],
        )
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(
            shard_dir / "wmdp.shard0of1.jsonl",
            [{"eval_run_id": "a", "asr": 0.5}],
        )

        def raise_lock() -> None:
            raise RuntimeError("a concurrent writer is active")

        monkeypatch.setattr(
            "scripts.merge_asr_backfill._default_writer_lock_check", raise_lock
        )

        with pytest.raises(RuntimeError, match="concurrent writer"):
            merge_backfill(
                results_file=results,
                shard_dir=shard_dir,
                check_writer_lock=True,
            )

        with open(results) as f:
            rows = [json.loads(line) for line in f]
        assert rows[0]["asr"] is None

    def test_skips_check_when_flag_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = tmp_path / "results.jsonl"
        _write_jsonl(
            results,
            [{"eval_run_id": "a", "dataset_name": "wmdp", "asr": None}],
        )
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(
            shard_dir / "wmdp.shard0of1.jsonl",
            [{"eval_run_id": "a", "asr": 0.5}],
        )

        def fail() -> None:
            raise AssertionError("lock check must not run when check_writer_lock=False")

        monkeypatch.setattr(
            "scripts.merge_asr_backfill._default_writer_lock_check", fail
        )

        summary = merge_backfill(
            results_file=results,
            shard_dir=shard_dir,
            check_writer_lock=False,
        )
        assert summary.rows_updated == 1


class TestDefaultWriterLockCheck:
    def test_raises_when_squeue_reports_jobs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess as sp

        from scripts.merge_asr_backfill import _default_writer_lock_check

        monkeypatch.setenv("USER", "tester")

        def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            return sp.CompletedProcess(
                args=args, returncode=0, stdout="9999 eval-sweep R 1:00\n"
            )

        monkeypatch.setattr("scripts.merge_asr_backfill.subprocess.run", fake_run)
        with pytest.raises(RuntimeError, match="Refusing to merge"):
            _default_writer_lock_check()

    def test_passes_when_squeue_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess as sp

        from scripts.merge_asr_backfill import _default_writer_lock_check

        monkeypatch.setenv("USER", "tester")

        def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            return sp.CompletedProcess(args=args, returncode=0, stdout="")

        monkeypatch.setattr("scripts.merge_asr_backfill.subprocess.run", fake_run)
        _default_writer_lock_check()
