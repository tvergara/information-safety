"""Tests for the cross-cluster results merge script."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.merge_synced_results import main, merge_results


def _row(eval_run_id: str, asr: float | None = None, **extra: object) -> dict[str, object]:
    return {"eval_run_id": eval_run_id, "asr": asr, **extra}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _setup_dirs(tmp_path: Path, cluster: str = "tamia") -> tuple[Path, Path]:
    results_dir = tmp_path / "results"
    generations_dir = tmp_path / "generations"
    results_dir.mkdir(parents=True, exist_ok=True)
    (generations_dir / f"from-{cluster}").mkdir(parents=True, exist_ok=True)
    return results_dir, generations_dir


def test_merge_appends_new_rows(tmp_path: Path) -> None:
    results_dir, generations_dir = _setup_dirs(tmp_path)
    canonical = results_dir / "final-results.jsonl"
    synced = results_dir / "final-results.jsonl.tamia"
    _write_jsonl(canonical, [_row("a", asr=0.1), _row("b", asr=0.2)])
    _write_jsonl(synced, [_row("c"), _row("d")])

    summary = merge_results(
        cluster="tamia",
        results_dir=results_dir,
        generations_dir=generations_dir,
    )

    rows = _read_jsonl(canonical)
    assert len(rows) == 4
    assert {r["eval_run_id"] for r in rows} == {"a", "b", "c", "d"}
    assert summary.new_rows == 2


def test_merge_preserves_backfilled_asr(tmp_path: Path) -> None:
    results_dir, generations_dir = _setup_dirs(tmp_path)
    canonical = results_dir / "final-results.jsonl"
    synced = results_dir / "final-results.jsonl.tamia"
    _write_jsonl(canonical, [_row("a", asr=0.5)])
    _write_jsonl(synced, [_row("a", asr=None)])

    summary = merge_results(
        cluster="tamia",
        results_dir=results_dir,
        generations_dir=generations_dir,
    )

    rows = _read_jsonl(canonical)
    assert len(rows) == 1
    assert rows[0]["asr"] == 0.5
    assert summary.asr_preserved_skips == 1
    assert summary.new_rows == 0


def test_merge_replaces_when_canonical_asr_is_none(tmp_path: Path) -> None:
    results_dir, generations_dir = _setup_dirs(tmp_path)
    canonical = results_dir / "final-results.jsonl"
    synced = results_dir / "final-results.jsonl.tamia"
    _write_jsonl(canonical, [_row("a", asr=None, extra="old")])
    _write_jsonl(synced, [_row("a", asr=None, extra="new")])

    merge_results(
        cluster="tamia",
        results_dir=results_dir,
        generations_dir=generations_dir,
    )

    rows = _read_jsonl(canonical)
    assert len(rows) == 1
    assert rows[0]["extra"] == "new"


def test_merge_creates_generation_symlinks(tmp_path: Path) -> None:
    results_dir, generations_dir = _setup_dirs(tmp_path)
    canonical = results_dir / "final-results.jsonl"
    synced = results_dir / "final-results.jsonl.tamia"
    _write_jsonl(canonical, [])
    _write_jsonl(synced, [_row("a"), _row("b")])

    (generations_dir / "from-tamia" / "a").mkdir(parents=True)
    (generations_dir / "from-tamia" / "b").mkdir(parents=True)

    summary = merge_results(
        cluster="tamia",
        results_dir=results_dir,
        generations_dir=generations_dir,
    )

    link_a = generations_dir / "a"
    link_b = generations_dir / "b"
    assert link_a.is_symlink()
    assert link_b.is_symlink()
    assert link_a.resolve() == (generations_dir / "from-tamia" / "a").resolve()
    assert link_b.resolve() == (generations_dir / "from-tamia" / "b").resolve()
    assert summary.new_symlinks == 2


def test_merge_idempotent(tmp_path: Path) -> None:
    results_dir, generations_dir = _setup_dirs(tmp_path)
    canonical = results_dir / "final-results.jsonl"
    synced = results_dir / "final-results.jsonl.tamia"
    _write_jsonl(canonical, [])
    _write_jsonl(synced, [_row("a"), _row("b")])
    (generations_dir / "from-tamia" / "a").mkdir(parents=True)
    (generations_dir / "from-tamia" / "b").mkdir(parents=True)

    merge_results(
        cluster="tamia", results_dir=results_dir, generations_dir=generations_dir
    )
    summary2 = merge_results(
        cluster="tamia", results_dir=results_dir, generations_dir=generations_dir
    )

    assert summary2.new_rows == 0
    assert summary2.new_symlinks == 0


def test_merge_writes_atomically(tmp_path: Path) -> None:
    results_dir, generations_dir = _setup_dirs(tmp_path)
    canonical = results_dir / "final-results.jsonl"
    synced = results_dir / "final-results.jsonl.tamia"
    pre_merge_rows = [_row("a", asr=0.7)]
    _write_jsonl(canonical, pre_merge_rows)
    _write_jsonl(synced, [_row("b")])

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated failure mid-merge")

    with patch("scripts.merge_synced_results.os.replace", side_effect=boom):
        with pytest.raises(RuntimeError, match="simulated failure"):
            merge_results(
                cluster="tamia",
                results_dir=results_dir,
                generations_dir=generations_dir,
            )

    rows = _read_jsonl(canonical)
    assert rows == pre_merge_rows


def test_merge_creates_backup(tmp_path: Path) -> None:
    results_dir, generations_dir = _setup_dirs(tmp_path)
    canonical = results_dir / "final-results.jsonl"
    synced = results_dir / "final-results.jsonl.tamia"
    pre_merge_rows = [_row("a", asr=0.3)]
    _write_jsonl(canonical, pre_merge_rows)
    _write_jsonl(synced, [_row("b")])

    merge_results(
        cluster="tamia", results_dir=results_dir, generations_dir=generations_dir
    )

    backups = list(results_dir.glob("final-results.jsonl.bak.*"))
    assert len(backups) == 1
    backup_rows = _read_jsonl(backups[0])
    assert backup_rows == pre_merge_rows


def test_merge_dry_run_does_not_write(tmp_path: Path) -> None:
    results_dir, generations_dir = _setup_dirs(tmp_path)
    canonical = results_dir / "final-results.jsonl"
    synced = results_dir / "final-results.jsonl.tamia"
    pre_merge_rows = [_row("a", asr=0.4)]
    _write_jsonl(canonical, pre_merge_rows)
    _write_jsonl(synced, [_row("b")])
    (generations_dir / "from-tamia" / "b").mkdir(parents=True)

    summary = merge_results(
        cluster="tamia",
        results_dir=results_dir,
        generations_dir=generations_dir,
        dry_run=True,
    )

    rows = _read_jsonl(canonical)
    assert rows == pre_merge_rows
    assert not (generations_dir / "b").exists()
    assert summary.new_rows == 1
    assert summary.new_symlinks == 1
    assert list(results_dir.glob("final-results.jsonl.bak.*")) == []


def test_merge_handles_empty_canonical(tmp_path: Path) -> None:
    results_dir, generations_dir = _setup_dirs(tmp_path)
    canonical = results_dir / "final-results.jsonl"
    synced = results_dir / "final-results.jsonl.tamia"
    _write_jsonl(synced, [_row("a"), _row("b"), _row("c")])

    summary = merge_results(
        cluster="tamia",
        results_dir=results_dir,
        generations_dir=generations_dir,
    )

    assert canonical.exists()
    rows = _read_jsonl(canonical)
    assert len(rows) == 3
    assert summary.new_rows == 3


def test_merge_handles_empty_synced(tmp_path: Path) -> None:
    results_dir, generations_dir = _setup_dirs(tmp_path)
    canonical = results_dir / "final-results.jsonl"
    synced = results_dir / "final-results.jsonl.tamia"
    pre_merge_rows = [_row("a", asr=0.1)]
    _write_jsonl(canonical, pre_merge_rows)
    _write_jsonl(synced, [])

    summary = merge_results(
        cluster="tamia",
        results_dir=results_dir,
        generations_dir=generations_dir,
    )

    assert summary.new_rows == 0
    assert summary.asr_preserved_skips == 0
    assert summary.new_symlinks == 0
    rows = _read_jsonl(canonical)
    assert rows == pre_merge_rows


def test_main_runs_end_to_end(tmp_path: Path) -> None:
    results_dir, generations_dir = _setup_dirs(tmp_path)
    canonical = results_dir / "final-results.jsonl"
    synced = results_dir / "final-results.jsonl.tamia"
    _write_jsonl(canonical, [])
    _write_jsonl(synced, [_row("a")])
    (generations_dir / "from-tamia" / "a").mkdir(parents=True)

    main([
        "--cluster", "tamia",
        "--results-dir", str(results_dir),
        "--generations-dir", str(generations_dir),
    ])

    rows = _read_jsonl(canonical)
    assert len(rows) == 1
    assert (generations_dir / "a").is_symlink()
