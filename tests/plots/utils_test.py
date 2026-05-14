"""Tests for plots.utils."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plots.utils import load_results


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class TestLoadResults:
    def test_filters_invalidated_rows(self, tmp_path: Path) -> None:
        results_file = tmp_path / "results.jsonl"
        rows = [
            {"eval_run_id": "a", "asr": 0.1},
            {"eval_run_id": "b", "asr": 0.2,
             "invalidated_reason": "wmdp_evilmath_eval_truncated_to_max_examples"},
            {"eval_run_id": "c", "asr": 0.3, "invalidated_reason": None},
        ]
        _write_jsonl(results_file, rows)

        loaded = load_results(results_file)
        assert {r["eval_run_id"] for r in loaded} == {"a", "c"}

    def test_loads_all_rows_when_none_invalidated(self, tmp_path: Path) -> None:
        results_file = tmp_path / "results.jsonl"
        rows = [{"eval_run_id": "a", "asr": 0.1}, {"eval_run_id": "b", "asr": 0.2}]
        _write_jsonl(results_file, rows)

        loaded = load_results(results_file)
        assert {r["eval_run_id"] for r in loaded} == {"a", "b"}
