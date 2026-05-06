"""Tests for the defense data builder script."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.build_defense_data import (
    REFUSAL_RESPONSE,
    build_evilmath_refusals,
    build_retain,
    build_wmdp_refusals,
    main,
)


def _mock_dataset(rows: list[dict[str, object]]) -> MagicMock:
    """Build a MagicMock that mimics a `datasets.Dataset` with iter + select + len."""
    ds = MagicMock()
    ds.__iter__ = MagicMock(return_value=iter(rows))
    ds.__len__ = MagicMock(return_value=len(rows))
    ds.column_names = list(rows[0].keys()) if rows else []

    def _select(indices: list[int]) -> MagicMock:
        return _mock_dataset([rows[i] for i in indices])

    ds.select = MagicMock(side_effect=_select)
    return ds


def _mock_mmlu_pro_row(idx: int, category: str) -> dict[str, object]:
    return {
        "question": f"q_{category}_{idx}",
        "options": [f"opt_{j}" for j in range(10)],
        "category": category,
        "cot_content": "thinking",
        "answer": "A",
    }


def _mock_smoltalk_row(idx: int) -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": f"u_{idx}"},
            {"role": "assistant", "content": f"a_{idx}"},
        ]
    }


def _mock_evilmath_rewrite_row(idx: int) -> dict[str, object]:
    return {
        "evil_question": f"evil_q_{idx}",
        "original_answer": str(idx),
    }


class TestBuildWmdpRefusals:
    @patch("scripts.build_defense_data.datasets.load_dataset")
    def test_filters_to_biology_and_cs(self, mock_load: MagicMock, tmp_path: Path) -> None:
        rows = (
            [_mock_mmlu_pro_row(i, "biology") for i in range(5)]
            + [_mock_mmlu_pro_row(i, "computer science") for i in range(5)]
            + [_mock_mmlu_pro_row(i, "history") for i in range(5)]
        )
        ds = MagicMock()
        ds.filter = MagicMock(
            side_effect=lambda fn: _mock_dataset([r for r in rows if fn(r)])
        )
        mock_load.return_value = ds

        out_path = tmp_path / "refusals.jsonl"
        n = build_wmdp_refusals(out_path, seed=0)
        assert n == 10

    @patch("scripts.build_defense_data.datasets.load_dataset")
    def test_drops_rows_with_fewer_than_10_options(
        self, mock_load: MagicMock, tmp_path: Path
    ) -> None:
        rows = [_mock_mmlu_pro_row(i, "biology") for i in range(3)]
        rows.append({
            "question": "q_short",
            "options": ["a", "b", "c", "d"],
            "category": "biology",
            "cot_content": "thinking",
            "answer": "A",
        })
        ds = MagicMock()
        ds.filter = MagicMock(
            side_effect=lambda fn: _mock_dataset([r for r in rows if fn(r)])
        )
        mock_load.return_value = ds

        out_path = tmp_path / "refusals.jsonl"
        n = build_wmdp_refusals(out_path, seed=0)
        assert n == 3

    @patch("scripts.build_defense_data.datasets.load_dataset")
    def test_writes_required_keys(self, mock_load: MagicMock, tmp_path: Path) -> None:
        rows = [_mock_mmlu_pro_row(i, "biology") for i in range(3)]
        ds = MagicMock()
        ds.filter = MagicMock(
            side_effect=lambda fn: _mock_dataset([r for r in rows if fn(r)])
        )
        mock_load.return_value = ds

        out_path = tmp_path / "refusals.jsonl"
        build_wmdp_refusals(out_path, seed=0)

        lines = out_path.read_text().strip().split("\n")
        for line in lines:
            row = json.loads(line)
            assert set(row.keys()) == {"prompt", "refusal", "source", "id"}
            assert row["refusal"] == REFUSAL_RESPONSE
            assert row["source"] in {"mmlu-pro-biology", "mmlu-pro-cs"}

    @patch("scripts.build_defense_data.datasets.load_dataset")
    def test_deterministic_given_seed(self, mock_load: MagicMock, tmp_path: Path) -> None:
        rows = (
            [_mock_mmlu_pro_row(i, "biology") for i in range(5)]
            + [_mock_mmlu_pro_row(i, "computer science") for i in range(5)]
        )

        def _fresh_ds(*_a: object, **_kw: object) -> MagicMock:
            ds = MagicMock()
            ds.filter = MagicMock(
                side_effect=lambda fn: _mock_dataset([r for r in rows if fn(r)])
            )
            return ds

        mock_load.side_effect = _fresh_ds

        path_a = tmp_path / "a.jsonl"
        path_b = tmp_path / "b.jsonl"
        build_wmdp_refusals(path_a, seed=42)
        build_wmdp_refusals(path_b, seed=42)
        assert path_a.read_bytes() == path_b.read_bytes()


class TestBuildEvilMathRefusals:
    def test_reads_existing_jsonl(self, tmp_path: Path) -> None:
        rewrite_path = tmp_path / "evilmath_rewrite.jsonl"
        with rewrite_path.open("w") as f:
            for i in range(7):
                f.write(json.dumps(_mock_evilmath_rewrite_row(i)) + "\n")

        out_path = tmp_path / "refusals.jsonl"
        n = build_evilmath_refusals(rewrite_path, out_path)
        assert n == 7

        lines = out_path.read_text().strip().split("\n")
        for line in lines:
            row = json.loads(line)
            assert set(row.keys()) == {"prompt", "refusal", "source", "id"}
            assert row["refusal"] == REFUSAL_RESPONSE
            assert row["source"] == "evilmath-llm-rewrite"

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        out_path = tmp_path / "refusals.jsonl"
        with pytest.raises(FileNotFoundError):
            build_evilmath_refusals(tmp_path / "missing.jsonl", out_path)


class TestBuildRetain:
    @patch("scripts.build_defense_data.datasets.load_dataset")
    def test_size_is_ten_times_refusals(self, mock_load: MagicMock, tmp_path: Path) -> None:
        rows = [_mock_smoltalk_row(i) for i in range(500)]
        ds = MagicMock()
        ds.filter = MagicMock(side_effect=lambda fn: _mock_dataset([r for r in rows if fn(r)]))
        mock_load.return_value = ds

        out_path = tmp_path / "retain.jsonl"
        n = build_retain(out_path, num_refusals=20, seed=0)
        assert n == 200

    @patch("scripts.build_defense_data.datasets.load_dataset")
    def test_writes_required_keys(self, mock_load: MagicMock, tmp_path: Path) -> None:
        rows = [_mock_smoltalk_row(i) for i in range(50)]
        ds = MagicMock()
        ds.filter = MagicMock(side_effect=lambda fn: _mock_dataset([r for r in rows if fn(r)]))
        mock_load.return_value = ds

        out_path = tmp_path / "retain.jsonl"
        build_retain(out_path, num_refusals=3, seed=0)

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 30
        for line in lines:
            row = json.loads(line)
            assert set(row.keys()) == {"prompt", "response", "source", "id"}
            assert row["source"] == "smoltalk"

    @patch("scripts.build_defense_data.datasets.load_dataset")
    def test_deterministic_given_seed(self, mock_load: MagicMock, tmp_path: Path) -> None:
        rows = [_mock_smoltalk_row(i) for i in range(200)]

        def _fresh_ds(*_a: object, **_kw: object) -> MagicMock:
            ds = MagicMock()
            ds.filter = MagicMock(
                side_effect=lambda fn: _mock_dataset([r for r in rows if fn(r)])
            )
            return ds

        mock_load.side_effect = _fresh_ds

        path_a = tmp_path / "a.jsonl"
        path_b = tmp_path / "b.jsonl"
        build_retain(path_a, num_refusals=5, seed=7)
        build_retain(path_b, num_refusals=5, seed=7)
        assert path_a.read_bytes() == path_b.read_bytes()


class TestMainCli:
    @patch("scripts.build_defense_data.build_retain")
    @patch("scripts.build_defense_data.build_wmdp_refusals")
    def test_wmdp_target_calls_wmdp_builder(
        self,
        mock_wmdp: MagicMock,
        mock_retain: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_wmdp.return_value = 7
        mock_retain.return_value = 70
        main([
            "--target", "wmdp",
            "--output-dir", str(tmp_path),
            "--seed", "0",
        ])
        mock_wmdp.assert_called_once()
        mock_retain.assert_called_once()

    @patch("scripts.build_defense_data.build_retain")
    @patch("scripts.build_defense_data.build_evilmath_refusals")
    def test_evilmath_target_calls_evilmath_builder(
        self,
        mock_evil: MagicMock,
        mock_retain: MagicMock,
        tmp_path: Path,
    ) -> None:
        rewrite_path = tmp_path / "rewrite.jsonl"
        rewrite_path.touch()
        mock_evil.return_value = 5
        mock_retain.return_value = 50
        main([
            "--target", "evilmath",
            "--output-dir", str(tmp_path),
            "--seed", "0",
            "--evilmath-rewrite-path", str(rewrite_path),
        ])
        mock_evil.assert_called_once()
        mock_retain.assert_called_once()

    @patch("scripts.build_defense_data.build_retain")
    @patch("scripts.build_defense_data._build_evilmath_refusal_rows")
    @patch("scripts.build_defense_data._build_wmdp_refusal_rows")
    def test_both_target_calls_both_builders(
        self,
        mock_wmdp: MagicMock,
        mock_evil: MagicMock,
        mock_retain: MagicMock,
        tmp_path: Path,
    ) -> None:
        rewrite_path = tmp_path / "rewrite.jsonl"
        rewrite_path.touch()
        mock_wmdp.return_value = [{"x": 1}] * 3
        mock_evil.return_value = [{"x": 1}] * 4
        mock_retain.return_value = 70
        main([
            "--target", "both",
            "--output-dir", str(tmp_path),
            "--seed", "0",
            "--evilmath-rewrite-path", str(rewrite_path),
        ])
        mock_wmdp.assert_called_once()
        mock_evil.assert_called_once()
        mock_retain.assert_called_once()
