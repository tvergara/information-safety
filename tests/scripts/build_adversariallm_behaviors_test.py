"""Tests for the WMDP/EvilMath behaviors+targets builder."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.build_adversariallm_behaviors import (
    EVILMATH_GENERIC_TARGET,
    WMDP_GENERIC_TARGET,
    build_evilmath,
    build_wmdp,
    main,
)

HARMBENCH_CSV_COLUMNS = [
    "Behavior",
    "FunctionalCategory",
    "SemanticCategory",
    "Tags",
    "ContextString",
    "BehaviorID",
]


def _fake_wmdp_subset(n: int) -> list[dict[str, object]]:
    return [
        {
            "question": f"q-{i}",
            "choices": ["a", "b", "c", "d"],
            "answer": i % 4,
        }
        for i in range(n)
    ]


def _fake_evilmath_split(n: int) -> list[dict[str, object]]:
    return [
        {
            "evil_question": f"eq-{i}",
            "original_answer": f"answer #### {i}",
            "original_question": f"oq-{i}",
            "index": i,
        }
        for i in range(n)
    ]


class TestBuildWmdp:
    @patch("scripts.build_adversariallm_behaviors.datasets")
    def test_writes_csv_and_targets_paths(
        self, mock_datasets: MagicMock, tmp_path: Path
    ) -> None:
        mock_datasets.load_dataset.side_effect = lambda *_, **__: _fake_wmdp_subset(5)
        csv_path = tmp_path / "wmdp_behaviors.csv"
        targets_path = tmp_path / "wmdp_targets_text.json"

        n_rows = build_wmdp(
            csv_path=csv_path, targets_path=targets_path
        )

        assert csv_path.exists()
        assert targets_path.exists()
        # 3 WMDP subsets * 5 rows each
        assert n_rows == 15

    @patch("scripts.build_adversariallm_behaviors.datasets")
    def test_csv_has_expected_columns(
        self, mock_datasets: MagicMock, tmp_path: Path
    ) -> None:
        mock_datasets.load_dataset.side_effect = lambda *_, **__: _fake_wmdp_subset(2)
        csv_path = tmp_path / "wmdp_behaviors.csv"
        targets_path = tmp_path / "wmdp_targets_text.json"
        build_wmdp(csv_path=csv_path, targets_path=targets_path)

        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == HARMBENCH_CSV_COLUMNS

    @patch("scripts.build_adversariallm_behaviors.datasets")
    def test_targets_use_generic_string(
        self, mock_datasets: MagicMock, tmp_path: Path
    ) -> None:
        mock_datasets.load_dataset.side_effect = lambda *_, **__: _fake_wmdp_subset(3)
        csv_path = tmp_path / "wmdp_behaviors.csv"
        targets_path = tmp_path / "wmdp_targets_text.json"
        build_wmdp(csv_path=csv_path, targets_path=targets_path)

        with open(targets_path) as f:
            targets = json.load(f)
        assert isinstance(targets, dict)
        assert len(targets) == 9
        for value in targets.values():
            assert value == WMDP_GENERIC_TARGET

    @patch("scripts.build_adversariallm_behaviors.datasets")
    def test_targets_keys_match_csv_behavior_ids(
        self, mock_datasets: MagicMock, tmp_path: Path
    ) -> None:
        mock_datasets.load_dataset.side_effect = lambda *_, **__: _fake_wmdp_subset(2)
        csv_path = tmp_path / "wmdp_behaviors.csv"
        targets_path = tmp_path / "wmdp_targets_text.json"
        build_wmdp(csv_path=csv_path, targets_path=targets_path)

        with open(csv_path) as f:
            ids = {row["BehaviorID"] for row in csv.DictReader(f)}
        with open(targets_path) as f:
            target_ids = set(json.load(f).keys())
        assert ids == target_ids


class TestBuildEvilmath:
    @patch("scripts.build_adversariallm_behaviors.datasets")
    def test_writes_csv_and_targets_paths(
        self, mock_datasets: MagicMock, tmp_path: Path
    ) -> None:
        mock_datasets.load_dataset.return_value = _fake_evilmath_split(11)
        csv_path = tmp_path / "evilmath_behaviors.csv"
        targets_path = tmp_path / "evilmath_targets_text.json"
        n_rows = build_evilmath(
            csv_path=csv_path, targets_path=targets_path
        )

        assert csv_path.exists()
        assert targets_path.exists()
        assert n_rows == 11

    @patch("scripts.build_adversariallm_behaviors.datasets")
    def test_csv_has_expected_columns(
        self, mock_datasets: MagicMock, tmp_path: Path
    ) -> None:
        mock_datasets.load_dataset.return_value = _fake_evilmath_split(4)
        csv_path = tmp_path / "evilmath_behaviors.csv"
        targets_path = tmp_path / "evilmath_targets_text.json"
        build_evilmath(csv_path=csv_path, targets_path=targets_path)

        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == HARMBENCH_CSV_COLUMNS

    @patch("scripts.build_adversariallm_behaviors.datasets")
    def test_targets_use_generic_string(
        self, mock_datasets: MagicMock, tmp_path: Path
    ) -> None:
        mock_datasets.load_dataset.return_value = _fake_evilmath_split(3)
        csv_path = tmp_path / "evilmath_behaviors.csv"
        targets_path = tmp_path / "evilmath_targets_text.json"
        build_evilmath(csv_path=csv_path, targets_path=targets_path)

        with open(targets_path) as f:
            targets = json.load(f)
        assert isinstance(targets, dict)
        assert len(targets) == 3
        for value in targets.values():
            assert value == EVILMATH_GENERIC_TARGET

    @patch("scripts.build_adversariallm_behaviors.datasets")
    def test_targets_keys_match_csv_behavior_ids(
        self, mock_datasets: MagicMock, tmp_path: Path
    ) -> None:
        mock_datasets.load_dataset.return_value = _fake_evilmath_split(5)
        csv_path = tmp_path / "evilmath_behaviors.csv"
        targets_path = tmp_path / "evilmath_targets_text.json"
        build_evilmath(csv_path=csv_path, targets_path=targets_path)

        with open(csv_path) as f:
            ids = {row["BehaviorID"] for row in csv.DictReader(f)}
        with open(targets_path) as f:
            target_ids = set(json.load(f).keys())
        assert ids == target_ids


class TestNoDataInspection:
    @patch("scripts.build_adversariallm_behaviors.datasets")
    def test_wmdp_builder_does_not_print_behavior_text(
        self,
        mock_datasets: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_datasets.load_dataset.side_effect = lambda *_, **__: _fake_wmdp_subset(2)
        build_wmdp(
            csv_path=tmp_path / "wmdp_behaviors.csv",
            targets_path=tmp_path / "wmdp_targets_text.json",
        )
        captured = capsys.readouterr()
        assert "q-0" not in captured.out
        assert "q-0" not in captured.err
        assert WMDP_GENERIC_TARGET not in captured.out

    @patch("scripts.build_adversariallm_behaviors.datasets")
    def test_evilmath_builder_does_not_print_behavior_text(
        self,
        mock_datasets: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_datasets.load_dataset.return_value = _fake_evilmath_split(3)
        build_evilmath(
            csv_path=tmp_path / "evilmath_behaviors.csv",
            targets_path=tmp_path / "evilmath_targets_text.json",
        )
        captured = capsys.readouterr()
        assert "eq-0" not in captured.out
        assert "eq-0" not in captured.err
        assert EVILMATH_GENERIC_TARGET not in captured.out


class TestMain:
    @patch("scripts.build_adversariallm_behaviors.datasets")
    def test_main_writes_all_four_files(
        self, mock_datasets: MagicMock, tmp_path: Path
    ) -> None:
        def fake_load(repo: str, *args: object, **kwargs: object) -> object:
            if "evilmath" in repo.lower():
                return _fake_evilmath_split(4)
            return _fake_wmdp_subset(2)

        mock_datasets.load_dataset.side_effect = fake_load
        main([
            "--behaviors-csv-dir",
            str(tmp_path / "behaviors"),
        ])
        assert (tmp_path / "behaviors" / "wmdp_behaviors.csv").exists()
        assert (tmp_path / "behaviors" / "wmdp_targets_text.json").exists()
        assert (tmp_path / "behaviors" / "evilmath_behaviors.csv").exists()
        assert (tmp_path / "behaviors" / "evilmath_targets_text.json").exists()

    @patch("scripts.build_attack_queue.write_strongreject_targets", return_value=7)
    @patch("scripts.build_attack_queue.write_strongreject_csv", return_value=7)
    @patch("scripts.build_adversariallm_behaviors.datasets")
    def test_main_with_strongreject_invokes_writers(
        self,
        mock_datasets: MagicMock,
        mock_csv: MagicMock,
        mock_targets: MagicMock,
        tmp_path: Path,
    ) -> None:
        def fake_load(repo: str, *args: object, **kwargs: object) -> object:
            if "evilmath" in repo.lower():
                return _fake_evilmath_split(2)
            return _fake_wmdp_subset(1)

        mock_datasets.load_dataset.side_effect = fake_load
        out_dir = tmp_path / "behaviors"
        main([
            "--behaviors-csv-dir",
            str(out_dir),
            "--with-strongreject",
        ])
        mock_csv.assert_called_once_with(out_dir / "strongreject_behaviors.csv")
        mock_targets.assert_called_once_with(
            out_dir / "strongreject_behaviors.csv",
            out_dir / "strongreject_targets_text.json",
        )

    @patch("scripts.build_adversariallm_behaviors.datasets")
    def test_main_without_strongreject_skips_writers(
        self, mock_datasets: MagicMock, tmp_path: Path
    ) -> None:
        def fake_load(repo: str, *args: object, **kwargs: object) -> object:
            if "evilmath" in repo.lower():
                return _fake_evilmath_split(2)
            return _fake_wmdp_subset(1)

        mock_datasets.load_dataset.side_effect = fake_load
        main([
            "--behaviors-csv-dir",
            str(tmp_path / "behaviors"),
        ])
        assert not (tmp_path / "behaviors" / "strongreject_behaviors.csv").exists()
