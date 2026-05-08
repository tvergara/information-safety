from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.build_attack_queue import (
    build_attack_queue,
    compute_shards,
    write_strongreject_csv,
)


class TestComputeShards:
    def test_evenly_divides_harmbench(self) -> None:
        shards = compute_shards(total=200, num_shards=8)
        assert shards == [
            (0, 25), (25, 50), (50, 75), (75, 100),
            (100, 125), (125, 150), (150, 175), (175, 200),
        ]

    def test_unevenly_divides_strongreject(self) -> None:
        shards = compute_shards(total=313, num_shards=8)
        assert len(shards) == 8
        assert shards[0][0] == 0
        assert shards[-1][1] == 313
        sizes = [end - start for start, end in shards]
        assert sum(sizes) == 313
        assert max(sizes) - min(sizes) <= 1
        for i in range(len(shards) - 1):
            assert shards[i][1] == shards[i + 1][0]

    def test_drops_empty_shards_when_total_smaller_than_num_shards(self) -> None:
        assert compute_shards(total=3, num_shards=8) == [(0, 1), (1, 2), (2, 3)]
        assert compute_shards(total=0, num_shards=8) == []


class TestWriteStrongrejectCsv:
    @patch("scripts.build_attack_queue.datasets")
    def test_writes_one_row_per_prompt(
        self, mock_datasets: MagicMock, tmp_path: Path
    ) -> None:
        mock_datasets.load_dataset.return_value = [
            {"prompt": f"p{i}", "category": "cat"} for i in range(5)
        ]
        out = tmp_path / "sr.csv"
        n = write_strongreject_csv(out)
        assert n == 5
        assert out.exists()
        with open(out) as f:
            lines = f.read().splitlines()
        assert len(lines) == 6
        assert lines[0].split(",") == [
            "Behavior", "FunctionalCategory", "SemanticCategory",
            "Tags", "ContextString", "BehaviorID",
        ]


class TestBuildAttackQueue:
    def _setup_csvs(self, tmp_path: Path, *, rows: int = 16) -> Path:
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        body = "header\n" + "".join(f"r{i}\n" for i in range(rows))
        (csv_dir / "harmbench_behaviors_standard.csv").write_text(body)
        (csv_dir / "strongreject_behaviors.csv").write_text(body)
        return csv_dir

    def test_writes_one_job_per_combination(self, tmp_path: Path) -> None:
        queue_root = tmp_path / "queue"
        csv_dir = self._setup_csvs(tmp_path)

        written = build_attack_queue(
            attacks=["gcg"],
            models=["meta-llama/Meta-Llama-3-8B-Instruct"],
            dataset_names=["harmbench", "strongreject"],
            num_shards=8,
            queue_root=queue_root,
            behaviors_csv_dir=csv_dir,
            attacks_dir=tmp_path / "attacks",
            adversariallm_venv=tmp_path / "adv-venv",
            repo_root=tmp_path / "repo",
        )
        assert len(written) == 1 * 1 * 2 * 8

    def test_skips_existing_queue_entries(self, tmp_path: Path) -> None:
        queue_root = tmp_path / "queue"
        csv_dir = self._setup_csvs(tmp_path)

        def run() -> list[Path]:
            return build_attack_queue(
                attacks=["gcg"],
                models=["m1"],
                dataset_names=["harmbench"],
                num_shards=2,
                queue_root=queue_root,
                behaviors_csv_dir=csv_dir,
                attacks_dir=tmp_path / "attacks",
                adversariallm_venv=tmp_path / "adv-venv",
                repo_root=tmp_path / "repo",
            )

        assert len(run()) == 2
        assert len(run()) == 0

    def test_strongreject_paths_use_dataset_in_filename(self, tmp_path: Path) -> None:
        queue_root = tmp_path / "queue"
        csv_dir = self._setup_csvs(tmp_path)
        build_attack_queue(
            attacks=["gcg"],
            models=["meta-llama/M1"],
            dataset_names=["strongreject"],
            num_shards=1,
            queue_root=queue_root,
            behaviors_csv_dir=csv_dir,
            attacks_dir=tmp_path / "attacks",
            adversariallm_venv=tmp_path / "adv-venv",
            repo_root=tmp_path / "repo",
        )
        files = list((queue_root / "pending").glob("*.json"))
        assert len(files) == 1
        payload = json.loads(files[0].read_text())
        cmd = payload["command"]
        output_arg = cmd[cmd.index("--output-jsonl") + 1]
        assert "strongreject" in output_arg
        assert "M1" in output_arg

    def test_skips_if_merged_jsonl_exists(self, tmp_path: Path) -> None:
        queue_root = tmp_path / "queue"
        csv_dir = self._setup_csvs(tmp_path)
        attacks_dir = tmp_path / "attacks"
        attacks_dir.mkdir()
        (attacks_dir / "gcg-meta-llama_M1-harmbench-merged.jsonl").write_text("")
        written = build_attack_queue(
            attacks=["gcg"],
            models=["meta-llama/M1"],
            dataset_names=["harmbench"],
            num_shards=2,
            queue_root=queue_root,
            behaviors_csv_dir=csv_dir,
            attacks_dir=attacks_dir,
            adversariallm_venv=tmp_path / "adv-venv",
            repo_root=tmp_path / "repo",
        )
        assert len(written) == 0


class TestMissingCsvFile:
    def test_raises_if_harmbench_csv_missing(self, tmp_path: Path) -> None:
        queue_root = tmp_path / "queue"
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="harmbench"):
            build_attack_queue(
                attacks=["gcg"],
                models=["m1"],
                dataset_names=["harmbench"],
                num_shards=1,
                queue_root=queue_root,
                behaviors_csv_dir=csv_dir,
                attacks_dir=tmp_path / "attacks",
                adversariallm_venv=tmp_path / "adv-venv",
                repo_root=tmp_path / "repo",
            )
