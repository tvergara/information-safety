from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.build_attack_queue import (
    _count_csv_rows,
    build_attack_queue,
    compute_shards,
    write_strongreject_csv,
    write_strongreject_targets,
)


class TestCountCsvRows:
    def test_counts_logical_rows_when_fields_contain_newlines(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "with_newlines.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Behavior"])
            writer.writerow(["line1\nline2"])
            writer.writerow(["plain"])
            writer.writerow(["a\nb\nc"])
        assert _count_csv_rows(path) == 3

    def test_counts_zero_for_header_only_file(self, tmp_path: Path) -> None:
        path = tmp_path / "header_only.csv"
        path.write_text("Behavior\n")
        assert _count_csv_rows(path) == 0


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

    @patch("scripts.build_attack_queue.datasets")
    def test_semantic_category_is_harmbench_compatible(
        self, mock_datasets: MagicMock, tmp_path: Path
    ) -> None:
        mock_datasets.load_dataset.return_value = [
            {"prompt": f"p{i}", "category": f"sr-cat-{i}"} for i in range(4)
        ]
        out = tmp_path / "sr.csv"
        write_strongreject_csv(out)

        with open(out) as f:
            rows = list(csv.DictReader(f))
        harmbench_categories = {
            "chemical_biological", "illegal", "misinformation_disinformation",
            "harmful", "harassment_bullying", "cybercrime_intrusion",
        }
        assert len(rows) == 4
        for row in rows:
            assert row["SemanticCategory"] in harmbench_categories


class TestBuildAttackQueue:
    def _setup_csvs(self, tmp_path: Path, *, rows: int = 16) -> Path:
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        body = "header\n" + "".join(f"r{i}\n" for i in range(rows))
        (csv_dir / "harmbench_behaviors_standard.csv").write_text(body)
        (csv_dir / "strongreject_behaviors.csv").write_text(body)
        (csv_dir / "strongreject_targets_text.json").write_text("{}")
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

    def test_strongreject_command_uses_dataset_paths(self, tmp_path: Path) -> None:
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
        cmd = json.loads(files[0].read_text())["command"]
        output_arg = cmd[cmd.index("--output-jsonl") + 1]
        assert "strongreject" in output_arg
        assert "M1" in output_arg
        targets_arg = cmd[cmd.index("--targets-json") + 1]
        assert targets_arg == str(csv_dir / "strongreject_targets_text.json")

    def test_harmbench_command_uses_packaged_targets(self, tmp_path: Path) -> None:
        queue_root = tmp_path / "queue"
        csv_dir = self._setup_csvs(tmp_path)
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
        files = list((queue_root / "pending").glob("*.json"))
        assert len(files) == 1
        cmd = json.loads(files[0].read_text())["command"]
        targets_arg = cmd[cmd.index("--targets-json") + 1]
        assert targets_arg.endswith("harmbench_targets_text.json")
        assert "third_party" in targets_arg
        assert Path(targets_arg).exists()

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


class TestMissingInputFiles:
    def test_raises_if_harmbench_csv_missing(self, tmp_path: Path) -> None:
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="harmbench"):
            build_attack_queue(
                attacks=["gcg"],
                models=["m1"],
                dataset_names=["harmbench"],
                num_shards=1,
                queue_root=tmp_path / "queue",
                behaviors_csv_dir=csv_dir,
                attacks_dir=tmp_path / "attacks",
                adversariallm_venv=tmp_path / "adv-venv",
                repo_root=tmp_path / "repo",
            )

    def test_raises_if_strongreject_targets_json_missing(
        self, tmp_path: Path
    ) -> None:
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        body = "header\n" + "".join(f"r{i}\n" for i in range(4))
        (csv_dir / "strongreject_behaviors.csv").write_text(body)
        with pytest.raises(FileNotFoundError, match="strongreject"):
            build_attack_queue(
                attacks=["gcg"],
                models=["m1"],
                dataset_names=["strongreject"],
                num_shards=1,
                queue_root=tmp_path / "queue",
                behaviors_csv_dir=csv_dir,
                attacks_dir=tmp_path / "attacks",
                adversariallm_venv=tmp_path / "adv-venv",
                repo_root=tmp_path / "repo",
            )


class TestWmdpEvilmathDatasets:
    def _setup_csvs(self, tmp_path: Path, *, rows: int = 16) -> Path:
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        body = "header\n" + "".join(f"r{i}\n" for i in range(rows))
        (csv_dir / "wmdp_behaviors.csv").write_text(body)
        (csv_dir / "wmdp_targets_text.json").write_text("{}")
        (csv_dir / "evilmath_behaviors.csv").write_text(body)
        (csv_dir / "evilmath_targets_text.json").write_text("{}")
        return csv_dir

    def test_wmdp_command_uses_wmdp_paths(self, tmp_path: Path) -> None:
        queue_root = tmp_path / "queue"
        csv_dir = self._setup_csvs(tmp_path)
        build_attack_queue(
            attacks=["gcg"],
            models=["sft-wmdp-Llama-3.1-8B-Instruct-ec55867d84a0"],
            dataset_names=["wmdp"],
            num_shards=1,
            queue_root=queue_root,
            behaviors_csv_dir=csv_dir,
            attacks_dir=tmp_path / "attacks",
            adversariallm_venv=tmp_path / "adv-venv",
            repo_root=tmp_path / "repo",
        )
        files = list((queue_root / "pending").glob("*.json"))
        assert len(files) == 1
        cmd = json.loads(files[0].read_text())["command"]
        targets_arg = cmd[cmd.index("--targets-json") + 1]
        assert targets_arg == str(csv_dir / "wmdp_targets_text.json")
        behaviors_arg = cmd[cmd.index("--behaviors-csv") + 1]
        assert behaviors_arg == str(csv_dir / "wmdp_behaviors.csv")

    def test_evilmath_command_uses_evilmath_paths(self, tmp_path: Path) -> None:
        queue_root = tmp_path / "queue"
        csv_dir = self._setup_csvs(tmp_path)
        build_attack_queue(
            attacks=["gcg"],
            models=["sft-evilmath-Llama-3.1-8B-Instruct-d650794f965d"],
            dataset_names=["evilmath"],
            num_shards=1,
            queue_root=queue_root,
            behaviors_csv_dir=csv_dir,
            attacks_dir=tmp_path / "attacks",
            adversariallm_venv=tmp_path / "adv-venv",
            repo_root=tmp_path / "repo",
        )
        files = list((queue_root / "pending").glob("*.json"))
        assert len(files) == 1
        cmd = json.loads(files[0].read_text())["command"]
        targets_arg = cmd[cmd.index("--targets-json") + 1]
        assert targets_arg == str(csv_dir / "evilmath_targets_text.json")
        behaviors_arg = cmd[cmd.index("--behaviors-csv") + 1]
        assert behaviors_arg == str(csv_dir / "evilmath_behaviors.csv")

    def test_defense_id_model_passes_through_to_run_attack(
        self, tmp_path: Path
    ) -> None:
        """defense_id has no `/`, so the slug equals the defense_id."""
        queue_root = tmp_path / "queue"
        csv_dir = self._setup_csvs(tmp_path)
        defense = "sft-wmdp-Llama-3.1-8B-Instruct-ec55867d84a0"
        build_attack_queue(
            attacks=["gcg"],
            models=[defense],
            dataset_names=["wmdp"],
            num_shards=1,
            queue_root=queue_root,
            behaviors_csv_dir=csv_dir,
            attacks_dir=tmp_path / "attacks",
            adversariallm_venv=tmp_path / "adv-venv",
            repo_root=tmp_path / "repo",
        )
        files = list((queue_root / "pending").glob("*.json"))
        cmd = json.loads(files[0].read_text())["command"]
        model_arg = cmd[cmd.index("--model") + 1]
        assert model_arg == defense
        output_arg = cmd[cmd.index("--output-jsonl") + 1]
        assert defense in output_arg
        assert "wmdp" in output_arg

    def test_raises_if_wmdp_csv_missing(self, tmp_path: Path) -> None:
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="wmdp"):
            build_attack_queue(
                attacks=["gcg"],
                models=["m1"],
                dataset_names=["wmdp"],
                num_shards=1,
                queue_root=tmp_path / "queue",
                behaviors_csv_dir=csv_dir,
                attacks_dir=tmp_path / "attacks",
                adversariallm_venv=tmp_path / "adv-venv",
                repo_root=tmp_path / "repo",
            )

    def test_raises_if_evilmath_targets_missing(self, tmp_path: Path) -> None:
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        body = "header\n" + "".join(f"r{i}\n" for i in range(4))
        (csv_dir / "evilmath_behaviors.csv").write_text(body)
        with pytest.raises(FileNotFoundError, match="evilmath"):
            build_attack_queue(
                attacks=["gcg"],
                models=["m1"],
                dataset_names=["evilmath"],
                num_shards=1,
                queue_root=tmp_path / "queue",
                behaviors_csv_dir=csv_dir,
                attacks_dir=tmp_path / "attacks",
                adversariallm_venv=tmp_path / "adv-venv",
                repo_root=tmp_path / "repo",
            )


class TestCliDefaults:
    def test_num_shards_default_is_16(self) -> None:
        from scripts.build_attack_queue import main as _main

        captured: dict[str, object] = {}

        def fake_build(**kwargs: object) -> list[Path]:
            captured["num_shards"] = kwargs["num_shards"]
            return []

        with patch(
            "scripts.build_attack_queue.build_attack_queue", side_effect=fake_build
        ):
            _main([
                "--queue-root", "/q",
                "--behaviors-csv-dir", "/b",
                "--attacks-dir", "/a",
                "--adversariallm-venv", "/v",
                "--repo-root", "/r",
            ])
        assert captured["num_shards"] == 16


class TestWriteStrongrejectTargets:
    def test_keys_match_csv_and_values_have_prefix(
        self, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "sr.csv"
        json_path = tmp_path / "sr_targets.json"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Behavior", "FunctionalCategory", "SemanticCategory",
                "Tags", "ContextString", "BehaviorID",
            ])
            for i in range(7):
                writer.writerow([
                    f"opaque-behavior-{i}", "standard", "harmful",
                    "", "", f"strongreject_{i:04d}",
                ])

        n = write_strongreject_targets(csv_path, json_path)

        assert n == 7
        with open(json_path) as f:
            targets = json.load(f)
        assert isinstance(targets, dict)
        assert len(targets) == 7
        expected_ids = {f"strongreject_{i:04d}" for i in range(7)}
        assert set(targets.keys()) == expected_ids
        for value in targets.values():
            assert isinstance(value, str)
            assert value.startswith("Sure, here is ")
            assert value == value.strip()

    def test_lowercases_and_strips_behavior(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sr.csv"
        json_path = tmp_path / "sr_targets.json"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Behavior", "FunctionalCategory", "SemanticCategory",
                "Tags", "ContextString", "BehaviorID",
            ])
            writer.writerow([
                "  MIXED-CaSe-PROMPT  ", "standard", "harmful",
                "", "", "strongreject_0000",
            ])

        write_strongreject_targets(csv_path, json_path)

        with open(json_path) as f:
            targets = json.load(f)
        assert targets["strongreject_0000"] == "Sure, here is mixed-case-prompt"
