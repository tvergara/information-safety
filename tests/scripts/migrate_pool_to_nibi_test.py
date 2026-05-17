"""Tests for migrate_pool_to_nibi.py — Tamia pending → Nibi single-GPU sbatch."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.migrate_pool_to_nibi import (
    NIBI_GENERATIONS_DIR,
    NIBI_REPO_DIR,
    NIBI_RESULTS_FILE,
    NIBI_ROBOT,
    NIBI_RUN_SCRIPT,
    main,
)


def _write_spec(path: Path, spec_id: str, model: str) -> None:
    spec = {
        "id": spec_id,
        "command": [
            "python", "information_safety/main.py",
            "experiment=finetune-with-strategy",
            "algorithm/strategy=data",
            f"algorithm.model.pretrained_model_name_or_path={model}",
            "algorithm/dataset_handler=wmdp",
        ],
        "config": {
            "experiment_name": "DataStrategy",
            "dataset_name": "wmdp",
            "model_name": model,
        },
        "attempts": 0,
    }
    path.write_text(json.dumps(spec))


def test_main_filters_to_hf_hosted_only(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "aaa.json", "aaa", "Qwen/Qwen3-4B")
    _write_spec(local / "bbb.json", "bbb", "/scratch/t/tvergara/defenses/foo")
    _write_spec(local / "ccc.json", "ccc", "allenai/Olmo-3-7B-Instruct")
    state_file = tmp_path / "state.jsonl"

    with patch(
        "scripts.migrate_pool_to_nibi._rsync_pending_to_local", return_value=local
    ), patch("scripts.migrate_pool_to_nibi.subprocess.run") as run:
        run.return_value = MagicMock(returncode=0, stdout="Submitted batch job 9001\n", stderr="")
        main([
            "--queue-root", "run-wmdp-rerun",
            "--count", "5",
            "--state-file", str(state_file),
        ])
        submit_calls = [c for c in run.call_args_list if "sbatch" in " ".join(c.args[0])]
        joined_all = " ".join(" ".join(c.args[0]) for c in submit_calls)
        assert "/scratch/t/tvergara" not in joined_all
        assert "aaa" in joined_all
        assert "ccc" in joined_all
        assert "bbb" not in joined_all
        assert len(submit_calls) == 2


def test_main_honors_count_cap(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    for i in range(5):
        _write_spec(local / f"{i:02d}.json", f"id{i:02d}", "Qwen/Qwen3-4B")
    state_file = tmp_path / "state.jsonl"

    with patch(
        "scripts.migrate_pool_to_nibi._rsync_pending_to_local", return_value=local
    ), patch("scripts.migrate_pool_to_nibi.subprocess.run") as run:
        run.return_value = MagicMock(returncode=0, stdout="Submitted batch job 9002\n", stderr="")
        main([
            "--queue-root", "q",
            "--count", "3",
            "--state-file", str(state_file),
        ])
        submit_calls = [c for c in run.call_args_list if "sbatch" in " ".join(c.args[0])]
        assert len(submit_calls) == 3


def test_main_sbatch_uses_nibi_robot_and_run_script(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "aaa.json", "aaa", "Qwen/Qwen3-4B")
    state_file = tmp_path / "state.jsonl"

    with patch(
        "scripts.migrate_pool_to_nibi._rsync_pending_to_local", return_value=local
    ), patch("scripts.migrate_pool_to_nibi.subprocess.run") as run:
        run.return_value = MagicMock(returncode=0, stdout="Submitted batch job 9003\n", stderr="")
        main(["--queue-root", "q", "--count", "1", "--state-file", str(state_file)])
        argv = run.call_args_list[0].args[0]
        assert argv[0] == "ssh"
        assert argv[1] == NIBI_ROBOT
        assert "sbatch" in argv
        joined = " ".join(argv)
        assert NIBI_RUN_SCRIPT in joined
        assert "Qwen/Qwen3-4B" in joined
        assert "information_safety/main.py" in joined


def test_main_writes_state_file_with_slurm_job_id(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "aaa.json", "aaa", "Qwen/Qwen3-4B")
    state_file = tmp_path / "state.jsonl"

    with patch(
        "scripts.migrate_pool_to_nibi._rsync_pending_to_local", return_value=local
    ), patch("scripts.migrate_pool_to_nibi.subprocess.run") as run:
        run.return_value = MagicMock(
            returncode=0, stdout="Submitted batch job 12345\n", stderr=""
        )
        main(["--queue-root", "q", "--count", "1", "--state-file", str(state_file)])
        rows = [
            json.loads(line)
            for line in state_file.read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 1
        assert rows[0]["spec_id"] == "aaa"
        assert rows[0]["slurm_job_id"] == "12345"
        assert "submitted_at" in rows[0]


def test_main_deletes_tamia_pending_after_successful_submit(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "aaa.json", "aaa", "Qwen/Qwen3-4B")
    state_file = tmp_path / "state.jsonl"
    deleted_ids: list[str] = []

    def fake_delete(spec_id: str, queue_root: str) -> None:
        deleted_ids.append(spec_id)

    with patch(
        "scripts.migrate_pool_to_nibi._rsync_pending_to_local", return_value=local
    ), patch(
        "scripts.migrate_pool_to_nibi._delete_tamia_pending", side_effect=fake_delete
    ), patch("scripts.migrate_pool_to_nibi.subprocess.run") as run:
        run.return_value = MagicMock(
            returncode=0, stdout="Submitted batch job 9004\n", stderr=""
        )
        main(["--queue-root", "q", "--count", "5", "--state-file", str(state_file)])
        assert deleted_ids == ["aaa"]


def test_main_skips_tamia_delete_when_sbatch_fails(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "aaa.json", "aaa", "Qwen/Qwen3-4B")
    state_file = tmp_path / "state.jsonl"
    deleted_ids: list[str] = []

    def fake_delete(spec_id: str, queue_root: str) -> None:
        deleted_ids.append(spec_id)

    with patch(
        "scripts.migrate_pool_to_nibi._rsync_pending_to_local", return_value=local
    ), patch(
        "scripts.migrate_pool_to_nibi._delete_tamia_pending", side_effect=fake_delete
    ), patch(
        "scripts.migrate_pool_to_nibi.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["ssh", "robot.nibi"], "", "boom"),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            main(["--queue-root", "q", "--count", "5", "--state-file", str(state_file)])
        assert deleted_ids == []


def test_main_dry_run_does_not_submit_or_delete(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "aaa.json", "aaa", "Qwen/Qwen3-4B")
    state_file = tmp_path / "state.jsonl"
    deleted_ids: list[str] = []

    def fake_delete(spec_id: str, queue_root: str) -> None:
        deleted_ids.append(spec_id)

    with patch(
        "scripts.migrate_pool_to_nibi._rsync_pending_to_local", return_value=local
    ), patch(
        "scripts.migrate_pool_to_nibi._delete_tamia_pending", side_effect=fake_delete
    ), patch("scripts.migrate_pool_to_nibi.subprocess.run") as run:
        main([
            "--queue-root", "q",
            "--count", "5",
            "--state-file", str(state_file),
            "--dry-run",
        ])
        run.assert_not_called()
        assert deleted_ids == []
        assert not state_file.exists()


def test_main_raises_when_sbatch_output_has_no_job_id(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "aaa.json", "aaa", "Qwen/Qwen3-4B")
    state_file = tmp_path / "state.jsonl"

    with patch(
        "scripts.migrate_pool_to_nibi._rsync_pending_to_local", return_value=local
    ), patch("scripts.migrate_pool_to_nibi.subprocess.run") as run:
        run.return_value = MagicMock(returncode=0, stdout="something weird\n", stderr="")
        with pytest.raises(ValueError, match="batch job"):
            main(["--queue-root", "q", "--count", "1", "--state-file", str(state_file)])


def test_nibi_constants_use_alliance_paths() -> None:
    assert NIBI_ROBOT == "robot.nibi.alliancecan.ca"
    assert NIBI_REPO_DIR == "/home/tvergara/information-safety"
    assert NIBI_RESULTS_FILE.startswith("/scratch/tvergara/information-safety")
    assert NIBI_GENERATIONS_DIR.startswith("/scratch/tvergara/information-safety")
    assert NIBI_RUN_SCRIPT.endswith("/slurm/run-nibi-single.sh")
