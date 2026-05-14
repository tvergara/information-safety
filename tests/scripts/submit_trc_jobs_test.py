"""Tests for the trc eai-based job submission producer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.submit_trc_jobs import (
    TRC_BASE_DIR,
    TRC_EAI_IMAGE,
    TRC_HF_HOME,
    build_eai_submit_argv,
    deterministic_job_name,
    iter_pending_configs,
    load_submitted_state,
    main,
    record_submission,
)


@pytest.fixture(autouse=True)
def _scratch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRATCH", "/scratch/t/tvergara")


@pytest.fixture
def _state_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "trc-submitted.jsonl"
    monkeypatch.setattr("scripts.submit_trc_jobs.DEFAULT_STATE_FILE", path)
    return path


@pytest.fixture
def _local_attacks_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "local_attacks"
    path.mkdir()
    monkeypatch.setattr("scripts.submit_trc_jobs.default_attacks_dir", lambda: path)
    return path


def _baseline_config() -> dict[str, object]:
    return {
        "experiment_name": "BaselineStrategy",
        "dataset_name": "advbench_harmbench",
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "max_examples": None,
        "epoch": 0,
    }


def _data_config() -> dict[str, object]:
    return {
        "experiment_name": "DataStrategy",
        "dataset_name": "advbench_harmbench",
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "max_examples": 32,
        "max_epochs": 32,
        "epoch": 31,
    }


def _wmdp_defense_config() -> dict[str, object]:
    return {
        "experiment_name": "BaselineStrategy",
        "dataset_name": "wmdp",
        "model_name": "sft-wmdp-Llama-3.1-8B-Instruct-ec55867d84a0",
        "max_examples": None,
        "epoch": 0,
    }


def test_dedup_via_results_file(
    tmp_path: Path,
    _state_file: Path,
) -> None:
    config = _baseline_config()
    rows = [{**config, "asr": 0.1}]
    pending = iter_pending_configs(
        configs=[config],
        rows=rows,
        state_file=_state_file,
    )
    assert pending == []


def test_dedup_via_submitted_state(
    tmp_path: Path,
    _state_file: Path,
) -> None:
    config = _baseline_config()
    record_submission(
        state_file=_state_file,
        config=config,
        job_id=deterministic_job_name(config),
        eai_uuid="some-uuid",
    )
    pending = iter_pending_configs(
        configs=[config],
        rows=[],
        state_file=_state_file,
    )
    assert pending == []


def test_dry_run_does_not_write_state(
    tmp_path: Path,
    _state_file: Path,
    _local_attacks_dir: Path,
) -> None:
    results_file = tmp_path / "final-results.jsonl"
    results_file.write_text("")
    with patch("scripts.submit_trc_jobs.subprocess.run") as run:
        main(
            [
                "--dry-run",
                "--results-file",
                str(results_file),
            ]
        )
        run.assert_not_called()
    assert not _state_file.exists()


def test_eai_command_has_required_flags() -> None:
    config = _baseline_config()
    argv = build_eai_submit_argv(config=config)
    joined = " ".join(argv)
    assert argv[0] == "ssh"
    assert argv[1] == "trc"
    assert "eai job submit" in joined
    assert "--gpu 1" in joined
    assert "--mem 64" in joined
    assert "--cpu 8" in joined
    assert "--max-run-time 43200" in joined
    assert "--non-preemptable" in joined
    assert TRC_EAI_IMAGE in joined
    assert "snow.research.mmteb.safety:/work:rw" in joined


def test_container_command_sets_env_vars() -> None:
    config = _baseline_config()
    argv = build_eai_submit_argv(config=config)
    joined = " ".join(argv)
    assert "export RESULTS_FILE=" in joined
    assert "export GENERATIONS_DIR=" in joined
    results_file = f"{TRC_BASE_DIR}/results/final-results.jsonl"
    generations_dir = f"{TRC_BASE_DIR}/generations"
    assert results_file in joined
    assert generations_dir in joined
    assert "source .venv/bin/activate" in joined
    assert "cd /work/information-safety" in joined
    assert "python" in joined and "information_safety/main.py" in joined


def test_container_command_sets_hf_home_on_work() -> None:
    config = _baseline_config()
    argv = build_eai_submit_argv(config=config)
    joined = " ".join(argv)
    assert f"export HF_HOME={TRC_HF_HOME}" in joined
    assert TRC_HF_HOME.startswith("/work/")


def test_paths_use_work_not_scratch_for_defense() -> None:
    defense_config = _wmdp_defense_config()
    argv = build_eai_submit_argv(config=defense_config)
    joined = " ".join(argv)
    assert "/scratch/" not in joined
    assert TRC_BASE_DIR in joined
    assert f"{TRC_BASE_DIR}/defenses/" in joined


def test_paths_use_work_not_scratch_for_train_data() -> None:
    config = _data_config()
    argv = build_eai_submit_argv(config=config)
    joined = " ".join(argv)
    assert "/scratch/" not in joined
    assert f"{TRC_BASE_DIR}/data/" in joined


def test_paths_use_work_not_scratch_for_precomputed_suffix(
    _local_attacks_dir: Path,
) -> None:
    canonical = _local_attacks_dir / "gcg-meta-llama_Llama-2-7b-chat-hf.jsonl"
    canonical.write_text("")
    config = {
        "experiment_name": "PrecomputedGCGStrategy",
        "dataset_name": "advbench_harmbench",
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "max_examples": None,
        "epoch": 0,
    }
    argv = build_eai_submit_argv(config=config)
    joined = " ".join(argv)
    assert "/scratch/" not in joined
    assert f"{TRC_BASE_DIR}/attacks/" in joined


def test_deterministic_job_name() -> None:
    config = _baseline_config()
    name_a = deterministic_job_name(config)
    name_b = deterministic_job_name(dict(config))
    assert name_a == name_b
    assert name_a.startswith("is_")
    assert len(name_a) == len("is_") + 16


def test_deterministic_job_name_differs_per_config() -> None:
    config_a = _baseline_config()
    config_b = _data_config()
    assert deterministic_job_name(config_a) != deterministic_job_name(config_b)


def test_job_name_matches_eai_charset() -> None:
    import re

    name = deterministic_job_name(_baseline_config())
    assert re.match(r"^[a-z0-9_]+$", name) is not None


def test_state_round_trip(
    _state_file: Path,
) -> None:
    config = _baseline_config()
    job_id = deterministic_job_name(config)
    record_submission(
        state_file=_state_file,
        config=config,
        job_id=job_id,
        eai_uuid="uuid-1",
    )
    loaded = load_submitted_state(_state_file)
    assert job_id in loaded


def test_load_submitted_state_missing_file(tmp_path: Path) -> None:
    assert load_submitted_state(tmp_path / "absent.jsonl") == set()


def test_main_dry_run_prints_commands(
    tmp_path: Path,
    _state_file: Path,
    _local_attacks_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_file = tmp_path / "final-results.jsonl"
    results_file.write_text("")
    with patch("scripts.submit_trc_jobs.subprocess.run") as run:
        main(
            [
                "--dry-run",
                "--results-file",
                str(results_file),
            ]
        )
        run.assert_not_called()
    captured = capsys.readouterr()
    assert "ssh trc" in captured.out
    assert "eai job submit" in captured.out


def test_main_idempotent_after_submission(
    tmp_path: Path,
    _state_file: Path,
    _local_attacks_dir: Path,
) -> None:
    results_file = tmp_path / "final-results.jsonl"
    results_file.write_text("")
    mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout="abcd-uuid\n", stderr=""))
    with patch("scripts.submit_trc_jobs.subprocess.run", mock_run):
        main(
            [
                "--results-file",
                str(results_file),
            ]
        )
        first_calls = mock_run.call_count
        assert first_calls > 0
        main(
            [
                "--results-file",
                str(results_file),
            ]
        )
        assert mock_run.call_count == first_calls


def test_reset_flag_wipes_state(
    tmp_path: Path,
    _state_file: Path,
    _local_attacks_dir: Path,
) -> None:
    results_file = tmp_path / "final-results.jsonl"
    results_file.write_text("")
    record_submission(
        state_file=_state_file,
        config=_baseline_config(),
        job_id="is_dummy",
        eai_uuid="uuid",
    )
    assert _state_file.exists()
    with patch("scripts.submit_trc_jobs.subprocess.run") as run:
        run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        main(
            [
                "--reset",
                "--dry-run",
                "--results-file",
                str(results_file),
            ]
        )
    assert load_submitted_state(_state_file) == set()


def test_iter_pending_includes_precomputed_config_regardless_of_suffix_file(
    tmp_path: Path,
    _state_file: Path,
) -> None:
    config = {
        "experiment_name": "PrecomputedGCGStrategy",
        "dataset_name": "advbench_harmbench",
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "max_examples": None,
        "epoch": 0,
    }
    pending = iter_pending_configs(
        configs=[config],
        rows=[],
        state_file=_state_file,
    )
    assert config in pending


def test_main_warns_and_skips_when_precomputed_suffix_missing(
    tmp_path: Path,
    _state_file: Path,
    _local_attacks_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = {
        "experiment_name": "PrecomputedGCGStrategy",
        "dataset_name": "advbench_harmbench",
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "max_examples": None,
        "epoch": 0,
    }
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with patch("scripts.submit_trc_jobs.subprocess.run"):
        main(
            ["--dry-run", "--results-file", str(empty)],
            configs=[config],
        )
    captured = capsys.readouterr()
    assert "skip" in captured.out.lower() or "warning" in captured.out.lower()


def test_state_records_eai_uuid(
    tmp_path: Path,
    _state_file: Path,
) -> None:
    config = _baseline_config()
    record_submission(
        state_file=_state_file,
        config=config,
        job_id=deterministic_job_name(config),
        eai_uuid="abc-123",
    )
    rows = [json.loads(line) for line in _state_file.read_text().splitlines() if line]
    assert rows[0]["eai_uuid"] == "abc-123"
    assert rows[0]["job_id"] == deterministic_job_name(config)
    assert "submitted_at" in rows[0]
    assert rows[0]["config"] == config


def test_main_actually_submits_when_not_dry_run(
    tmp_path: Path,
    _state_file: Path,
    _local_attacks_dir: Path,
) -> None:
    results_file = tmp_path / "final-results.jsonl"
    results_file.write_text("")
    mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout="eai-uuid-xyz\n", stderr=""))
    with patch("scripts.submit_trc_jobs.subprocess.run", mock_run):
        main(
            [
                "--results-file",
                str(results_file),
            ]
        )
    assert mock_run.call_count > 0
    first_call_args = mock_run.call_args_list[0][0][0]
    assert first_call_args[:2] == ["ssh", "trc"]


def test_default_state_file_path() -> None:
    import scripts.submit_trc_jobs as st

    assert str(st.DEFAULT_STATE_FILE).endswith("information-safety/trc-submitted.jsonl")
