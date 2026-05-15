"""Tests for the trc attack-shard submission producer."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.submit_trc_attack_jobs import (
    DATASET_TOTALS_DEFAULT,
    TRC_ADVERSARIALLM_VENV,
    TRC_BASE_DIR,
    TRC_EAI_IMAGE,
    TRC_REPO_DIR,
    build_eai_submit_argv,
    deterministic_job_name,
    iter_pending_attack_shards,
    load_submitted_state,
    main,
    record_submission,
)


def _spec(
    attack: str = "gcg",
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    dataset: str = "harmbench",
    shard: int = 0,
    start: int = 0,
    end: int = 12,
) -> dict[str, object]:
    return {
        "attack": attack,
        "model": model,
        "dataset": dataset,
        "shard": shard,
        "start": start,
        "end": end,
    }


@pytest.fixture
def _state_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "trc-attacks-submitted.jsonl"
    monkeypatch.setattr(
        "scripts.submit_trc_attack_jobs.DEFAULT_STATE_FILE", path
    )
    return path


def test_argv_targets_trc_via_ssh_with_gpu() -> None:
    argv = build_eai_submit_argv(spec=_spec())
    assert argv[0] == "ssh"
    assert argv[1] == "trc"
    joined = " ".join(argv)
    assert "eai job submit" in joined
    assert "--gpu 1" in joined
    assert "--preemptable" in joined
    assert "--non-preemptable" not in joined
    assert "--enforce-name" in joined
    assert TRC_EAI_IMAGE in joined
    assert "snow.research.mmteb.safety:/work:rw" in joined


def test_build_eai_submit_argv_emits_non_preemptable_when_requested() -> None:
    argv = build_eai_submit_argv(spec=_spec(), preemptable=False)
    joined = " ".join(argv)
    assert "--non-preemptable" in joined
    assert " --preemptable" not in joined


def test_main_forwards_non_preemptable_flag(
    _state_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("scripts.submit_trc_attack_jobs.subprocess.run"):
        main(
            [
                "--dry-run",
                "--non-preemptable",
                "--attacks", "gcg",
                "--models", "meta-llama/Meta-Llama-3-8B-Instruct",
                "--datasets", "harmbench",
                "--num-shards", "2",
            ]
        )
    captured = capsys.readouterr()
    assert "--non-preemptable" in captured.out
    assert " --preemptable" not in captured.out


def test_container_invokes_run_one_attack_with_work_paths() -> None:
    argv = build_eai_submit_argv(
        spec=_spec(attack="gcg", model="Qwen/Qwen3-4B", shard=3, start=36, end=48)
    )
    joined = " ".join(argv)
    assert "bash scripts/run_one_attack.sh" in joined
    assert "--attack gcg" in joined
    assert "--model Qwen/Qwen3-4B" in joined
    assert "--shard-start 36" in joined
    assert "--shard-end 48" in joined
    assert f"--adversariallm-venv {TRC_ADVERSARIALLM_VENV}" in joined
    assert f"--repo-root {TRC_REPO_DIR}" in joined


def test_container_uses_harmbench_targets_from_submodule() -> None:
    argv = build_eai_submit_argv(spec=_spec(dataset="harmbench"))
    joined = " ".join(argv)
    assert (
        f"--behaviors-csv {TRC_BASE_DIR}/attacks/harmbench_behaviors_standard.csv"
        in joined
    )
    assert (
        f"--targets-json {TRC_REPO_DIR}/third_party/adversariallm/data/"
        "optimizer_targets/harmbench_targets_text.json"
    ) in joined


def test_container_uses_strongreject_targets_on_work() -> None:
    argv = build_eai_submit_argv(spec=_spec(dataset="strongreject"))
    joined = " ".join(argv)
    assert (
        f"--behaviors-csv {TRC_BASE_DIR}/attacks/strongreject_behaviors.csv"
        in joined
    )
    assert (
        f"--targets-json {TRC_BASE_DIR}/attacks/strongreject_targets_text.json"
        in joined
    )


def test_container_uses_evilmath_targets_on_work() -> None:
    argv = build_eai_submit_argv(spec=_spec(dataset="evilmath"))
    joined = " ".join(argv)
    assert (
        f"--behaviors-csv {TRC_BASE_DIR}/attacks/evilmath_behaviors.csv"
        in joined
    )
    assert (
        f"--targets-json {TRC_BASE_DIR}/attacks/evilmath_targets_text.json"
        in joined
    )


def test_default_dataset_totals_includes_evilmath() -> None:
    assert DATASET_TOTALS_DEFAULT["evilmath"] == 298


def test_container_output_paths_match_build_attack_queue_layout() -> None:
    argv = build_eai_submit_argv(
        spec=_spec(
            attack="autodan",
            model="allenai/Olmo-3-7B-Instruct",
            dataset="harmbench",
            shard=5,
        )
    )
    joined = " ".join(argv)
    assert (
        f"--output-jsonl {TRC_BASE_DIR}/attacks/"
        "autodan-allenai_Olmo-3-7B-Instruct-harmbench-shard5.jsonl"
    ) in joined
    assert (
        f"--adv-save-dir {TRC_BASE_DIR}/adversariallm-outputs/"
        "autodan-allenai_Olmo-3-7B-Instruct-harmbench-shard5"
    ) in joined


def test_container_exports_hf_home_and_token() -> None:
    argv = build_eai_submit_argv(spec=_spec())
    joined = " ".join(argv)
    assert "export HF_HOME=/work/.hf-cache" in joined
    assert "export HUGGING_FACE_HUB_TOKEN=$(cat /work/.hf-cache/token)" in joined


def test_container_overrides_inherited_transformers_cache_to_work() -> None:
    argv = build_eai_submit_argv(spec=_spec())
    joined = " ".join(argv)
    assert "unset TRANSFORMERS_CACHE" in joined
    assert "export HF_HUB_CACHE=/work/.hf-cache/hub" in joined


def test_remote_command_string_is_shlex_parseable() -> None:
    argv = build_eai_submit_argv(spec=_spec())
    tokens = shlex.split(argv[2])
    assert tokens[:3] == ["eai", "job", "submit"]
    assert "bash" in tokens
    assert "-lc" in tokens


def test_deterministic_job_name_is_stable_and_namespaced() -> None:
    spec = _spec()
    a = deterministic_job_name(spec)
    b = deterministic_job_name(dict(spec))
    assert a == b
    assert a.startswith("is_attack_")
    other = deterministic_job_name(_spec(shard=1, start=12, end=24))
    assert a != other


def test_iter_pending_yields_one_spec_per_shard() -> None:
    pending = iter_pending_attack_shards(
        attacks=["gcg"],
        models=["meta-llama/Meta-Llama-3-8B-Instruct"],
        datasets=["harmbench"],
        dataset_totals={"harmbench": 200},
        num_shards=4,
        merged_attacks=set(),
        submitted_state=set(),
    )
    assert len(pending) == 4
    starts_ends = [(p["start"], p["end"]) for p in pending]
    assert starts_ends == [(0, 50), (50, 100), (100, 150), (150, 200)]
    for p in pending:
        assert p["attack"] == "gcg"
        assert p["model"] == "meta-llama/Meta-Llama-3-8B-Instruct"
        assert p["dataset"] == "harmbench"


def test_iter_pending_skips_merged_attacks() -> None:
    pending = iter_pending_attack_shards(
        attacks=["gcg", "autodan"],
        models=["meta-llama/Meta-Llama-3-8B-Instruct"],
        datasets=["harmbench"],
        dataset_totals={"harmbench": 200},
        num_shards=4,
        merged_attacks={
            ("gcg", "meta-llama_Meta-Llama-3-8B-Instruct", "harmbench"),
        },
        submitted_state=set(),
    )
    assert all(p["attack"] != "gcg" for p in pending)
    assert any(p["attack"] == "autodan" for p in pending)


def test_iter_pending_skips_already_submitted() -> None:
    spec = _spec()
    name = deterministic_job_name(spec)
    pending = iter_pending_attack_shards(
        attacks=["gcg"],
        models=["meta-llama/Meta-Llama-3-8B-Instruct"],
        datasets=["harmbench"],
        dataset_totals={"harmbench": 200},
        num_shards=4,
        merged_attacks=set(),
        submitted_state={name},
    )
    assert len(pending) == 3
    starts = [p["start"] for p in pending]
    assert spec["start"] not in starts


def test_record_and_load_state_round_trip(_state_file: Path) -> None:
    spec = _spec()
    job_id = deterministic_job_name(spec)
    record_submission(
        state_file=_state_file, spec=spec, job_id=job_id, eai_uuid="uuid-1"
    )
    loaded = load_submitted_state(_state_file)
    assert job_id in loaded


def test_state_records_eai_uuid_and_spec(_state_file: Path) -> None:
    spec = _spec()
    record_submission(
        state_file=_state_file,
        spec=spec,
        job_id=deterministic_job_name(spec),
        eai_uuid="abc",
    )
    rows = [
        json.loads(line)
        for line in _state_file.read_text().splitlines()
        if line.strip()
    ]
    assert rows[0]["eai_uuid"] == "abc"
    assert rows[0]["spec"] == spec


def test_main_dry_run_emits_commands_and_no_state(
    _state_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("scripts.submit_trc_attack_jobs.subprocess.run") as run:
        main(
            [
                "--dry-run",
                "--attacks", "gcg",
                "--models", "meta-llama/Meta-Llama-3-8B-Instruct",
                "--datasets", "harmbench",
                "--num-shards", "2",
            ]
        )
        run.assert_not_called()
    captured = capsys.readouterr()
    assert "ssh trc" in captured.out
    assert "eai job submit" in captured.out
    assert not _state_file.exists()


def test_main_submits_and_records_state(_state_file: Path) -> None:
    submit_stdout = (
        "9db35489-d4c2-4753-ba16-bbd815424302 QUEUING is_attack_xyz"
        " snow.research.mmteb snow.siva_reddy 2026-05-15T04:12:20Z 0"
        " [bash -lc ...] -\n"
    )
    mock_run = MagicMock(
        return_value=MagicMock(returncode=0, stdout=submit_stdout, stderr="")
    )
    with patch("scripts.submit_trc_attack_jobs.subprocess.run", mock_run):
        main(
            [
                "--attacks", "gcg",
                "--models", "meta-llama/Meta-Llama-3-8B-Instruct",
                "--datasets", "harmbench",
                "--num-shards", "2",
            ]
        )
    assert mock_run.call_count == 2
    state = load_submitted_state(_state_file)
    assert len(state) == 2


def test_main_records_clean_uuid_from_single_line_submit_output(
    _state_file: Path,
) -> None:
    submit_stdout = (
        "9db35489-d4c2-4753-ba16-bbd815424302 QUEUING is_attack_xyz"
        " snow.research.mmteb snow.siva_reddy 2026-05-15T04:12:20Z 0"
        " [bash -lc ...] -"
        " https://9db35489-d4c2-4753-ba16-bbd815424302.job.example/\n"
    )
    mock_run = MagicMock(
        return_value=MagicMock(returncode=0, stdout=submit_stdout, stderr="")
    )
    with patch("scripts.submit_trc_attack_jobs.subprocess.run", mock_run):
        main(
            [
                "--attacks", "gcg",
                "--models", "meta-llama/Meta-Llama-3-8B-Instruct",
                "--datasets", "harmbench",
                "--num-shards", "1",
            ]
        )
    rows = [
        json.loads(line)
        for line in _state_file.read_text().splitlines()
        if line.strip()
    ]
    assert rows[0]["eai_uuid"] == "9db35489-d4c2-4753-ba16-bbd815424302"


def test_main_max_submissions_caps_submissions(_state_file: Path) -> None:
    mock_run = MagicMock(
        return_value=MagicMock(returncode=0, stdout="aaaaaaaa-1111-2222-3333-cccccccccccc\n", stderr="")
    )
    with patch("scripts.submit_trc_attack_jobs.subprocess.run", mock_run):
        main(
            [
                "--attacks", "gcg",
                "--models", "meta-llama/Meta-Llama-3-8B-Instruct",
                "--datasets", "harmbench",
                "--num-shards", "4",
                "--max-submissions", "1",
            ]
        )
    assert mock_run.call_count == 1
    state = load_submitted_state(_state_file)
    assert len(state) == 1


def test_main_idempotent_after_submission(_state_file: Path) -> None:
    mock_run = MagicMock(
        return_value=MagicMock(returncode=0, stdout="aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb\n", stderr="")
    )
    args = [
        "--attacks", "gcg",
        "--models", "meta-llama/Meta-Llama-3-8B-Instruct",
        "--datasets", "harmbench",
        "--num-shards", "2",
    ]
    with patch("scripts.submit_trc_attack_jobs.subprocess.run", mock_run):
        main(args)
        first = mock_run.call_count
        main(args)
        assert mock_run.call_count == first
