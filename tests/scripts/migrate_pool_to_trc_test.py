"""Tests for migrate_pool_to_trc.py — Tamia pending → TRC eai jobs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.migrate_pool_to_trc import (
    is_hf_hosted_spec,
    main,
    pool_job_name,
    rewrite_defense_paths,
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


def test_is_hf_hosted_spec_accepts_hf_repo_ids() -> None:
    spec = {"command": ["x", "algorithm.model.pretrained_model_name_or_path=Qwen/Qwen3-4B"]}
    assert is_hf_hosted_spec(spec)


def test_is_hf_hosted_spec_rejects_local_path() -> None:
    spec = {
        "command": [
            "x",
            "algorithm.model.pretrained_model_name_or_path="
            "/scratch/t/tvergara/information-safety/defenses/tar-wmdp",
        ]
    }
    assert not is_hf_hosted_spec(spec)


def test_is_hf_hosted_spec_rejects_other_scratch_args() -> None:
    spec = {
        "command": [
            "python", "main.py",
            "algorithm.model.pretrained_model_name_or_path=Qwen/Qwen3-4B",
            "algorithm.strategy.suffix_file=/scratch/t/tvergara/information-safety/attacks/x.jsonl",
        ]
    }
    assert not is_hf_hosted_spec(spec)


def test_is_hf_hosted_spec_rejects_network_scratch_args() -> None:
    spec = {
        "command": [
            "python", "main.py",
            "algorithm.model.pretrained_model_name_or_path=Qwen/Qwen3-4B",
            "trainer.callbacks.dirpath=/network/scratch/b/brownet/foo",
        ]
    }
    assert not is_hf_hosted_spec(spec)


def test_pool_job_name_uses_spec_id() -> None:
    assert pool_job_name("004d4a94a0fd476c") == "is_pool_004d4a94a0fd476c"


def test_is_hf_hosted_spec_raises_when_model_override_missing() -> None:
    spec = {"id": "broken", "command": ["python", "main.py"]}
    with pytest.raises(ValueError, match="pretrained_model_name_or_path"):
        is_hf_hosted_spec(spec)


def test_main_filters_to_hf_hosted_only(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "aaa.json", "aaa", "Qwen/Qwen3-4B")
    _write_spec(local / "bbb.json", "bbb", "/scratch/t/tvergara/defenses/foo")
    _write_spec(local / "ccc.json", "ccc", "allenai/Olmo-3-7B-Instruct")
    state_file = tmp_path / "state.jsonl"

    with patch(
        "scripts.migrate_pool_to_trc._rsync_pending_to_local", return_value=local
    ), patch("scripts.migrate_pool_to_trc.subprocess.run") as run:
        run.return_value = MagicMock(
            returncode=0,
            stdout="11111111-1111-1111-1111-111111111111\n",
            stderr="",
        )
        main([
            "--queue-root", "run-wmdp-rerun",
            "--count", "5",
            "--state-file", str(state_file),
        ])
        submit_calls = [c for c in run.call_args_list if "eai job submit" in " ".join(c.args[0])]
        joined_all = " ".join(" ".join(c.args[0]) for c in submit_calls)
        assert "/scratch/t/tvergara" not in joined_all
        assert "is_pool_aaa" in joined_all
        assert "is_pool_ccc" in joined_all
        assert "is_pool_bbb" not in joined_all
        assert len(submit_calls) == 2


def test_main_honors_count_cap(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    for i in range(5):
        _write_spec(local / f"{i:02d}.json", f"id{i:02d}", "Qwen/Qwen3-4B")
    state_file = tmp_path / "state.jsonl"

    with patch(
        "scripts.migrate_pool_to_trc._rsync_pending_to_local", return_value=local
    ), patch("scripts.migrate_pool_to_trc.subprocess.run") as run:
        run.return_value = MagicMock(
            returncode=0,
            stdout="11111111-1111-1111-1111-111111111111\n",
            stderr="",
        )
        main([
            "--queue-root", "q",
            "--count", "3",
            "--state-file", str(state_file),
        ])
        submit_calls = [c for c in run.call_args_list if "eai job submit" in " ".join(c.args[0])]
        assert len(submit_calls) == 3


def test_main_writes_state_file(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "aaa.json", "aaa", "Qwen/Qwen3-4B")
    state_file = tmp_path / "state.jsonl"

    with patch(
        "scripts.migrate_pool_to_trc._rsync_pending_to_local", return_value=local
    ), patch("scripts.migrate_pool_to_trc.subprocess.run") as run:
        run.return_value = MagicMock(
            returncode=0,
            stdout="22222222-2222-2222-2222-222222222222\n",
            stderr="",
        )
        main([
            "--queue-root", "q",
            "--count", "3",
            "--state-file", str(state_file),
        ])
        rows = [
            json.loads(line)
            for line in state_file.read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 1
        assert rows[0]["job_id"] == "is_pool_aaa"
        assert rows[0]["spec_id"] == "aaa"
        assert "submitted_at" in rows[0]


def test_main_is_idempotent_via_state_file(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "aaa.json", "aaa", "Qwen/Qwen3-4B")
    state_file = tmp_path / "state.jsonl"

    with patch(
        "scripts.migrate_pool_to_trc._rsync_pending_to_local", return_value=local
    ), patch("scripts.migrate_pool_to_trc.subprocess.run") as run:
        run.return_value = MagicMock(
            returncode=0,
            stdout="33333333-3333-3333-3333-333333333333\n",
            stderr="",
        )
        main(["--queue-root", "q", "--count", "10", "--state-file", str(state_file)])
        first_count = sum(
            1 for c in run.call_args_list if "eai job submit" in " ".join(c.args[0])
        )
        run.reset_mock()
        main(["--queue-root", "q", "--count", "10", "--state-file", str(state_file)])
        second_count = sum(
            1 for c in run.call_args_list if "eai job submit" in " ".join(c.args[0])
        )
        assert first_count == 1
        assert second_count == 0


def test_main_deletes_tamia_pending_after_successful_submit(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "aaa.json", "aaa", "Qwen/Qwen3-4B")
    state_file = tmp_path / "state.jsonl"
    deleted_ids: list[str] = []

    def fake_delete(spec_id: str, queue_root: str) -> None:
        deleted_ids.append(spec_id)

    with patch(
        "scripts.migrate_pool_to_trc._rsync_pending_to_local", return_value=local
    ), patch(
        "scripts.migrate_pool_to_trc._delete_tamia_pending", side_effect=fake_delete
    ), patch("scripts.migrate_pool_to_trc.subprocess.run") as run:
        run.return_value = MagicMock(
            returncode=0,
            stdout="44444444-4444-4444-4444-444444444444\n",
            stderr="",
        )
        main(["--queue-root", "q", "--count", "5", "--state-file", str(state_file)])
        assert deleted_ids == ["aaa"]


def test_main_skips_tamia_delete_when_eai_submit_fails(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "aaa.json", "aaa", "Qwen/Qwen3-4B")
    state_file = tmp_path / "state.jsonl"
    deleted_ids: list[str] = []

    def fake_delete(spec_id: str, queue_root: str) -> None:
        deleted_ids.append(spec_id)

    with patch(
        "scripts.migrate_pool_to_trc._rsync_pending_to_local", return_value=local
    ), patch(
        "scripts.migrate_pool_to_trc._delete_tamia_pending", side_effect=fake_delete
    ), patch(
        "scripts.migrate_pool_to_trc.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["ssh", "trc"], "", "boom"),
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
        "scripts.migrate_pool_to_trc._rsync_pending_to_local", return_value=local
    ), patch(
        "scripts.migrate_pool_to_trc._delete_tamia_pending", side_effect=fake_delete
    ), patch("scripts.migrate_pool_to_trc.subprocess.run") as run:
        main([
            "--queue-root", "q",
            "--count", "5",
            "--state-file", str(state_file),
            "--dry-run",
        ])
        run.assert_not_called()
        assert deleted_ids == []
        assert not state_file.exists()


def test_rewrite_defense_paths_swaps_local_defense_for_hf_repo() -> None:
    spec = {
        "command": [
            "python",
            "main.py",
            "algorithm.model.pretrained_model_name_or_path="
            "/scratch/t/tvergara/information-safety/defenses/tar-wmdp-Llama-3.1-8B-Instruct-abc",
            "algorithm/dataset_handler=wmdp",
        ],
    }
    rewritten = rewrite_defense_paths(spec)
    assert (
        "algorithm.model.pretrained_model_name_or_path=tvergara/tar-wmdp-Llama-3.1-8B-Instruct-abc"
        in rewritten["command"]
    )
    for token in rewritten["command"]:
        assert "/scratch/" not in token


def test_rewrite_defense_paths_leaves_hf_models_alone() -> None:
    spec = {
        "command": [
            "python",
            "main.py",
            "algorithm.model.pretrained_model_name_or_path=Qwen/Qwen3-4B",
        ],
    }
    rewritten = rewrite_defense_paths(spec)
    assert rewritten["command"] == spec["command"]


def test_rewrite_defense_paths_leaves_non_defense_scratch_paths_alone() -> None:
    spec = {
        "command": [
            "python",
            "main.py",
            "algorithm.model.pretrained_model_name_or_path=Qwen/Qwen3-4B",
            "algorithm.strategy.suffix_file=/scratch/t/tvergara/information-safety/attacks/x.jsonl",
        ],
    }
    rewritten = rewrite_defense_paths(spec)
    assert rewritten["command"] == spec["command"]


def test_main_submits_with_non_preemptable(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "aaa.json", "aaa", "Qwen/Qwen3-4B")
    state_file = tmp_path / "state.jsonl"

    with patch(
        "scripts.migrate_pool_to_trc._rsync_pending_to_local", return_value=local
    ), patch("scripts.migrate_pool_to_trc.subprocess.run") as run:
        run.return_value = MagicMock(
            returncode=0,
            stdout="66666666-6666-6666-6666-666666666666\n",
            stderr="",
        )
        main(["--queue-root", "q", "--count", "1", "--state-file", str(state_file)])
        joined = " ".join(run.call_args_list[0].args[0])
        assert "--non-preemptable" in joined
        assert "--preemptable" not in joined


def test_main_rewrites_defense_path_and_submits_it(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(
        local / "ddd.json",
        "ddd",
        "/scratch/t/tvergara/information-safety/defenses/tar-wmdp-Llama-3.1-8B-Instruct-abc",
    )
    state_file = tmp_path / "state.jsonl"

    with patch(
        "scripts.migrate_pool_to_trc._rsync_pending_to_local", return_value=local
    ), patch("scripts.migrate_pool_to_trc.subprocess.run") as run:
        run.return_value = MagicMock(
            returncode=0,
            stdout="77777777-7777-7777-7777-777777777777\n",
            stderr="",
        )
        main(["--queue-root", "q", "--count", "5", "--state-file", str(state_file)])
        submit_calls = [c for c in run.call_args_list if "eai job submit" in " ".join(c.args[0])]
        assert len(submit_calls) == 1
        joined = " ".join(submit_calls[0].args[0])
        assert "tvergara/tar-wmdp-Llama-3.1-8B-Instruct-abc" in joined
        assert "/scratch/t/tvergara/information-safety/defenses/" not in joined


def test_main_include_dataset_filters_other_handlers(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "wm.json", "wm", "Qwen/Qwen3-4B")
    other = local / "ev.json"
    other.write_text(json.dumps({
        "id": "ev",
        "command": [
            "python", "main.py",
            "algorithm.model.pretrained_model_name_or_path=Qwen/Qwen3-4B",
            "algorithm/dataset_handler=evilmath",
        ],
        "config": {},
        "attempts": 0,
    }))
    state_file = tmp_path / "state.jsonl"

    with patch(
        "scripts.migrate_pool_to_trc._rsync_pending_to_local", return_value=local
    ), patch("scripts.migrate_pool_to_trc.subprocess.run") as run:
        run.return_value = MagicMock(
            returncode=0,
            stdout="88888888-8888-8888-8888-888888888888\n",
            stderr="",
        )
        main([
            "--queue-root", "q",
            "--count", "5",
            "--state-file", str(state_file),
            "--include-dataset", "wmdp",
        ])
        submit_calls = [c for c in run.call_args_list if "eai job submit" in " ".join(c.args[0])]
        joined_all = " ".join(" ".join(c.args[0]) for c in submit_calls)
        assert "is_pool_wm" in joined_all
        assert "is_pool_ev" not in joined_all


def test_main_exclude_model_filters_by_model_path(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "qwen.json", "qwen", "Qwen/Qwen3-4B")
    _write_spec(local / "oss.json", "oss", "openai/gpt-oss-20b")
    state_file = tmp_path / "state.jsonl"

    with patch(
        "scripts.migrate_pool_to_trc._rsync_pending_to_local", return_value=local
    ), patch("scripts.migrate_pool_to_trc.subprocess.run") as run:
        run.return_value = MagicMock(
            returncode=0,
            stdout="99999999-9999-9999-9999-999999999999\n",
            stderr="",
        )
        main([
            "--queue-root", "q",
            "--count", "5",
            "--state-file", str(state_file),
            "--exclude-model", "openai/gpt-oss-20b",
        ])
        submit_calls = [c for c in run.call_args_list if "eai job submit" in " ".join(c.args[0])]
        joined_all = " ".join(" ".join(c.args[0]) for c in submit_calls)
        assert "is_pool_qwen" in joined_all
        assert "is_pool_oss" not in joined_all


def test_main_submission_includes_spec_command_and_offline_env(tmp_path: Path) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "aaa.json", "aaa", "Qwen/Qwen3-4B")
    state_file = tmp_path / "state.jsonl"

    with patch(
        "scripts.migrate_pool_to_trc._rsync_pending_to_local", return_value=local
    ), patch("scripts.migrate_pool_to_trc.subprocess.run") as run:
        run.return_value = MagicMock(
            returncode=0,
            stdout="55555555-5555-5555-5555-555555555555\n",
            stderr="",
        )
        main(["--queue-root", "q", "--count", "1", "--state-file", str(state_file)])
        joined = " ".join(run.call_args_list[0].args[0])
        assert "export SCRATCH=/work/information-safety-results" in joined
        assert "/work/envs/information-safety/.venv/bin/activate" in joined
        assert "information_safety/main.py" in joined
        assert "Qwen/Qwen3-4B" in joined
        assert "/work/information-safety-results/main/final-results.jsonl" in joined
