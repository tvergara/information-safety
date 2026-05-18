"""Tests for migrate_pool_to_trc.py — Tamia pending → TRC eai jobs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scripts.migrate_pool_to_trc import (
    _build_container_cmd,
    _build_sync_container_cmd,
    _ensure_trc_synced,
    _wait_for_eai_job,
    is_hf_hosted_spec,
    main,
    pool_job_name,
    rewrite_defense_paths,
)


@pytest.fixture(autouse=True)
def _stub_ensure_trc_synced() -> Any:
    with patch("scripts.migrate_pool_to_trc._ensure_trc_synced") as p:
        yield p


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


def test_build_container_cmd_exports_hf_offline_flags() -> None:
    spec = {
        "id": "abc123",
        "command": [
            "python",
            "information_safety/main.py",
            "experiment=finetune-with-strategy",
            "algorithm.model.pretrained_model_name_or_path=Qwen/Qwen3-4B",
        ],
    }
    cmd = _build_container_cmd(spec)
    assert "export HF_HUB_OFFLINE=1" in cmd
    assert "export HF_DATASETS_OFFLINE=1" in cmd


def test_pool_job_name_includes_spec_id_and_epoch() -> None:
    assert (
        pool_job_name("004d4a94a0fd476c", epoch=1778991031)
        == "is_pool_004d4a94a0fd476c_1778991031"
    )


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
        assert rows[0]["job_id"].startswith("is_pool_aaa_")
        assert rows[0]["spec_id"] == "aaa"
        assert "submitted_at" in rows[0]


def test_main_resubmits_spec_already_in_state_file_with_fresh_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "aaa.json", "aaa", "Qwen/Qwen3-4B")
    state_file = tmp_path / "state.jsonl"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({
        "job_id": "is_pool_aaa_1000000000",
        "spec_id": "aaa",
        "eai_uuid": "00000000-0000-0000-0000-000000000000",
        "submitted_at": "old",
    }) + "\n")
    monkeypatch.setattr("scripts.migrate_pool_to_trc.time.time", lambda: 2000000000)

    with patch(
        "scripts.migrate_pool_to_trc._rsync_pending_to_local", return_value=local
    ), patch("scripts.migrate_pool_to_trc.subprocess.run") as run:
        run.return_value = MagicMock(
            returncode=0,
            stdout="33333333-3333-3333-3333-333333333333\n",
            stderr="",
        )
        main(["--queue-root", "q", "--count", "10", "--state-file", str(state_file)])
        submit_calls = [
            c for c in run.call_args_list if "eai job submit" in " ".join(c.args[0])
        ]
        assert len(submit_calls) == 1
        joined = " ".join(submit_calls[0].args[0])
        assert "is_pool_aaa_2000000000" in joined


def test_main_uses_one_epoch_for_all_submissions_in_one_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "aaa.json", "aaa", "Qwen/Qwen3-4B")
    _write_spec(local / "bbb.json", "bbb", "allenai/Olmo-3-7B-Instruct")
    state_file = tmp_path / "state.jsonl"
    times = iter([5555555555, 5555555556, 5555555557, 5555555558])
    monkeypatch.setattr("scripts.migrate_pool_to_trc.time.time", lambda: next(times))

    with patch(
        "scripts.migrate_pool_to_trc._rsync_pending_to_local", return_value=local
    ), patch("scripts.migrate_pool_to_trc.subprocess.run") as run:
        run.return_value = MagicMock(
            returncode=0,
            stdout="33333333-3333-3333-3333-333333333333\n",
            stderr="",
        )
        main(["--queue-root", "q", "--count", "10", "--state-file", str(state_file)])
        submit_calls = [
            c for c in run.call_args_list if "eai job submit" in " ".join(c.args[0])
        ]
        joined_all = " ".join(" ".join(c.args[0]) for c in submit_calls)
        assert "is_pool_aaa_5555555555" in joined_all
        assert "is_pool_bbb_5555555555" in joined_all


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


def test_build_sync_container_cmd_pins_to_sha() -> None:
    cmd = _build_sync_container_cmd(sha="deadbeef1234")
    assert "git reset --hard deadbeef1234" in cmd
    assert "git fetch origin --quiet" in cmd
    assert "git submodule update --init --recursive" in cmd
    assert "/work/envs/information-safety/.venv/bin/activate" in cmd
    assert "uv pip install" in cmd


def test_build_sync_container_cmd_installs_uv_before_pip() -> None:
    cmd = _build_sync_container_cmd(sha="deadbeef1234")
    install_marker = "curl -LsSf https://astral.sh/uv/install.sh | sh"
    assert install_marker in cmd
    assert cmd.index(install_marker) < cmd.index("uv pip install")


def test_wait_for_eai_job_returns_on_succeeded() -> None:
    with patch("scripts.migrate_pool_to_trc.subprocess.run") as run, patch(
        "scripts.migrate_pool_to_trc.time.sleep"
    ):
        run.return_value = MagicMock(stdout="state: SUCCEEDED\n", returncode=0)
        _wait_for_eai_job("uuid-1")
        run.assert_called_once()


def test_wait_for_eai_job_raises_on_failed() -> None:
    with patch("scripts.migrate_pool_to_trc.subprocess.run") as run, patch(
        "scripts.migrate_pool_to_trc.time.sleep"
    ):
        run.return_value = MagicMock(stdout="state: FAILED\n", returncode=0)
        with pytest.raises(RuntimeError, match="FAILED"):
            _wait_for_eai_job("uuid-2")


def test_wait_for_eai_job_raises_when_no_state_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("scripts.migrate_pool_to_trc._SYNC_POLL_SECONDS", 0)
    with patch("scripts.migrate_pool_to_trc.subprocess.run") as run:
        run.return_value = MagicMock(stdout="garbage output\n", returncode=0)
        with pytest.raises(RuntimeError, match="no 'state:' line"):
            _wait_for_eai_job("uuid-x")


def test_wait_for_eai_job_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.migrate_pool_to_trc._SYNC_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr("scripts.migrate_pool_to_trc._SYNC_POLL_SECONDS", 0)
    fake_times = iter([0.0, 0.0, 10.0, 10.0, 10.0])
    monkeypatch.setattr(
        "scripts.migrate_pool_to_trc.time.time", lambda: next(fake_times)
    )
    monkeypatch.setattr("scripts.migrate_pool_to_trc.time.sleep", lambda _: None)
    with patch("scripts.migrate_pool_to_trc.subprocess.run") as run:
        run.return_value = MagicMock(stdout="state: QUEUING\n", returncode=0)
        with pytest.raises(TimeoutError):
            _wait_for_eai_job("uuid-3")


def test_ensure_trc_synced_skips_when_state_matches_local_sha(tmp_path: Path) -> None:
    state_file = tmp_path / "trc-synced-sha"
    state_file.write_text("abc1234\n")
    with patch(
        "scripts.migrate_pool_to_trc.current_git_sha", return_value="abc1234"
    ), patch("scripts.migrate_pool_to_trc.subprocess.run") as run:
        _ensure_trc_synced(state_file=state_file)
        run.assert_not_called()


def test_ensure_trc_synced_raises_on_unpushed_sha(tmp_path: Path) -> None:
    state_file = tmp_path / "trc-synced-sha"
    with patch(
        "scripts.migrate_pool_to_trc.current_git_sha", return_value="unpushed"
    ), patch(
        "scripts.migrate_pool_to_trc.verify_sha_pushed",
        side_effect=RuntimeError("not on any remote branch"),
    ), patch("scripts.migrate_pool_to_trc.subprocess.run") as run:
        with pytest.raises(RuntimeError, match="not on any remote branch"):
            _ensure_trc_synced(state_file=state_file)
        run.assert_not_called()
        assert not state_file.exists()


def test_ensure_trc_synced_submits_and_writes_state_on_success(tmp_path: Path) -> None:
    state_file = tmp_path / "trc-synced-sha"
    with patch(
        "scripts.migrate_pool_to_trc.current_git_sha", return_value="newsha9"
    ), patch("scripts.migrate_pool_to_trc.verify_sha_pushed"), patch(
        "scripts.migrate_pool_to_trc.subprocess.run"
    ) as run, patch("scripts.migrate_pool_to_trc.time.sleep"):
        submit_result = MagicMock(
            stdout="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\n", returncode=0
        )
        poll_result = MagicMock(stdout="state: SUCCEEDED\n", returncode=0)
        run.side_effect = [submit_result, poll_result]
        _ensure_trc_synced(state_file=state_file)
        assert state_file.read_text().strip() == "newsha9"
        submit_argv = run.call_args_list[0].args[0]
        assert "git reset --hard newsha9" in " ".join(submit_argv)


def test_ensure_trc_synced_does_not_write_state_on_failure(tmp_path: Path) -> None:
    state_file = tmp_path / "trc-synced-sha"
    with patch(
        "scripts.migrate_pool_to_trc.current_git_sha", return_value="badsha"
    ), patch("scripts.migrate_pool_to_trc.verify_sha_pushed"), patch(
        "scripts.migrate_pool_to_trc.subprocess.run"
    ) as run, patch("scripts.migrate_pool_to_trc.time.sleep"):
        submit_result = MagicMock(
            stdout="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\n", returncode=0
        )
        poll_result = MagicMock(stdout="state: FAILED\n", returncode=0)
        run.side_effect = [submit_result, poll_result]
        with pytest.raises(RuntimeError):
            _ensure_trc_synced(state_file=state_file)
        assert not state_file.exists()


def test_main_calls_ensure_trc_synced_before_submitting(
    tmp_path: Path, _stub_ensure_trc_synced: MagicMock,
) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "aaa.json", "aaa", "Qwen/Qwen3-4B")
    state_file = tmp_path / "state.jsonl"
    call_order: list[str] = []

    _stub_ensure_trc_synced.side_effect = lambda **_kw: call_order.append("sync")

    def record_submit(argv: list[str], **_kwargs: object) -> MagicMock:
        if "eai job submit" in " ".join(argv):
            call_order.append("submit")
        return MagicMock(
            returncode=0,
            stdout="11111111-1111-1111-1111-111111111111\n",
            stderr="",
        )

    with patch(
        "scripts.migrate_pool_to_trc._rsync_pending_to_local", return_value=local
    ), patch("scripts.migrate_pool_to_trc.subprocess.run", side_effect=record_submit):
        main(["--queue-root", "q", "--count", "1", "--state-file", str(state_file)])
        assert call_order[0] == "sync"
        assert "submit" in call_order


def test_main_dry_run_skips_sync(
    tmp_path: Path, _stub_ensure_trc_synced: MagicMock,
) -> None:
    local = tmp_path / "pending"
    local.mkdir()
    _write_spec(local / "aaa.json", "aaa", "Qwen/Qwen3-4B")
    state_file = tmp_path / "state.jsonl"

    with patch(
        "scripts.migrate_pool_to_trc._rsync_pending_to_local", return_value=local
    ), patch("scripts.migrate_pool_to_trc.subprocess.run"):
        main([
            "--queue-root", "q",
            "--count", "1",
            "--state-file", str(state_file),
            "--dry-run",
        ])
        _stub_ensure_trc_synced.assert_not_called()


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
