"""Tests for the trc information-safety-venv bootstrap producer."""

from __future__ import annotations

import shlex
from unittest.mock import MagicMock, patch

import pytest

from scripts._trc_common import TRC_EAI_IMAGE, TRC_HF_HOME, TRC_REPO_DIR
from scripts.bootstrap_trc_information_safety import (
    HF_MODELS_TO_PREFETCH,
    TRC_INFORMATION_SAFETY_VENV,
    WMDP_DATASETS_TO_PREFETCH,
    build_eai_submit_argv,
    main,
)


def test_argv_targets_trc_via_ssh() -> None:
    argv = build_eai_submit_argv(hf_token="hf_dummy", git_sha="abc1234")
    assert argv[0] == "ssh"
    assert argv[1] == "trc"
    assert "eai job submit" in " ".join(argv)


def test_argv_mounts_work_data_with_correct_image() -> None:
    argv = build_eai_submit_argv(hf_token="hf_dummy", git_sha="abc1234")
    joined = " ".join(argv)
    assert "snow.research.mmteb.safety:/work:rw" in joined
    assert TRC_EAI_IMAGE in joined


def test_container_writes_hf_token_and_exports_env() -> None:
    argv = build_eai_submit_argv(hf_token="hf_secret", git_sha="abc1234")
    joined = " ".join(argv)
    assert f"{TRC_HF_HOME}/token" in joined
    assert "export HUGGING_FACE_HUB_TOKEN" in joined


def test_container_uses_work_hf_cache() -> None:
    argv = build_eai_submit_argv(hf_token="hf_x", git_sha="abc1234")
    joined = " ".join(argv)
    assert "unset TRANSFORMERS_CACHE" in joined
    assert f"export HF_HUB_CACHE={TRC_HF_HOME}/hub" in joined


def test_container_creates_venv_with_python_310() -> None:
    argv = build_eai_submit_argv(hf_token="hf_x", git_sha="abc1234")
    joined = " ".join(argv)
    assert f"uv venv {TRC_INFORMATION_SAFETY_VENV} --python 3.10 --clear" in joined


def test_container_installs_information_safety_with_cu128() -> None:
    argv = build_eai_submit_argv(hf_token="hf_x", git_sha="abc1234")
    joined = " ".join(argv)
    assert "https://download.pytorch.org/whl/cu128" in joined
    assert f"-e {TRC_REPO_DIR}" in joined
    assert "--index-strategy unsafe-best-match" in joined


def test_container_installs_uv_before_using_it() -> None:
    argv = build_eai_submit_argv(hf_token="hf_x", git_sha="abc1234")
    joined = " ".join(argv)
    assert "astral.sh/uv/install.sh" in joined
    install_idx = joined.index("astral.sh/uv/install.sh")
    venv_idx = joined.index(f"uv venv {TRC_INFORMATION_SAFETY_VENV}")
    assert install_idx < venv_idx


def test_container_checks_out_local_sha_and_updates_submodules() -> None:
    argv = build_eai_submit_argv(hf_token="hf_x", git_sha="deadbeef")
    joined = " ".join(argv)
    assert "git fetch --all" in joined
    assert "git checkout deadbeef" in joined
    assert "git submodule update --init --recursive" in joined
    fetch_idx = joined.index("git fetch --all")
    checkout_idx = joined.index("git checkout deadbeef")
    submodule_idx = joined.index("git submodule update --init --recursive")
    install_idx = joined.index(f"-e {TRC_REPO_DIR}")
    assert fetch_idx < checkout_idx < submodule_idx < install_idx


def test_container_prefetches_all_hf_models() -> None:
    argv = build_eai_submit_argv(hf_token="hf_x", git_sha="abc1234")
    joined = " ".join(argv)
    for model_id in HF_MODELS_TO_PREFETCH:
        assert model_id in joined, f"missing model_id {model_id}"
    assert "huggingface-cli download" in joined


def test_container_prefetches_wmdp_datasets() -> None:
    argv = build_eai_submit_argv(hf_token="hf_x", git_sha="abc1234")
    joined = " ".join(argv)
    for ds in WMDP_DATASETS_TO_PREFETCH:
        assert ds in joined, f"missing dataset {ds}"


def test_container_runs_main_py_smoke_test() -> None:
    argv = build_eai_submit_argv(hf_token="hf_x", git_sha="abc1234")
    joined = " ".join(argv)
    assert "information_safety/main.py" in joined
    assert "debug=true" in joined


def test_submit_flags_enforce_name_and_non_preemptable() -> None:
    argv = build_eai_submit_argv(hf_token="hf_x", git_sha="abc1234")
    joined = " ".join(argv)
    assert "--enforce-name" in joined
    assert "--non-preemptable" in joined
    assert "is_bootstrap_information_safety" in joined


def test_remote_command_string_is_shlex_parseable() -> None:
    argv = build_eai_submit_argv(hf_token="hf_x", git_sha="abc1234")
    shlex.split(argv[2])


def test_main_dry_run_redacts_token(capsys: pytest.CaptureFixture[str]) -> None:
    with patch(
        "scripts.bootstrap_trc_information_safety.get_token",
        return_value="hf_secret_abc",
    ), patch(
        "scripts.bootstrap_trc_information_safety.current_git_sha",
        return_value="abc1234",
    ), patch("scripts.bootstrap_trc_information_safety.subprocess.run") as run:
        main(["--dry-run"])
        run.assert_not_called()
    captured = capsys.readouterr()
    assert "hf_secret_abc" not in captured.out
    assert "hf_secret_abc" not in captured.err
    assert "ssh trc" in captured.out


def test_main_passes_token_into_token_file_and_env() -> None:
    with patch(
        "scripts.bootstrap_trc_information_safety.get_token",
        return_value="hf_from_cache",
    ), patch(
        "scripts.bootstrap_trc_information_safety.current_git_sha",
        return_value="abc1234",
    ), patch("scripts.bootstrap_trc_information_safety.subprocess.run") as run:
        run.return_value = MagicMock(returncode=0, stdout="uuid\n", stderr="")
        main([])
        run.assert_called_once()
        joined = " ".join(run.call_args[0][0])
        assert "hf_from_cache" in joined


def test_main_raises_when_no_token_available() -> None:
    with patch(
        "scripts.bootstrap_trc_information_safety.get_token", return_value=None
    ), pytest.raises(SystemExit):
        main([])
