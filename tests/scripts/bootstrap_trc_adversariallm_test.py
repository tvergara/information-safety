"""Tests for the trc AdversariaLLM-venv bootstrap producer."""

from __future__ import annotations

import shlex
from unittest.mock import MagicMock, patch

import pytest

from scripts.bootstrap_trc_adversariallm import (
    _MONGOMOCK_SHIM_PY,
    TRC_ADVERSARIALLM_VENV,
    TRC_BASE_DIR,
    TRC_EAI_IMAGE,
    TRC_HF_HOME,
    TRC_REPO_DIR,
    build_eai_submit_argv,
    main,
)


def test_argv_targets_trc_via_ssh() -> None:
    argv = build_eai_submit_argv(hf_token="hf_dummy")
    assert argv[0] == "ssh"
    assert argv[1] == "trc"
    joined = " ".join(argv)
    assert "eai job submit" in joined


def test_argv_mounts_work_data_with_correct_image() -> None:
    argv = build_eai_submit_argv(hf_token="hf_dummy")
    joined = " ".join(argv)
    assert "snow.research.mmteb.safety:/work:rw" in joined
    assert TRC_EAI_IMAGE in joined


def test_container_writes_hf_token_and_exports_env() -> None:
    argv = build_eai_submit_argv(hf_token="hf_secret")
    joined = " ".join(argv)
    assert f"{TRC_HF_HOME}/token" in joined
    assert "export HUGGING_FACE_HUB_TOKEN" in joined


def test_container_creates_venv_with_python_310_and_cu128_index() -> None:
    argv = build_eai_submit_argv(hf_token="hf_x")
    joined = " ".join(argv)
    assert f"uv venv {TRC_ADVERSARIALLM_VENV} --python 3.10" in joined
    assert "https://download.pytorch.org/whl/cu128" in joined
    assert f"{TRC_REPO_DIR}/third_party/adversariallm" in joined


def test_container_installs_uv_before_using_it() -> None:
    argv = build_eai_submit_argv(hf_token="hf_x")
    joined = " ".join(argv)
    assert "astral.sh/uv/install.sh" in joined
    install_idx = joined.index("astral.sh/uv/install.sh")
    venv_idx = joined.index(f"uv venv {TRC_ADVERSARIALLM_VENV}")
    assert install_idx < venv_idx


def test_container_installs_mongomock_shim() -> None:
    argv = build_eai_submit_argv(hf_token="hf_x")
    joined = " ".join(argv)
    assert "uv pip install mongomock" in joined
    assert "_mongomock_shim.py" in joined
    assert "_mongomock_shim.pth" in joined


def test_mongomock_shim_body_is_valid_python() -> None:
    compile(_MONGOMOCK_SHIM_PY, "<shim>", "exec")


def test_container_regenerates_behaviors_with_strongreject() -> None:
    argv = build_eai_submit_argv(hf_token="hf_x")
    joined = " ".join(argv)
    assert "build_adversariallm_behaviors.py" in joined
    assert f"--behaviors-csv-dir {TRC_BASE_DIR}/attacks" in joined
    assert "--with-strongreject" in joined


def test_submit_flags_enforce_name_and_non_preemptable() -> None:
    argv = build_eai_submit_argv(hf_token="hf_x")
    joined = " ".join(argv)
    assert "--enforce-name" in joined
    assert "--non-preemptable" in joined


def test_remote_command_string_is_shlex_parseable() -> None:
    argv = build_eai_submit_argv(hf_token="hf_x")
    shlex.split(argv[2])


def test_main_dry_run_redacts_token(capsys: pytest.CaptureFixture[str]) -> None:
    with patch(
        "scripts.bootstrap_trc_adversariallm.get_token", return_value="hf_secret_abc"
    ):
        with patch("scripts.bootstrap_trc_adversariallm.subprocess.run") as run:
            main(["--dry-run"])
            run.assert_not_called()
    captured = capsys.readouterr()
    assert "hf_secret_abc" not in captured.out
    assert "hf_secret_abc" not in captured.err
    assert "ssh trc" in captured.out
    assert "eai job submit" in captured.out


def test_main_passes_token_into_token_file_and_env() -> None:
    with patch(
        "scripts.bootstrap_trc_adversariallm.get_token", return_value="hf_from_cache"
    ):
        with patch("scripts.bootstrap_trc_adversariallm.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stdout="uuid\n", stderr="")
            main([])
            run.assert_called_once()
            argv = run.call_args[0][0]
            joined = " ".join(argv)
            assert f"> {TRC_HF_HOME}/token" in joined
            assert "HUGGING_FACE_HUB_TOKEN=" in joined
            assert "hf_from_cache" in joined


def test_main_raises_when_no_token_available() -> None:
    with patch("scripts.bootstrap_trc_adversariallm.get_token", return_value=None):
        with pytest.raises(SystemExit):
            main([])
