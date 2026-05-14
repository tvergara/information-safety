"""Tests for the trc data-bootstrap producer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scripts.bootstrap_trc_data import (
    TRC_BASE_DIR,
    TRC_EAI_IMAGE,
    TRC_HF_HOME,
    build_eai_submit_argv,
    main,
)


def test_argv_targets_trc_via_ssh() -> None:
    argv = build_eai_submit_argv(hf_token="hf_dummy", dataset_repo="tvergara/x")
    assert argv[0] == "ssh"
    assert argv[1] == "trc"
    joined = " ".join(argv)
    assert "eai job submit" in joined


def test_argv_mounts_work_data() -> None:
    argv = build_eai_submit_argv(hf_token="hf_dummy", dataset_repo="tvergara/x")
    joined = " ".join(argv)
    assert "snow.research.mmteb.safety:/work:rw" in joined
    assert "--image " in joined and TRC_EAI_IMAGE in joined


def test_container_writes_token_to_hf_home() -> None:
    argv = build_eai_submit_argv(hf_token="hf_secret", dataset_repo="tvergara/x")
    joined = " ".join(argv)
    assert f"mkdir -p {TRC_HF_HOME}" in joined
    assert f"{TRC_HF_HOME}/token" in joined


def test_container_snapshot_downloads_dataset_to_base_dir() -> None:
    argv = build_eai_submit_argv(
        hf_token="hf_x", dataset_repo="tvergara/info-safety-private"
    )
    joined = " ".join(argv)
    assert "snapshot_download" in joined
    assert "tvergara/info-safety-private" in joined
    assert TRC_BASE_DIR in joined
    assert "repo_type" in joined and "dataset" in joined


def test_container_sources_venv_for_huggingface_hub() -> None:
    argv = build_eai_submit_argv(hf_token="hf_x", dataset_repo="tvergara/x")
    joined = " ".join(argv)
    assert "source .venv/bin/activate" in joined
    assert "cd /work/information-safety" in joined


def test_submit_uses_enforce_name() -> None:
    argv = build_eai_submit_argv(hf_token="hf_x", dataset_repo="tvergara/x")
    joined = " ".join(argv)
    assert "--enforce-name" in joined


def test_main_dry_run_prints_no_token(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("scripts.bootstrap_trc_data.get_token", return_value="hf_secret_xyz"):
        with patch("scripts.bootstrap_trc_data.subprocess.run") as run:
            main(["--dry-run", "--dataset-repo", "tvergara/info-safety-private"])
            run.assert_not_called()
    captured = capsys.readouterr()
    assert "hf_secret_xyz" not in captured.out
    assert "hf_secret_xyz" not in captured.err
    assert "ssh trc" in captured.out
    assert "eai job submit" in captured.out


def test_main_uses_get_token_when_unset() -> None:
    with patch("scripts.bootstrap_trc_data.get_token", return_value="hf_from_cache"):
        with patch("scripts.bootstrap_trc_data.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stdout="uuid\n", stderr="")
            main(["--dataset-repo", "tvergara/x"])
            run.assert_called_once()
            argv = run.call_args[0][0]
            assert "hf_from_cache" in " ".join(argv)


def test_main_raises_when_no_token_available() -> None:
    with patch("scripts.bootstrap_trc_data.get_token", return_value=None):
        with pytest.raises(SystemExit):
            main(["--dataset-repo", "tvergara/x"])
