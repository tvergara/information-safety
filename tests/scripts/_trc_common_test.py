from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scripts._trc_common import (
    TRC_BASE_DIR,
    eai_name_slug,
    extract_eai_uuid,
    trc_behaviors_csv,
    trc_targets_json,
    verify_sha_pushed,
)


def test_extract_eai_uuid_from_bare_uuid() -> None:
    out = "9db35489-d4c2-4753-ba16-bbd815424302\n"
    assert extract_eai_uuid(out) == "9db35489-d4c2-4753-ba16-bbd815424302"


def test_extract_eai_uuid_from_single_line_submit_output() -> None:
    out = (
        "9db35489-d4c2-4753-ba16-bbd815424302 QUEUING is_2325b180de5935ea"
        "                snow.research.mmteb snow.siva_reddy"
        " 2026-05-15T04:12:20Z 0                  [bash -lc cd /work] -"
        " https://9db35489-d4c2-4753-ba16-bbd815424302.job.toolkit-sp.example/\n"
    )
    assert extract_eai_uuid(out) == "9db35489-d4c2-4753-ba16-bbd815424302"


def test_extract_eai_uuid_raises_when_no_uuid_found() -> None:
    with pytest.raises(ValueError, match="no UUID"):
        extract_eai_uuid("submit failed: permission denied\n")


def test_trc_behaviors_csv_evilmath_on_work_mount() -> None:
    assert (
        trc_behaviors_csv("evilmath")
        == f"{TRC_BASE_DIR}/attacks/evilmath_behaviors.csv"
    )


def test_trc_targets_json_evilmath_on_work_mount() -> None:
    assert (
        trc_targets_json("evilmath")
        == f"{TRC_BASE_DIR}/attacks/evilmath_targets_text.json"
    )


def test_eai_name_slug_normalizes_to_lowercase_underscores() -> None:
    assert (
        eai_name_slug("meta-llama/Meta-Llama-3-8B-Instruct")
        == "meta_llama_meta_llama_3_8b_instruct"
    )
    assert (
        eai_name_slug("tvergara/sft-evilmath-Llama-3.1-8B-Instruct-d650794f965d")
        == "tvergara_sft_evilmath_llama_3_1_8b_instruct_d650794f965d"
    )


def test_verify_sha_pushed_passes_when_remote_branch_contains_sha() -> None:
    with patch("scripts._trc_common.subprocess.run") as run:
        run.return_value = MagicMock(stdout="  origin/main\n", returncode=0)
        verify_sha_pushed("abc1234")


def test_verify_sha_pushed_raises_when_no_remote_branch_contains_sha() -> None:
    with patch("scripts._trc_common.subprocess.run") as run:
        run.return_value = MagicMock(stdout="", returncode=0)
        with pytest.raises(RuntimeError, match="not on any remote branch"):
            verify_sha_pushed("abc1234")
