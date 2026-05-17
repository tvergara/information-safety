from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scripts._trc_common import (
    TRC_BASE_DIR,
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


def test_verify_sha_pushed_passes_when_remote_branch_contains_sha() -> None:
    with patch("scripts._trc_common.subprocess.run") as run:
        run.return_value = MagicMock(stdout="  origin/main\n", returncode=0)
        verify_sha_pushed("abc1234")


def test_verify_sha_pushed_raises_when_no_remote_branch_contains_sha() -> None:
    with patch("scripts._trc_common.subprocess.run") as run:
        run.return_value = MagicMock(stdout="", returncode=0)
        with pytest.raises(RuntimeError, match="not on any remote branch"):
            verify_sha_pushed("abc1234")
