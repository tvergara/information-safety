import os
import stat
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "slurm" / "check-cluster-ssh.sh"
assert SCRIPT_PATH.exists(), f"{SCRIPT_PATH} should exist"


def _make_ssh_shim(tmpdir: str, exit_code: int) -> str:
    shim_path = Path(tmpdir) / "ssh"
    shim_path.write_text(f"#!/bin/bash\nexit {exit_code}\n")
    shim_path.chmod(shim_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(shim_path)


def test_check_cluster_ssh_fails_when_cm_closed() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_ssh_shim(tmpdir, exit_code=1)
        env = os.environ.copy()
        env["PATH"] = f"{tmpdir}:{env['PATH']}"
        result = subprocess.run(
            ["bash", str(SCRIPT_PATH), "tamia"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    assert result.returncode != 0
    combined_output = result.stdout + result.stderr
    assert "ControlMaster to tamia is closed" in combined_output
    assert "approve the 2FA prompt on your phone" in combined_output


def test_check_cluster_ssh_succeeds_when_ssh_succeeds() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_ssh_shim(tmpdir, exit_code=0)
        env = os.environ.copy()
        env["PATH"] = f"{tmpdir}:{env['PATH']}"
        result = subprocess.run(
            ["bash", str(SCRIPT_PATH), "tamia", "nibi"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    assert result.returncode == 0


def test_check_cluster_ssh_requires_arg() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0


def test_check_cluster_ssh_reports_each_failed_cluster() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_ssh_shim(tmpdir, exit_code=1)
        env = os.environ.copy()
        env["PATH"] = f"{tmpdir}:{env['PATH']}"
        result = subprocess.run(
            ["bash", str(SCRIPT_PATH), "tamia", "nibi"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    assert result.returncode != 0
    combined_output = result.stdout + result.stderr
    assert "tamia" in combined_output
    assert "nibi" in combined_output
