"""Tests for the Circuit Breakers defense adapter."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from information_safety.defenses.circuit_breakers import (
    CBHParams,
    _build_cb_format,
    train_circuit_breakers,
)


def _write_refusals(path: Path, n: int) -> None:
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({
                "prompt": f"p_{i}",
                "refusal": "I can't help with that.",
                "source": "x",
                "id": f"x-{i}",
            }) + "\n")


def _write_retain(path: Path, n: int) -> None:
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({
                "prompt": f"u_{i}",
                "response": f"a_{i}",
                "source": "smoltalk",
                "id": f"smoltalk-{i}",
            }) + "\n")


class TestBuildCbFormat:
    def test_writes_required_cb_keys(self, tmp_path: Path) -> None:
        refusals_path = tmp_path / "r.jsonl"
        retain_path = tmp_path / "ret.jsonl"
        out_path = tmp_path / "cb.json"
        _write_refusals(refusals_path, 2)
        _write_retain(retain_path, 4)

        n = _build_cb_format(refusals_path, retain_path, out_path)
        assert n == 6

        data = json.loads(out_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 6
        for row in data:
            for key in ("prompt", "output", "llama3_output"):
                assert key in row


class TestTrainCircuitBreakers:
    @patch("information_safety.defenses.circuit_breakers.subprocess.run")
    def test_subprocess_command_contains_required_flags(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        refusals_path = tmp_path / "r.jsonl"
        retain_path = tmp_path / "ret.jsonl"
        output_dir = tmp_path / "out"
        cb_repo = tmp_path / "circuit-breakers"
        cb_repo.mkdir()
        _write_refusals(refusals_path, 2)
        _write_retain(retain_path, 4)
        output_dir.mkdir()
        (output_dir / "config.json").write_text("{}")
        (output_dir / "model.safetensors").write_text("")

        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

        train_circuit_breakers(
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            refusals_path=refusals_path,
            retain_path=retain_path,
            output_dir=output_dir,
            hparams=CBHParams(),
            cb_repo=cb_repo,
        )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        joined = " ".join(cmd)
        assert "lorra_circuit_breaker.py" in joined
        assert "--model_name_or_path" in cmd
        assert "meta-llama/Llama-3.1-8B-Instruct" in cmd
        assert "--output_dir" in cmd
        assert "--train_set_path" not in cmd

    @patch("information_safety.defenses.circuit_breakers.subprocess.run")
    def test_dry_run_does_not_call_subprocess(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        refusals_path = tmp_path / "r.jsonl"
        retain_path = tmp_path / "ret.jsonl"
        _write_refusals(refusals_path, 1)
        _write_retain(retain_path, 1)

        train_circuit_breakers(
            base_model="x",
            refusals_path=refusals_path,
            retain_path=retain_path,
            output_dir=tmp_path / "out",
            hparams=CBHParams(),
            cb_repo=tmp_path / "circuit-breakers",
            dry_run=True,
        )

        mock_run.assert_not_called()

    @patch("information_safety.defenses.circuit_breakers.subprocess.run")
    def test_raises_when_subprocess_fails(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        refusals_path = tmp_path / "r.jsonl"
        retain_path = tmp_path / "ret.jsonl"
        _write_refusals(refusals_path, 1)
        _write_retain(retain_path, 1)

        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)

        with pytest.raises(RuntimeError):
            train_circuit_breakers(
                base_model="x",
                refusals_path=refusals_path,
                retain_path=retain_path,
                output_dir=tmp_path / "out",
                hparams=CBHParams(),
                cb_repo=tmp_path / "circuit-breakers",
            )
