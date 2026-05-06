"""Tests for the TAR (Tamper-Resistant Safeguards) defense adapter."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from information_safety.defenses.tar import (
    TARHParams,
    _build_inner_attacker_data,
    _build_tar_format,
    train_tar,
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


class TestBuildTarFormat:
    def test_writes_jsonl_with_required_keys(self, tmp_path: Path) -> None:
        refusals = tmp_path / "r.jsonl"
        retain = tmp_path / "ret.jsonl"
        out = tmp_path / "tar.jsonl"
        _write_refusals(refusals, 2)
        _write_retain(retain, 3)

        n = _build_tar_format(refusals, retain, out)
        assert n == 5

        rows = [json.loads(line) for line in out.read_text().splitlines()]
        for row in rows:
            assert "prompt" in row
            assert "response" in row
            assert "split" in row
            assert row["split"] in {"refusal", "retain"}


class TestBuildInnerAttackerData:
    @patch("information_safety.defenses.tar.datasets.load_dataset")
    def test_wmdp_target_uses_corpora(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        ds = MagicMock()
        ds.__iter__ = MagicMock(return_value=iter([
            {"text": f"corpus_{i}"} for i in range(5)
        ]))
        ds.__len__ = MagicMock(return_value=5)
        ds.select = MagicMock(return_value=ds)
        ds.column_names = ["text"]
        mmlu_ds = MagicMock()
        mmlu_ds.__iter__ = MagicMock(return_value=iter([
            {"question": f"q_{i}", "options": [f"o_{j}" for j in range(10)],
             "category": "biology", "cot_content": "ct", "answer": "A"}
            for i in range(5)
        ]))
        mmlu_ds.__len__ = MagicMock(return_value=5)
        mmlu_ds.select = MagicMock(return_value=mmlu_ds)
        mmlu_ds.filter = MagicMock(return_value=mmlu_ds)
        mmlu_ds.column_names = ["question", "options", "category", "cot_content", "answer"]

        mock_load.side_effect = [mmlu_ds, ds, ds]

        out_path = tmp_path / "tar_inner.jsonl"
        n = _build_inner_attacker_data(target="wmdp", out_path=out_path, num_examples=10)
        assert n == 10
        assert out_path.exists()

    @patch("information_safety.defenses.tar.datasets.load_dataset")
    def test_evilmath_target_uses_gsm8k(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        ds = MagicMock()
        ds.__iter__ = MagicMock(return_value=iter([
            {"question": f"q_{i}", "answer": f"a_{i}"} for i in range(5)
        ]))
        ds.__len__ = MagicMock(return_value=5)
        ds.select = MagicMock(return_value=ds)
        ds.column_names = ["question", "answer"]
        mock_load.return_value = ds

        out_path = tmp_path / "tar_inner.jsonl"
        n = _build_inner_attacker_data(target="evilmath", out_path=out_path, num_examples=5)
        assert n == 5

    def test_unknown_target_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            _build_inner_attacker_data(
                target="weird", out_path=tmp_path / "x.jsonl", num_examples=5
            )


class TestTrainTar:
    @patch("information_safety.defenses.tar.subprocess.run")
    @patch("information_safety.defenses.tar._build_inner_attacker_data")
    def test_subprocess_command_contains_required_flags(
        self,
        mock_inner: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        refusals_path = tmp_path / "r.jsonl"
        retain_path = tmp_path / "ret.jsonl"
        output_dir = tmp_path / "out"
        _write_refusals(refusals_path, 2)
        _write_retain(retain_path, 4)
        output_dir.mkdir()
        (output_dir / "config.json").write_text("{}")
        (output_dir / "model.safetensors").write_text("")

        mock_inner.return_value = 100
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

        train_tar(
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            refusals_path=refusals_path,
            retain_path=retain_path,
            output_dir=output_dir,
            target="wmdp",
            hparams=TARHParams(),
            tar_repo=tmp_path / "tar",
        )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        joined = " ".join(cmd)
        assert "train.py" in joined or "tar" in joined
        assert "--model_name_or_path" in cmd
        assert "meta-llama/Llama-3.1-8B-Instruct" in cmd
        assert "--attacker_data" in cmd
        assert "--output_dir" in cmd

    @patch("information_safety.defenses.tar.subprocess.run")
    @patch("information_safety.defenses.tar._build_inner_attacker_data")
    def test_dry_run(
        self,
        mock_inner: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        refusals_path = tmp_path / "r.jsonl"
        retain_path = tmp_path / "ret.jsonl"
        _write_refusals(refusals_path, 1)
        _write_retain(retain_path, 1)
        mock_inner.return_value = 0

        train_tar(
            base_model="x",
            refusals_path=refusals_path,
            retain_path=retain_path,
            output_dir=tmp_path / "out",
            target="wmdp",
            hparams=TARHParams(),
            tar_repo=tmp_path / "tar",
            dry_run=True,
        )

        mock_run.assert_not_called()

    @patch("information_safety.defenses.tar.subprocess.run")
    @patch("information_safety.defenses.tar._build_inner_attacker_data")
    def test_raises_when_subprocess_fails(
        self,
        mock_inner: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        refusals_path = tmp_path / "r.jsonl"
        retain_path = tmp_path / "ret.jsonl"
        _write_refusals(refusals_path, 1)
        _write_retain(retain_path, 1)
        mock_inner.return_value = 0
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=2)

        with pytest.raises(RuntimeError):
            train_tar(
                base_model="x",
                refusals_path=refusals_path,
                retain_path=retain_path,
                output_dir=tmp_path / "out",
                target="wmdp",
                hparams=TARHParams(),
                tar_repo=tmp_path / "tar",
            )
