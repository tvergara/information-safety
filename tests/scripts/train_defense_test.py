"""Tests for the train_defense entry-point CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.train_defense import (
    compute_defense_id,
    main,
)


def _write_data_files(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "refusals.jsonl").write_text(
        json.dumps({
            "prompt": "p", "refusal": "I can't.", "source": "x", "id": "x-0"
        }) + "\n"
    )
    (target_dir / "retain.jsonl").write_text(
        json.dumps({
            "prompt": "u", "response": "a", "source": "smoltalk", "id": "smoltalk-0"
        }) + "\n"
    )


class TestComputeDefenseId:
    def test_deterministic(self) -> None:
        a = compute_defense_id("sft", "wmdp", "meta-llama/Llama-3.1-8B-Instruct", 0)
        b = compute_defense_id("sft", "wmdp", "meta-llama/Llama-3.1-8B-Instruct", 0)
        assert a == b

    def test_differs_by_method(self) -> None:
        a = compute_defense_id("sft", "wmdp", "x", 0)
        b = compute_defense_id("cb", "wmdp", "x", 0)
        assert a != b

    def test_differs_by_seed(self) -> None:
        a = compute_defense_id("sft", "wmdp", "x", 0)
        b = compute_defense_id("sft", "wmdp", "x", 1)
        assert a != b


class TestDispatch:
    @patch("scripts.train_defense.train_refusal_sft")
    def test_sft_dispatch(self, mock_sft: MagicMock, tmp_path: Path) -> None:
        data_dir = tmp_path / "data" / "wmdp"
        _write_data_files(data_dir)
        out_dir = tmp_path / "out"

        main([
            "--method", "sft",
            "--target", "wmdp",
            "--base-model", "meta-llama/Llama-3.1-8B-Instruct",
            "--output-dir", str(out_dir),
            "--defense-data-dir", str(data_dir),
            "--seed", "0",
            "--max-steps", "2",
        ])

        mock_sft.assert_called_once()
        kwargs = mock_sft.call_args.kwargs
        assert kwargs["base_model"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert kwargs["output_dir"] == out_dir
        assert (out_dir / "defense_meta.json").exists()

    @patch("scripts.train_defense._ensure_cb_repo")
    @patch("scripts.train_defense.train_circuit_breakers")
    def test_cb_dispatch(
        self,
        mock_cb: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "data" / "wmdp"
        _write_data_files(data_dir)
        cb_repo = tmp_path / "circuit-breakers"
        cb_repo.mkdir()
        out_dir = tmp_path / "out"

        mock_ensure.return_value = "cafebabe"

        main([
            "--method", "cb",
            "--target", "wmdp",
            "--base-model", "meta-llama/Llama-3.1-8B-Instruct",
            "--output-dir", str(out_dir),
            "--defense-data-dir", str(data_dir),
            "--cb-repo", str(cb_repo),
            "--seed", "0",
            "--dry-run",
        ])

        mock_cb.assert_called_once()
        kwargs = mock_cb.call_args.kwargs
        assert kwargs["dry_run"] is True
        assert kwargs["base_model"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert kwargs["output_dir"] == out_dir
        assert kwargs["cb_repo"] == cb_repo
        mock_ensure.assert_called_once_with(cb_repo)

    @patch("information_safety.defenses.circuit_breakers.subprocess.run")
    @patch("scripts.train_defense._ensure_cb_repo")
    def test_cb_dispatch_constructs_subprocess_command(
        self,
        mock_ensure: MagicMock,
        mock_cb_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        import subprocess as _sp

        data_dir = tmp_path / "data" / "wmdp"
        _write_data_files(data_dir)
        cb_repo = tmp_path / "circuit-breakers"
        cb_repo.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "model.safetensors").write_text("")

        mock_ensure.return_value = "cafebabe"
        mock_cb_run.return_value = _sp.CompletedProcess(args=[], returncode=0)

        main([
            "--method", "cb",
            "--target", "wmdp",
            "--base-model", "meta-llama/Llama-3.1-8B-Instruct",
            "--output-dir", str(out_dir),
            "--defense-data-dir", str(data_dir),
            "--cb-repo", str(cb_repo),
            "--seed", "0",
        ])

        mock_cb_run.assert_called_once()
        cmd = mock_cb_run.call_args[0][0]
        assert "--model_name_or_path" in cmd
        assert "meta-llama/Llama-3.1-8B-Instruct" in cmd

    @patch("scripts.train_defense.train_tar")
    @patch("scripts.train_defense._ensure_tar_repo")
    def test_tar_dispatch(
        self,
        mock_ensure: MagicMock,
        mock_tar: MagicMock,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "data" / "wmdp"
        _write_data_files(data_dir)
        tar_repo = tmp_path / "tar"
        tar_repo.mkdir()
        out_dir = tmp_path / "out"

        mock_ensure.return_value = "deadbeef"

        main([
            "--method", "tar",
            "--target", "wmdp",
            "--base-model", "meta-llama/Llama-3.1-8B-Instruct",
            "--output-dir", str(out_dir),
            "--defense-data-dir", str(data_dir),
            "--tar-repo", str(tar_repo),
            "--seed", "0",
            "--dry-run",
        ])

        mock_tar.assert_called_once()
        kwargs = mock_tar.call_args.kwargs
        assert kwargs["target"] == "wmdp"
        assert kwargs["base_model"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert kwargs["output_dir"] == out_dir
        assert kwargs["tar_repo"] == tar_repo
        mock_ensure.assert_called_once_with(tar_repo)

    @patch("scripts.train_defense.train_circuit_breakers")
    @patch("scripts.train_defense.train_tar")
    @patch("scripts.train_defense.train_refusal_sft")
    def test_only_one_adapter_called(
        self,
        mock_sft: MagicMock,
        mock_tar: MagicMock,
        mock_cb: MagicMock,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "data" / "wmdp"
        _write_data_files(data_dir)
        out_dir = tmp_path / "out"

        main([
            "--method", "sft",
            "--target", "wmdp",
            "--base-model", "x",
            "--output-dir", str(out_dir),
            "--defense-data-dir", str(data_dir),
            "--seed", "0",
            "--max-steps", "2",
        ])

        mock_sft.assert_called_once()
        mock_cb.assert_not_called()
        mock_tar.assert_not_called()


class TestDefenseMeta:
    @patch("scripts.train_defense.train_refusal_sft")
    def test_meta_contains_lineage(self, mock_sft: MagicMock, tmp_path: Path) -> None:
        data_dir = tmp_path / "data" / "wmdp"
        _write_data_files(data_dir)
        out_dir = tmp_path / "out"

        main([
            "--method", "sft",
            "--target", "wmdp",
            "--base-model", "meta-llama/Llama-3.1-8B-Instruct",
            "--output-dir", str(out_dir),
            "--defense-data-dir", str(data_dir),
            "--seed", "0",
            "--max-steps", "2",
        ])

        meta = json.loads((out_dir / "defense_meta.json").read_text())
        assert meta["method"] == "sft"
        assert meta["target"] == "wmdp"
        assert meta["base_model"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert meta["seed"] == 0
        assert "defense_id" in meta
        assert "refusals_path" in meta
        assert "retain_path" in meta


class TestErrors:
    def test_missing_data_dir_raises(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        with pytest.raises(FileNotFoundError):
            main([
                "--method", "sft",
                "--target", "wmdp",
                "--base-model", "x",
                "--output-dir", str(out_dir),
                "--defense-data-dir", str(tmp_path / "missing"),
                "--seed", "0",
                "--no-build-data-if-missing",
            ])
