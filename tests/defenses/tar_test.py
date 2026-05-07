"""Tests for the TAR (Tamper-Resistant Safeguards) defense adapter."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from information_safety.defenses.tar import TARHParams, train_tar


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "tar"
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / "accel_config_2_gpu.yaml").write_text("")
    (repo / "tar.py").write_text("")
    return repo


@pytest.fixture
def fake_evilmath_data(tmp_path: Path) -> Path:
    path = tmp_path / "defense-data" / "evilmath" / "llm_rewrite.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("")
    return path


def _output_with_weights(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    (out / "config.json").write_text("{}")
    (out / "model.safetensors").write_text("")
    return out


class TestTrainTarSubprocessCommand:
    @patch("information_safety.defenses.tar.AutoTokenizer")
    @patch("information_safety.defenses.tar.subprocess.run")
    @patch("information_safety.defenses.tar.torch.cuda")
    def test_wmdp_uses_subject_bio(
        self,
        mock_cuda: MagicMock,
        mock_run: MagicMock,
        mock_tokenizer_cls: MagicMock,
        fake_repo: Path,
        tmp_path: Path,
    ) -> None:
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 2
        output_dir = _output_with_weights(tmp_path)
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

        train_tar(
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            output_dir=output_dir,
            target="wmdp",
            hparams=TARHParams(),
            tar_repo=fake_repo,
        )

        cmd = mock_run.call_args[0][0]
        assert "accelerate" in cmd
        assert "launch" in cmd
        assert str(fake_repo / "tar.py") in cmd
        assert cmd[cmd.index("--subject") + 1] == "bio"
        assert (
            cmd[cmd.index("--base_model_name") + 1]
            == "meta-llama/Llama-3.1-8B-Instruct"
        )
        assert cmd[cmd.index("--output_dir") + 1] == str(output_dir)
        assert "--evilmath_data_path" not in cmd

    @patch("information_safety.defenses.tar.AutoTokenizer")
    @patch("information_safety.defenses.tar.subprocess.run")
    @patch("information_safety.defenses.tar.torch.cuda")
    def test_evilmath_uses_subject_evilmath_with_data_path(
        self,
        mock_cuda: MagicMock,
        mock_run: MagicMock,
        mock_tokenizer_cls: MagicMock,
        fake_repo: Path,
        fake_evilmath_data: Path,
        tmp_path: Path,
    ) -> None:
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 2
        output_dir = _output_with_weights(tmp_path)
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

        train_tar(
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            output_dir=output_dir,
            target="evilmath",
            hparams=TARHParams(),
            tar_repo=fake_repo,
            evilmath_data_path=fake_evilmath_data,
        )

        cmd = mock_run.call_args[0][0]
        assert cmd[cmd.index("--subject") + 1] == "evilmath"
        assert cmd[cmd.index("--evilmath_data_path") + 1] == str(fake_evilmath_data)

    @patch("information_safety.defenses.tar.subprocess.run")
    @patch("information_safety.defenses.tar.torch.cuda")
    def test_dry_run_does_not_invoke_subprocess(
        self,
        mock_cuda: MagicMock,
        mock_run: MagicMock,
        fake_repo: Path,
        tmp_path: Path,
    ) -> None:
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 2
        train_tar(
            base_model="x",
            output_dir=tmp_path / "out",
            target="wmdp",
            hparams=TARHParams(),
            tar_repo=fake_repo,
            dry_run=True,
        )
        mock_run.assert_not_called()

    @patch("information_safety.defenses.tar.subprocess.run")
    @patch("information_safety.defenses.tar.torch.cuda")
    def test_failed_subprocess_raises(
        self,
        mock_cuda: MagicMock,
        mock_run: MagicMock,
        fake_repo: Path,
        tmp_path: Path,
    ) -> None:
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 2
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=2)

        with pytest.raises(RuntimeError):
            train_tar(
                base_model="x",
                output_dir=tmp_path / "out",
                target="wmdp",
                hparams=TARHParams(),
                tar_repo=fake_repo,
            )

    def test_unknown_target_raises(self, fake_repo: Path, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            train_tar(
                base_model="x",
                output_dir=tmp_path / "out",
                target="weird",
                hparams=TARHParams(),
                tar_repo=fake_repo,
                dry_run=True,
            )

    def test_evilmath_without_data_path_raises(
        self, fake_repo: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="evilmath_data_path"):
            train_tar(
                base_model="x",
                output_dir=tmp_path / "out",
                target="evilmath",
                hparams=TARHParams(),
                tar_repo=fake_repo,
                dry_run=True,
            )

    def test_evilmath_with_missing_data_path_raises(
        self, fake_repo: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            train_tar(
                base_model="x",
                output_dir=tmp_path / "out",
                target="evilmath",
                hparams=TARHParams(),
                tar_repo=fake_repo,
                evilmath_data_path=tmp_path / "does-not-exist.jsonl",
                dry_run=True,
            )

    @patch("information_safety.defenses.tar.AutoTokenizer")
    @patch("information_safety.defenses.tar.subprocess.run")
    @patch("information_safety.defenses.tar.torch.cuda")
    def test_saves_tokenizer_to_output_dir(
        self,
        mock_cuda: MagicMock,
        mock_run: MagicMock,
        mock_tokenizer_cls: MagicMock,
        fake_repo: Path,
        tmp_path: Path,
    ) -> None:
        """TAR's vendored tar.py only saves the model, not the tokenizer.

        Without tokenizer.json / tokenizer_config.json next to the weights, vLLM crashes with
        'Couldn't instantiate the backend tokenizer'. The adapter must save the tokenizer after the
        subprocess returns.
        """
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 2
        output_dir = _output_with_weights(tmp_path)
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        train_tar(
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            output_dir=output_dir,
            target="wmdp",
            hparams=TARHParams(),
            tar_repo=fake_repo,
        )

        mock_tokenizer_cls.from_pretrained.assert_called_once_with(
            "meta-llama/Llama-3.1-8B-Instruct"
        )
        mock_tokenizer.save_pretrained.assert_called_once_with(str(output_dir))

    @patch("information_safety.defenses.tar.AutoTokenizer")
    @patch("information_safety.defenses.tar.subprocess.run")
    @patch("information_safety.defenses.tar.torch.cuda")
    def test_passes_hparam_overrides(
        self,
        mock_cuda: MagicMock,
        mock_run: MagicMock,
        mock_tokenizer_cls: MagicMock,
        fake_repo: Path,
        tmp_path: Path,
    ) -> None:
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 2
        output_dir = _output_with_weights(tmp_path)
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        hparams = TARHParams(
            max_steps=11,
            tar_inner_loop_steps=7,
            batch_size=3,
            gradient_accumulation_steps=5,
            lr=1.5e-5,
        )

        train_tar(
            base_model="x",
            output_dir=output_dir,
            target="wmdp",
            hparams=hparams,
            tar_repo=fake_repo,
        )

        cmd = mock_run.call_args[0][0]
        assert cmd[cmd.index("--max_steps") + 1] == "11"
        assert cmd[cmd.index("--tar_inner_loop_steps") + 1] == "7"
        assert cmd[cmd.index("--batch_size") + 1] == "3"
        assert cmd[cmd.index("--gradient_accumulation_steps") + 1] == "5"
        assert cmd[cmd.index("--lr") + 1] == "1.5e-05"
