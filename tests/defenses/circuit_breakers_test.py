"""Tests for the Circuit Breakers defense adapter."""

from __future__ import annotations

import ast
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

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_CB_DATASET = REPO_ROOT / "circuit-breakers" / "src" / "cb_train_dataset.py"


def _write_refusals(path: Path, n: int) -> None:
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({
                "prompt": f"p_{i}",
                "refusal": "I can't help with that.",
                "source": "x",
                "id": f"x-{i}",
            }) + "\n")


class TestBuildCbFormat:
    def test_writes_only_refusal_rows(self, tmp_path: Path) -> None:
        """CB's loader iterates EVERY row of circuit_breakers_train.json and treats it as a target
        whose representations should be pushed away from base.

        Retain rows must NOT be written here — CB loads its retain set internally from ultrachat +
        xstest. Writing retain rows collapses MMLU (we hit acc=0.17 vs the published ~0.63).
        """
        refusals_path = tmp_path / "r.jsonl"
        out_path = tmp_path / "cb.json"
        _write_refusals(refusals_path, 2)

        n = _build_cb_format(refusals_path, out_path)
        assert n == 2

        data = json.loads(out_path.read_text())
        assert len(data) == 2
        for row in data:
            for key in ("prompt", "output", "llama3_output"):
                assert key in row


class TestTrainCircuitBreakers:
    @patch("information_safety.defenses.circuit_breakers.torch.cuda")
    @patch("information_safety.defenses.circuit_breakers.subprocess.run")
    def test_subprocess_command_contains_required_flags(
        self,
        mock_run: MagicMock,
        mock_cuda: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 2
        refusals_path = tmp_path / "r.jsonl"
        output_dir = tmp_path / "out"
        cb_repo = tmp_path / "circuit-breakers"
        cb_repo.mkdir()
        _write_refusals(refusals_path, 2)
        output_dir.mkdir()
        (output_dir / "config.json").write_text("{}")
        (output_dir / "model.safetensors").write_text("")

        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

        train_circuit_breakers(
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            refusals_path=refusals_path,
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
        _write_refusals(refusals_path, 1)

        train_circuit_breakers(
            base_model="x",
            refusals_path=refusals_path,
            output_dir=tmp_path / "out",
            hparams=CBHParams(),
            cb_repo=tmp_path / "circuit-breakers",
            dry_run=True,
        )

        mock_run.assert_not_called()

    @patch("information_safety.defenses.circuit_breakers.torch.cuda")
    @patch("information_safety.defenses.circuit_breakers.subprocess.run")
    def test_raises_when_subprocess_fails(
        self,
        mock_run: MagicMock,
        mock_cuda: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 2
        refusals_path = tmp_path / "r.jsonl"
        _write_refusals(refusals_path, 1)

        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)

        with pytest.raises(RuntimeError):
            train_circuit_breakers(
                base_model="x",
                refusals_path=refusals_path,
                output_dir=tmp_path / "out",
                hparams=CBHParams(),
                cb_repo=tmp_path / "circuit-breakers",
            )


class TestVendoredCbTrainDataset:
    """Regression guards on circuit-breakers/src/cb_train_dataset.py.

    The vendored CircuitBreakerDataset.__init__ hard-codes a template-selection branch on
    model_name_or_path that raises NotImplementedError for any model that isn't llama-3 or mistral.
    New supported bases need an explicit branch.
    """

    def test_qwen3_branch_uses_chatml_tags(self) -> None:
        """Without a qwen3 branch CB training crashes at dataset construction for Qwen/Qwen3-4B
        before any GPU work happens."""
        tree = ast.parse(VENDORED_CB_DATASET.read_text())
        qwen_branches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If) and _branch_compares_against(node.test, "qwen3")
        ]
        assert len(qwen_branches) == 1, (
            f"Expected exactly one `'qwen3' in model_name_or_path` branch, got {len(qwen_branches)}"
        )
        assignments = {
            target.id: ast.literal_eval(stmt.value)
            for stmt in qwen_branches[0].body
            if isinstance(stmt, ast.Assign)
            for target in stmt.targets
            if isinstance(target, ast.Name)
        }
        assert assignments.get("user_tag") == "<|im_start|>user\n", (
            f"qwen3 branch must set user_tag to ChatML user prefix, got {assignments.get('user_tag')!r}"
        )
        assert assignments.get("assistant_tag") == "<|im_end|>\n<|im_start|>assistant\n", (
            f"qwen3 branch must set assistant_tag to ChatML assistant prefix, "
            f"got {assignments.get('assistant_tag')!r}"
        )
        assert assignments.get("switch_select") == [0, 1], (
            f"qwen3 branch must match llama-3 switch_select=[0, 1], "
            f"got {assignments.get('switch_select')!r}"
        )
        assert assignments.get("use_refusal_retain") is True, (
            f"qwen3 branch must match llama-3 use_refusal_retain=True, "
            f"got {assignments.get('use_refusal_retain')!r}"
        )


def _branch_compares_against(test: ast.expr, literal: str) -> bool:
    """Return True if `test` is a `<literal> in <name>` comparison."""
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.In):
        return False
    left = test.left
    return isinstance(left, ast.Constant) and left.value == literal
