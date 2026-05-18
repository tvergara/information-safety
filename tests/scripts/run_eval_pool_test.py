"""Tests for scripts/run_eval_pool.py — vLLM-backed eval pool worker.

vLLM is mocked; the worker exercises the queue mechanics, prompt rendering plumbing, result-row
schema, and completions writing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scripts import run_eval_pool


def _make_pending_spec(
    tmp_path: Path,
    *,
    spec_id: str = "spec0001",
    epoch: int = 0,
    base_model: str = "tiny-model",
) -> Path:
    adapter_path = tmp_path / "adapters" / spec_id / f"epoch_{epoch}"
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapter_config.json").write_text("{}")
    (adapter_path / "adapter_model.safetensors").write_text("dummy")
    eval_meta: dict[str, Any] = {
        "spec_id": spec_id,
        "epoch": epoch,
        "max_epochs": 2,
        "max_examples": 16,
        "model_name": base_model,
        "dataset_name": "wmdp",
        "program_bits": 1234,
        "independent_flops": 999,
        "strategy_hparams": {"r": 16, "lr": 0.001},
        "train_time_seconds": 12.5,
        "seed": 0,
    }
    pending_dir = tmp_path / "eval-pool" / "pending"
    pending_dir.mkdir(parents=True)
    spec = {
        "spec_id": spec_id,
        "epoch": epoch,
        "base_model": base_model,
        "adapter_path": str(adapter_path),
        "eval_meta": eval_meta,
    }
    path = pending_dir / f"{spec_id}_ep{epoch}.json"
    path.write_text(json.dumps(spec))
    return path


def _fake_prompts() -> tuple[list[str], list[str]]:
    prompts = ["p0", "p1"]
    metas = [
        json.dumps({
            "question": "q0",
            "choices": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "subject": "wmdp-bio",
        }),
        json.dumps({
            "question": "q1",
            "choices": ["A", "B", "C", "D"],
            "correct_answer": "C",
            "subject": "wmdp-bio",
        }),
    ]
    return prompts, metas


@pytest.fixture
def mock_vllm(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Patch vLLM imports inside run_eval_pool with mocks."""
    fake_llm = MagicMock(name="LLM-instance")

    def fake_generate(prompts: list[str], **_kw: Any) -> list[Any]:
        out = []
        canned = ["A", "C"]
        for letter in canned[: len(prompts)]:
            o = MagicMock()
            o.outputs = [MagicMock(text=f"final {letter}")]
            out.append(o)
        return out

    fake_llm.generate.side_effect = fake_generate
    llm_cls = MagicMock(return_value=fake_llm)
    sampling_cls = MagicMock()
    lora_cls = MagicMock()
    monkeypatch.setattr(run_eval_pool, "LLM", llm_cls, raising=False)
    monkeypatch.setattr(run_eval_pool, "SamplingParams", sampling_cls, raising=False)
    monkeypatch.setattr(run_eval_pool, "LoRARequest", lora_cls, raising=False)
    return {"LLM": llm_cls, "instance": fake_llm, "SamplingParams": sampling_cls,
            "LoRARequest": lora_cls}


@pytest.fixture
def mock_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts, metas = _fake_prompts()
    monkeypatch.setattr(
        run_eval_pool, "load_wmdp_val_prompts",
        lambda *a, **kw: (prompts, metas),
    )


class TestClaimAndEnsureDirs:
    def test_ensure_eval_queue_dirs_creates_subdirs(self, tmp_path: Path) -> None:
        root = tmp_path / "queue"
        run_eval_pool.ensure_eval_queue_dirs(root)
        for sub in ("pending", "claimed", "done", "failed", "logs"):
            assert (root / sub).is_dir()

    def test_try_claim_moves_pending_to_claimed(self, tmp_path: Path) -> None:
        root = tmp_path / "queue"
        run_eval_pool.ensure_eval_queue_dirs(root)
        pending = root / "pending" / "x.json"
        pending.write_text("{}")
        claimed = run_eval_pool.try_claim(pending, root / "claimed", pool_id="p")
        assert claimed is not None and claimed.exists()
        assert not pending.exists()


class TestWorkerEvaluatesSpec:
    def test_writes_result_row_with_expected_schema(
        self, tmp_path: Path, mock_vllm: dict[str, MagicMock], mock_prompts: None,
    ) -> None:
        pending = _make_pending_spec(tmp_path)
        queue_root = tmp_path / "eval-pool"
        results_file = tmp_path / "results" / "final-results.jsonl"
        generations_dir = tmp_path / "generations"

        run_eval_pool.run_worker(
            queue_root=queue_root,
            base_model="tiny-model",
            adapter_root=tmp_path / "adapters",
            results_file=results_file,
            generations_dir=generations_dir,
            keep_adapters=True,
        )

        assert results_file.exists()
        rows = [json.loads(line) for line in results_file.read_text().splitlines()]
        assert len(rows) == 1
        row = rows[0]
        assert row["experiment_name"] == "DataStrategyDeferredEval"
        assert row["spec_id"] == "spec0001"
        assert row["epoch"] == 0
        assert row["max_epochs"] == 2
        assert row["max_examples"] == 16
        assert row["model_name"] == "tiny-model"
        assert row["dataset_name"] == "wmdp"
        assert row["program_bits"] == 1234
        assert row["independent_flops"] == 999
        assert row["asr"] is None
        assert isinstance(row["eval_run_id"], str) and row["eval_run_id"]
        assert "performance" in row
        assert 0.0 <= row["performance"] <= 1.0
        # canned answers were "A" and "C"; both meta correct answers were "A" and "C"
        assert row["performance"] == 1.0

        # done/ now contains the spec, pending/ is empty
        assert not list((queue_root / "pending").glob("*.json"))
        assert (queue_root / "done" / "spec0001_ep0.json").exists()
        assert not pending.exists()

    def test_writes_completions_to_generations_dir(
        self, tmp_path: Path, mock_vllm: dict[str, MagicMock], mock_prompts: None,
    ) -> None:
        _make_pending_spec(tmp_path)
        queue_root = tmp_path / "eval-pool"
        results_file = tmp_path / "results" / "final-results.jsonl"
        generations_dir = tmp_path / "generations"

        run_eval_pool.run_worker(
            queue_root=queue_root,
            base_model="tiny-model",
            adapter_root=tmp_path / "adapters",
            results_file=results_file,
            generations_dir=generations_dir,
            keep_adapters=True,
        )

        row = json.loads(results_file.read_text().splitlines()[0])
        eval_run_id = row["eval_run_id"]
        run_dir = generations_dir / eval_run_id
        assert (run_dir / "input_data.jsonl").exists()
        assert (run_dir / "responses.jsonl").exists()
        responses = [
            json.loads(line)
            for line in (run_dir / "responses.jsonl").read_text().splitlines()
        ]
        assert len(responses) == 2
        for r in responses:
            for key in ("question", "response", "predicted_answer", "correct"):
                assert key in r

    def test_adapter_deleted_after_success_when_not_keep(
        self, tmp_path: Path, mock_vllm: dict[str, MagicMock], mock_prompts: None,
    ) -> None:
        _make_pending_spec(tmp_path)
        queue_root = tmp_path / "eval-pool"
        results_file = tmp_path / "results" / "final-results.jsonl"
        generations_dir = tmp_path / "generations"

        run_eval_pool.run_worker(
            queue_root=queue_root,
            base_model="tiny-model",
            adapter_root=tmp_path / "adapters",
            results_file=results_file,
            generations_dir=generations_dir,
            keep_adapters=False,
        )
        adapter_dir = tmp_path / "adapters" / "spec0001" / "epoch_0"
        assert not adapter_dir.exists()

    def test_keep_adapters_preserves_dir(
        self, tmp_path: Path, mock_vllm: dict[str, MagicMock], mock_prompts: None,
    ) -> None:
        _make_pending_spec(tmp_path)
        queue_root = tmp_path / "eval-pool"
        results_file = tmp_path / "results" / "final-results.jsonl"
        generations_dir = tmp_path / "generations"

        run_eval_pool.run_worker(
            queue_root=queue_root,
            base_model="tiny-model",
            adapter_root=tmp_path / "adapters",
            results_file=results_file,
            generations_dir=generations_dir,
            keep_adapters=True,
        )
        adapter_dir = tmp_path / "adapters" / "spec0001" / "epoch_0"
        assert adapter_dir.exists()


class TestWorkerFailurePath:
    def test_vllm_exception_moves_spec_to_failed(
        self, tmp_path: Path, mock_prompts: None, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _make_pending_spec(tmp_path)
        queue_root = tmp_path / "eval-pool"
        results_file = tmp_path / "results" / "final-results.jsonl"
        generations_dir = tmp_path / "generations"

        fake_llm = MagicMock()
        fake_llm.generate.side_effect = RuntimeError("vllm boom")
        monkeypatch.setattr(run_eval_pool, "LLM", MagicMock(return_value=fake_llm), raising=False)
        monkeypatch.setattr(run_eval_pool, "SamplingParams", MagicMock(), raising=False)
        monkeypatch.setattr(run_eval_pool, "LoRARequest", MagicMock(), raising=False)

        with pytest.raises(RuntimeError, match="vllm boom"):
            run_eval_pool.run_worker(
                queue_root=queue_root,
                base_model="tiny-model",
                adapter_root=tmp_path / "adapters",
                results_file=results_file,
                generations_dir=generations_dir,
                keep_adapters=True,
            )

        assert (queue_root / "failed" / "spec0001_ep0.json").exists()
        assert not (queue_root / "done" / "spec0001_ep0.json").exists()


class TestParseAnswerLetterIntegration:
    def test_parse_used_for_correctness(self) -> None:
        from information_safety.algorithms.dataset_handlers.wmdp import parse_answer_letter
        assert parse_answer_letter("final A") == "A"
        assert parse_answer_letter("B is the answer") == "B"


class TestRunWorkerNoPending:
    def test_returns_immediately(self, tmp_path: Path) -> None:
        queue_root = tmp_path / "eval-pool"
        run_eval_pool.ensure_eval_queue_dirs(queue_root)
        with patch.object(run_eval_pool, "LLM", MagicMock()) as llm_cls:
            run_eval_pool.run_worker(
                queue_root=queue_root,
                base_model="tiny",
                adapter_root=tmp_path / "adapters",
                results_file=tmp_path / "r.jsonl",
                generations_dir=tmp_path / "g",
                keep_adapters=True,
            )
        llm_cls.assert_not_called()
