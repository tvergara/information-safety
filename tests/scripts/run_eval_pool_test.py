"""Tests for scripts/run_eval_pool.py — vLLM-backed eval pool worker.

vLLM is mocked; the worker exercises the queue mechanics, prompt rendering plumbing, result-row
schema, and completions writing.
"""

from __future__ import annotations

import json
import os
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
    dataset_name: str = "wmdp",
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
        "dataset_name": dataset_name,
        "program_bits": 1234,
        "independent_flops": 999,
        "strategy_hparams": {"r": 16, "lr": 0.001},
        "train_time_seconds": 12.5,
        "seed": 0,
    }
    pending_dir = tmp_path / "eval-pool" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
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


def _fake_wmdp_prompts() -> tuple[list[list[int]], list[str]]:
    prompts = [[1, 2, 3], [4, 5, 6]]
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


def _fake_evilmath_prompts() -> tuple[list[list[int]], list[str]]:
    prompts = [[10, 20, 30], [40, 50, 60]]
    metas = [
        json.dumps({
            "question": "q0",
            "correct_answer": 42,
            "original_question": "orig0",
        }),
        json.dumps({
            "question": "q1",
            "correct_answer": 7,
            "original_question": "orig1",
        }),
    ]
    return prompts, metas


@pytest.fixture
def mock_vllm(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Patch vLLM imports inside run_eval_pool with mocks."""
    fake_llm = MagicMock(name="LLM-instance")

    def fake_generate(prompts: list[Any], **_kw: Any) -> list[Any]:
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
    tokens_prompt_cls = MagicMock(
        side_effect=lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids},
    )
    monkeypatch.setattr(
        run_eval_pool, "_load_vllm",
        lambda: (llm_cls, sampling_cls, lora_cls, tokens_prompt_cls),
    )
    return {"LLM": llm_cls, "instance": fake_llm, "SamplingParams": sampling_cls,
            "LoRARequest": lora_cls, "TokensPrompt": tokens_prompt_cls}


@pytest.fixture
def mock_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    wmdp_prompts, wmdp_metas = _fake_wmdp_prompts()
    evilmath_prompts, evilmath_metas = _fake_evilmath_prompts()

    def fake_load(
        base_model: str,
        dataset_name: str,
        *,
        max_examples: int | None = None,
    ) -> tuple[list[list[int]], list[str]]:
        if dataset_name == "wmdp":
            prompts, metas = wmdp_prompts, wmdp_metas
        elif dataset_name == "evilmath":
            prompts, metas = evilmath_prompts, evilmath_metas
        else:
            raise ValueError(f"unknown dataset {dataset_name}")
        if max_examples is not None:
            prompts = prompts[:max_examples]
            metas = metas[:max_examples]
        return prompts, metas

    monkeypatch.setattr(run_eval_pool, "load_val_prompts", fake_load)


class TestClaimAndEnsureDirs:
    def test_ensure_eval_queue_dirs_creates_subdirs(self, tmp_path: Path) -> None:
        root = tmp_path / "queue"
        run_eval_pool.ensure_eval_queue_dirs(root)
        for sub in ("pending", "claimed", "done", "failed", "logs"):
            assert (root / sub).is_dir()

    def test_try_claim_filter_match_claims(self, tmp_path: Path) -> None:
        root = tmp_path / "queue"
        run_eval_pool.ensure_eval_queue_dirs(root)
        pending = root / "pending" / "s.json"
        pending.write_text(json.dumps({"base_model": "model-a"}))
        claimed = run_eval_pool.try_claim(
            pending, root / "claimed", pool_id="p", base_model_filter="model-a",
        )
        assert claimed is not None and claimed.exists()
        assert not pending.exists()

    def test_try_claim_filter_mismatch_skips(self, tmp_path: Path) -> None:
        root = tmp_path / "queue"
        run_eval_pool.ensure_eval_queue_dirs(root)
        pending = root / "pending" / "s.json"
        pending.write_text(json.dumps({"base_model": "model-b"}))
        claimed = run_eval_pool.try_claim(
            pending, root / "claimed", pool_id="p", base_model_filter="model-a",
        )
        assert claimed is None
        assert pending.exists()
        assert not list((root / "claimed").glob("*.json"))

    def test_try_claim_file_vanishes_between_peek_and_rename(
        self, tmp_path: Path,
    ) -> None:
        root = tmp_path / "queue"
        run_eval_pool.ensure_eval_queue_dirs(root)
        pending = root / "pending" / "ghost.json"
        claimed = run_eval_pool.try_claim(
            pending, root / "claimed", pool_id="p", base_model_filter="model-a",
        )
        assert claimed is None

    def test_sort_pending_by_mtime_skips_vanished_files(self, tmp_path: Path) -> None:
        pending_dir = tmp_path / "pending"
        pending_dir.mkdir()
        present = pending_dir / "alive.json"
        present.write_text("{}")
        ghost = pending_dir / "ghost.json"
        ordered = run_eval_pool._sort_pending_by_mtime([present, ghost])
        assert present in ordered
        assert ghost in ordered  # included but sorted last (mtime=inf)
        assert ordered[-1] == ghost


class TestLoadValPrompts:
    def test_returns_token_ids_for_wmdp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_dataset = [
            {
                "input_ids": [1, 2, 3, 4],
                "labels": json.dumps({
                    "question": "q",
                    "choices": ["A", "B", "C", "D"],
                    "correct_answer": "A",
                    "subject": "wmdp-bio",
                }),
            },
            {
                "input_ids": [5, 6, 7],
                "labels": json.dumps({
                    "question": "q2",
                    "choices": ["A", "B", "C", "D"],
                    "correct_answer": "B",
                    "subject": "wmdp-chem",
                }),
            },
        ]
        fake_handler = MagicMock()
        fake_handler.get_val_dataset.return_value = fake_dataset
        fake_tokenizer = MagicMock()
        fake_tokenizer.pad_token = None
        fake_tokenizer.eos_token = "</s>"
        monkeypatch.setitem(
            run_eval_pool._HANDLERS, "wmdp", MagicMock(return_value=fake_handler),
        )
        monkeypatch.setattr(
            run_eval_pool, "AutoTokenizer",
            MagicMock(from_pretrained=MagicMock(return_value=fake_tokenizer)),
        )

        prompt_token_ids, label_jsons = run_eval_pool.load_val_prompts(
            "tiny-model", "wmdp",
        )

        assert isinstance(prompt_token_ids, list)
        assert len(prompt_token_ids) == 2
        for ids in prompt_token_ids:
            assert isinstance(ids, list)
            assert all(isinstance(t, int) for t in ids)
        assert len(label_jsons) == 2
        meta = json.loads(label_jsons[0])
        assert "correct_answer" in meta

    def test_returns_token_ids_for_evilmath(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_dataset = [
            {
                "input_ids": [11, 12, 13],
                "labels": json.dumps({
                    "question": "evil0",
                    "correct_answer": 99,
                    "original_question": "orig0",
                }),
            },
        ]
        fake_handler = MagicMock()
        fake_handler.get_val_dataset.return_value = fake_dataset
        fake_tokenizer = MagicMock()
        fake_tokenizer.pad_token = None
        fake_tokenizer.eos_token = "</s>"
        monkeypatch.setitem(
            run_eval_pool._HANDLERS, "evilmath", MagicMock(return_value=fake_handler),
        )
        monkeypatch.setattr(
            run_eval_pool, "AutoTokenizer",
            MagicMock(from_pretrained=MagicMock(return_value=fake_tokenizer)),
        )

        prompt_token_ids, label_jsons = run_eval_pool.load_val_prompts(
            "tiny-model", "evilmath",
        )

        assert isinstance(prompt_token_ids, list)
        assert len(prompt_token_ids) == 1
        assert all(isinstance(t, int) for t in prompt_token_ids[0])
        meta = json.loads(label_jsons[0])
        assert "correct_answer" in meta

    def test_raises_on_unknown_dataset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            run_eval_pool, "AutoTokenizer",
            MagicMock(from_pretrained=MagicMock(return_value=MagicMock(pad_token="x"))),
        )
        with pytest.raises(KeyError):
            run_eval_pool.load_val_prompts("tiny-model", "not-a-dataset")


class TestEvalOneParserDispatch:
    def test_uses_letter_parser_for_wmdp(self) -> None:
        prompts, metas = _fake_wmdp_prompts()
        fake_llm = MagicMock()

        def gen(prompts_arg: list[Any], **_kw: Any) -> list[Any]:
            texts = ["final A", "final B"]
            return [
                MagicMock(outputs=[MagicMock(text=t)])
                for t in texts[: len(prompts_arg)]
            ]

        fake_llm.generate.side_effect = gen
        tokens_prompt_cls = MagicMock(
            side_effect=lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids},
        )
        spec = {
            "spec_id": "s0",
            "epoch": 0,
            "adapter_path": "/tmp/adapter",
        }
        performance, _resps, predicted, correct, _metas = run_eval_pool._eval_one(
            fake_llm, spec,
            prompt_token_ids=prompts,
            label_jsons=metas,
            sampling_params=MagicMock(),
            lora_request_cls=MagicMock(),
            tokens_prompt_cls=tokens_prompt_cls,
            lora_int_id=1,
            dataset_name="wmdp",
        )
        assert predicted == ["A", "B"]
        assert correct == [True, False]
        assert performance == 0.5

    def test_uses_number_parser_for_evilmath(self) -> None:
        prompts, metas = _fake_evilmath_prompts()
        fake_llm = MagicMock()

        def gen(prompts_arg: list[Any], **_kw: Any) -> list[Any]:
            texts = ["The answer is 42", "The answer is 13"]
            return [
                MagicMock(outputs=[MagicMock(text=t)])
                for t in texts[: len(prompts_arg)]
            ]

        fake_llm.generate.side_effect = gen
        tokens_prompt_cls = MagicMock(
            side_effect=lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids},
        )
        spec = {
            "spec_id": "s0",
            "epoch": 0,
            "adapter_path": "/tmp/adapter",
        }
        performance, _resps, predicted, correct, _metas = run_eval_pool._eval_one(
            fake_llm, spec,
            prompt_token_ids=prompts,
            label_jsons=metas,
            sampling_params=MagicMock(),
            lora_request_cls=MagicMock(),
            tokens_prompt_cls=tokens_prompt_cls,
            lora_int_id=1,
            dataset_name="evilmath",
        )
        assert predicted == ["42", "13"]
        assert correct == [True, False]
        assert performance == 0.5

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

    def test_dispatches_by_dataset_name(
        self, tmp_path: Path, mock_vllm: dict[str, MagicMock], mock_prompts: None,
    ) -> None:
        _make_pending_spec(
            tmp_path, spec_id="wmdp_spec", epoch=0, dataset_name="wmdp",
        )
        _make_pending_spec(
            tmp_path, spec_id="evilmath_spec", epoch=0, dataset_name="evilmath",
        )
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

        rows = [json.loads(line) for line in results_file.read_text().splitlines()]
        assert len(rows) == 2
        by_spec = {r["spec_id"]: r for r in rows}
        assert by_spec["wmdp_spec"]["dataset_name"] == "wmdp"
        assert by_spec["evilmath_spec"]["dataset_name"] == "evilmath"

        wmdp_prompts, _ = _fake_wmdp_prompts()
        evilmath_prompts, _ = _fake_evilmath_prompts()
        generate_calls = mock_vllm["instance"].generate.call_args_list
        seen_token_id_sets: set[tuple[int, ...]] = set()
        for call in generate_calls:
            args, kwargs = call
            passed_prompts = args[0] if args else kwargs.get("prompts")
            assert isinstance(passed_prompts, list)
            for p in passed_prompts:
                ids = tuple(p["prompt_token_ids"])
                seen_token_id_sets.add(ids)
        wmdp_sets = {tuple(ids) for ids in wmdp_prompts}
        evilmath_sets = {tuple(ids) for ids in evilmath_prompts}
        assert wmdp_sets.issubset(seen_token_id_sets)
        assert evilmath_sets.issubset(seen_token_id_sets)

    def test_passes_token_ids_to_vllm_not_strings(
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
        tokens_prompt_cls = mock_vllm["TokensPrompt"]
        assert tokens_prompt_cls.call_count == 2
        for call in tokens_prompt_cls.call_args_list:
            assert "prompt_token_ids" in call.kwargs
            assert isinstance(call.kwargs["prompt_token_ids"], list)
            assert all(isinstance(t, int) for t in call.kwargs["prompt_token_ids"])

        generate_call = mock_vllm["instance"].generate.call_args
        args, kwargs = generate_call
        passed_prompts = args[0] if args else kwargs.get("prompts")
        assert isinstance(passed_prompts, list)
        assert all(not isinstance(p, str) for p in passed_prompts)


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
        tokens_prompt_cls = MagicMock(
            side_effect=lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids},
        )
        monkeypatch.setattr(
            run_eval_pool, "_load_vllm",
            lambda: (MagicMock(return_value=fake_llm), MagicMock(), MagicMock(),
                     tokens_prompt_cls),
        )

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
        assert parse_answer_letter("Answer: B") == "B"


class TestVLLMMxfp4Workaround:
    """Gpt-oss-20b ships PTX that TRC's CUDA 12.4 driver can't JIT in the MARLIN MXFP4 path.

    Why: TRC H100 nodes run driver 550.127.05 (CUDA 12.4); vLLM 0.19.1's Marlin FP4 MoE
    kernel was compiled with a newer toolchain. Setting VLLM_MXFP4_USE_MARLIN=0 routes the
    LoRA-enabled engine to the Triton MoE backend on sm_90 (verified by probe job).
    """

    def test_configure_vllm_env_for_gpt_oss_sets_marlin_off(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("VLLM_MXFP4_USE_MARLIN", raising=False)
        run_eval_pool._configure_vllm_env("openai/gpt-oss-20b")
        assert os.environ["VLLM_MXFP4_USE_MARLIN"] == "0"

    def test_configure_vllm_env_leaves_non_mxfp4_models_alone(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("VLLM_MXFP4_USE_MARLIN", raising=False)
        run_eval_pool._configure_vllm_env("meta-llama/Meta-Llama-3-8B-Instruct")
        assert "VLLM_MXFP4_USE_MARLIN" not in os.environ

    def test_run_worker_invokes_configure_for_gpt_oss(
        self, tmp_path: Path, mock_vllm: dict[str, MagicMock], mock_prompts: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("VLLM_MXFP4_USE_MARLIN", raising=False)
        _make_pending_spec(tmp_path, base_model="openai/gpt-oss-20b")
        queue_root = tmp_path / "eval-pool"
        run_eval_pool.run_worker(
            queue_root=queue_root,
            base_model="openai/gpt-oss-20b",
            adapter_root=tmp_path / "adapters",
            results_file=tmp_path / "results" / "final-results.jsonl",
            generations_dir=tmp_path / "generations",
            keep_adapters=True,
        )
        assert os.environ["VLLM_MXFP4_USE_MARLIN"] == "0"


class TestRunWorkerNoPending:
    def test_returns_immediately(self, tmp_path: Path) -> None:
        queue_root = tmp_path / "eval-pool"
        run_eval_pool.ensure_eval_queue_dirs(queue_root)
        with patch.object(run_eval_pool, "_load_vllm") as load_vllm:
            run_eval_pool.run_worker(
                queue_root=queue_root,
                base_model="tiny",
                adapter_root=tmp_path / "adapters",
                results_file=tmp_path / "r.jsonl",
                generations_dir=tmp_path / "g",
                keep_adapters=True,
            )
        load_vllm.assert_not_called()


class TestRunWorkerModelFilter:
    """Worker scoped to its base_model; pending specs for other models are left alone."""

    def test_only_claims_matching_specs(
        self, tmp_path: Path, mock_vllm: dict[str, MagicMock], mock_prompts: None,
    ) -> None:
        _make_pending_spec(
            tmp_path, spec_id="speca", epoch=0, base_model="tiny-model",
        )
        _make_pending_spec(
            tmp_path, spec_id="specb", epoch=0, base_model="other-model",
        )
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

        assert (queue_root / "done" / "speca_ep0.json").exists()
        assert (queue_root / "pending" / "specb_ep0.json").exists()
        assert not (queue_root / "done" / "specb_ep0.json").exists()
        rows = [json.loads(line) for line in results_file.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["spec_id"] == "speca"

    def test_no_matching_specs_skips_vllm_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _make_pending_spec(
            tmp_path, spec_id="specb", epoch=0, base_model="other-model",
        )
        queue_root = tmp_path / "eval-pool"
        with patch.object(run_eval_pool, "_load_vllm") as load_vllm:
            run_eval_pool.run_worker(
                queue_root=queue_root,
                base_model="tiny-model",
                adapter_root=tmp_path / "adapters",
                results_file=tmp_path / "r.jsonl",
                generations_dir=tmp_path / "g",
                keep_adapters=True,
            )
        load_vllm.assert_not_called()
        assert (queue_root / "pending" / "specb_ep0.json").exists()
