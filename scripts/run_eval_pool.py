"""vLLM-backed eval pool worker: drains pending eval specs and writes one
result row per spec with experiment_name="DataStrategyDeferredEval"."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from information_safety.algorithms.dataset_handlers.base import BaseDatasetHandler
from information_safety.algorithms.dataset_handlers.evilmath import (
    EvilMathHandler,
    extract_numerical_answer,
)
from information_safety.algorithms.dataset_handlers.wmdp import (
    WMDPHandler,
    parse_answer_letter,
)

_MXFP4_BASE_MODELS = frozenset({"openai/gpt-oss-20b"})


def _configure_vllm_env(base_model: str) -> None:
    """Route mxfp4 LoRA loads through Triton instead of MARLIN.

    TRC H100 nodes run CUDA driver 550 (12.4); vLLM 0.19.1's Marlin FP4 MoE kernel
    ships PTX from a newer toolchain that the driver cannot JIT, crashing during
    `prepare_moe_mxfp4_layer_for_marlin`. Triton on sm_90 has no such constraint.
    """
    if base_model in _MXFP4_BASE_MODELS:
        os.environ["VLLM_MXFP4_USE_MARLIN"] = "0"


def _load_vllm() -> tuple[Any, Any, Any, Any]:
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.lora.request import LoRARequest
    return LLM, SamplingParams, LoRARequest, TokensPrompt


def ensure_eval_queue_dirs(queue_root: Path) -> None:
    for sub in ("pending", "claimed", "done", "failed", "logs"):
        (queue_root / sub).mkdir(parents=True, exist_ok=True)


def _spec_base_model(path: Path) -> str | None:
    try:
        return json.loads(path.read_text()).get("base_model")
    except FileNotFoundError:
        return None


def try_claim(
    pending_path: Path,
    claimed_dir: Path,
    *,
    pool_id: str,
    base_model_filter: str,
) -> Path | None:
    if _spec_base_model(pending_path) != base_model_filter:
        return None
    job_id = pending_path.stem
    target = claimed_dir / f"{job_id}.{pool_id}.json"
    try:
        os.rename(pending_path, target)
    except FileNotFoundError:
        return None
    return target


_HANDLERS: dict[str, Callable[..., BaseDatasetHandler]] = {
    "wmdp": WMDPHandler,
    "evilmath": EvilMathHandler,
}


def load_val_prompts(
    base_model: str,
    dataset_name: str,
    *,
    max_length: int = 500,
) -> tuple[list[list[int]], list[str]]:
    """Return ``(prompt_token_ids, label_jsons)`` for the dataset's val split."""
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    handler = _HANDLERS[dataset_name](max_length=max_length, generations_dir="")
    dataset = handler.get_val_dataset(tokenizer)
    prompt_token_ids: list[list[int]] = []
    labels: list[str] = []
    for item in dataset:
        prompt_token_ids.append(list(item["input_ids"]))
        labels.append(item["labels"])
    return prompt_token_ids, labels


def _write_result_row(
    results_file: Path,
    *,
    eval_run_id: str,
    spec: dict[str, Any],
    performance: float,
) -> None:
    meta = spec["eval_meta"]
    row: dict[str, Any] = {
        "experiment_name": "DataStrategyDeferredEval",
        "experiment_id": None,
        "eval_run_id": eval_run_id,
        "spec_id": spec["spec_id"],
        "model_name": meta["model_name"],
        "dataset_name": meta["dataset_name"],
        "max_examples": meta["max_examples"],
        "max_epochs": meta["max_epochs"],
        "performance": performance,
        "asr": None,
        "program_bits": meta["program_bits"],
        "independent_flops": meta["independent_flops"],
        "seed": meta["seed"],
        "epoch": meta["epoch"],
        "strategy_hparams": meta["strategy_hparams"],
        "avg_time_per_example": meta["train_time_seconds"],
    }
    for k in ("corpus_fraction", "corpus_subset"):
        if k in meta:
            row[k] = meta[k]

    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "a") as f:
        f.write(json.dumps(row) + "\n")


def _write_completions(
    generations_dir: Path,
    eval_run_id: str,
    *,
    dataset_name: str,
    metas: list[dict[str, Any]],
    responses: list[str],
    predicted: list[str | None],
    correct: list[bool],
) -> None:
    run_dir = generations_dir / eval_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "input_data.jsonl", "w") as f:
        for meta in metas:
            row: dict[str, Any] = {"question": meta["question"]}
            if dataset_name == "wmdp":
                row["choices"] = meta["choices"]
                row["correct_answer"] = meta["correct_answer"]
                row["subject"] = meta["subject"]
            else:
                row["correct_answer"] = meta["correct_answer"]
                row["original_question"] = meta["original_question"]
            f.write(json.dumps(row) + "\n")
    with open(run_dir / "responses.jsonl", "w") as f:
        for meta, resp, pred, ok in zip(metas, responses, predicted, correct):
            f.write(json.dumps({
                "question": meta["question"],
                "response": resp,
                "predicted_answer": pred,
                "correct": ok,
            }) + "\n")


def _predict_and_score(
    dataset_name: str, responses: list[str], metas: list[dict[str, Any]],
) -> tuple[list[str | None], list[bool]]:
    if dataset_name == "wmdp":
        predicted: list[str | None] = [parse_answer_letter(r) for r in responses]
        correct = [p == m["correct_answer"] for p, m in zip(predicted, metas)]
        return predicted, correct
    numeric = [extract_numerical_answer(r) for r in responses]
    predicted = [str(n) if n is not None else None for n in numeric]
    correct = [n is not None and n == m["correct_answer"]
               for n, m in zip(numeric, metas)]
    return predicted, correct


def _eval_one(
    llm: Any,
    spec: dict[str, Any],
    *,
    prompt_token_ids: list[list[int]],
    label_jsons: list[str],
    sampling_params: Any,
    lora_request_cls: Any,
    tokens_prompt_cls: Any,
    lora_int_id: int,
    dataset_name: str,
) -> tuple[float, list[str], list[str | None], list[bool], list[dict[str, Any]]]:
    lora_request = lora_request_cls(
        lora_name=f"{spec['spec_id']}_ep{spec['epoch']}",
        lora_int_id=lora_int_id,
        lora_path=spec["adapter_path"],
    )
    token_prompts = [tokens_prompt_cls(prompt_token_ids=ids) for ids in prompt_token_ids]
    outputs = llm.generate(
        token_prompts, sampling_params=sampling_params, lora_request=lora_request,
    )
    responses: list[str] = [o.outputs[0].text for o in outputs]
    metas: list[dict[str, Any]] = [json.loads(s) for s in label_jsons]
    predicted, correct = _predict_and_score(dataset_name, responses, metas)
    performance = sum(correct) / len(correct)
    return performance, responses, predicted, correct, metas


def _process_one_spec(
    claimed_path: Path,
    *,
    queue_root: Path,
    llm: Any,
    sampling_params: Any,
    lora_request_cls: Any,
    tokens_prompt_cls: Any,
    prompts_by_dataset: dict[str, tuple[list[list[int]], list[str]]],
    results_file: Path,
    generations_dir: Path,
    lora_int_id: int,
    keep_adapters: bool,
) -> None:
    spec = json.loads(claimed_path.read_text())
    dataset_name: str = spec["eval_meta"]["dataset_name"]
    prompt_token_ids, label_jsons = prompts_by_dataset[dataset_name]
    try:
        performance, responses, predicted, correct, metas = _eval_one(
            llm, spec,
            prompt_token_ids=prompt_token_ids,
            label_jsons=label_jsons,
            sampling_params=sampling_params,
            lora_request_cls=lora_request_cls,
            tokens_prompt_cls=tokens_prompt_cls,
            lora_int_id=lora_int_id,
            dataset_name=dataset_name,
        )
    except Exception:
        target = queue_root / "failed" / f"{spec['spec_id']}_ep{spec['epoch']}.json"
        os.rename(claimed_path, target)
        raise

    eval_run_id = str(uuid.uuid4())
    _write_result_row(
        results_file, eval_run_id=eval_run_id, spec=spec, performance=performance,
    )
    _write_completions(
        generations_dir, eval_run_id,
        dataset_name=dataset_name,
        metas=metas, responses=responses, predicted=predicted, correct=correct,
    )

    if not keep_adapters:
        shutil.rmtree(spec["adapter_path"])

    target = queue_root / "done" / f"{spec['spec_id']}_ep{spec['epoch']}.json"
    os.rename(claimed_path, target)


def run_worker(
    *,
    queue_root: Path,
    base_model: str,
    adapter_root: Path,
    results_file: Path,
    generations_dir: Path,
    keep_adapters: bool,
    max_new_tokens: int = 1024,
    gpu_memory_utilization: float = 0.85,
    max_lora_rank: int = 16,
) -> None:
    """Drain the eval-pool pending queue, evaluating one spec at a time."""
    ensure_eval_queue_dirs(queue_root)
    pending_dir = queue_root / "pending"
    claimed_dir = queue_root / "claimed"

    if not any(_spec_base_model(p) == base_model for p in pending_dir.glob("*.json")):
        return

    _configure_vllm_env(base_model)
    llm_cls, sampling_params_cls, lora_request_cls, tokens_prompt_cls = _load_vllm()
    llm = llm_cls(
        model=base_model,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = sampling_params_cls(
        temperature=0.0, max_tokens=max_new_tokens, stop=["\n\n"],
    )
    prompts_by_dataset: dict[str, tuple[list[list[int]], list[str]]] = {
        name: load_val_prompts(base_model, name) for name in _HANDLERS
    }

    pool_id = os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"
    lora_int_id = 1

    while True:
        candidates = sorted(pending_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not any(_spec_base_model(p) == base_model for p in candidates):
            return
        claimed: Path | None = None
        for cand in candidates:
            claimed = try_claim(
                cand, claimed_dir, pool_id=pool_id, base_model_filter=base_model,
            )
            if claimed is not None:
                break
        if claimed is None:
            time.sleep(0.05)
            continue

        _process_one_spec(
            claimed,
            queue_root=queue_root,
            llm=llm,
            sampling_params=sampling_params,
            lora_request_cls=lora_request_cls,
            tokens_prompt_cls=tokens_prompt_cls,
            prompts_by_dataset=prompts_by_dataset,
            results_file=results_file,
            generations_dir=generations_dir,
            lora_int_id=lora_int_id,
            keep_adapters=keep_adapters,
        )
        lora_int_id += 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--results-file", type=Path, required=True)
    parser.add_argument("--generations-dir", type=Path, required=True)
    parser.add_argument("--keep-adapters", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-lora-rank", type=int, default=16)
    args = parser.parse_args(argv)

    run_worker(
        queue_root=args.queue_root,
        base_model=args.base_model,
        adapter_root=args.adapter_root,
        results_file=args.results_file,
        generations_dir=args.generations_dir,
        keep_adapters=args.keep_adapters,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_lora_rank=args.max_lora_rank,
    )


if __name__ == "__main__":
    main()
