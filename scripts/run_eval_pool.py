"""vLLM-backed eval pool worker: drains pending eval specs and writes one
result row per spec with experiment_name="DataStrategyDeferredEval"."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from information_safety.algorithms.dataset_handlers.wmdp import (
    WMDPHandler,
    parse_answer_letter,
)


def ensure_eval_queue_dirs(queue_root: Path) -> None:
    for sub in ("pending", "claimed", "done", "failed", "logs"):
        (queue_root / sub).mkdir(parents=True, exist_ok=True)


def try_claim(
    pending_path: Path, claimed_dir: Path, *, pool_id: str
) -> Path | None:
    job_id = pending_path.stem
    target = claimed_dir / f"{job_id}.{pool_id}.json"
    try:
        os.rename(pending_path, target)
    except FileNotFoundError:
        return None
    return target


def load_wmdp_val_prompts(
    base_model: str, max_length: int = 500,
) -> tuple[list[str], list[str]]:
    """Render the same WMDP val prompts the HF eval path produces.

    Returns ``(prompts, label_jsons)`` parallel lists.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    handler = WMDPHandler(max_length=max_length, generations_dir="")
    dataset = handler.get_val_dataset(tokenizer)
    prompts: list[str] = []
    labels: list[str] = []
    for item in dataset:  # type: ignore[union-attr]
        text = tokenizer.decode(item["input_ids"], skip_special_tokens=False)
        prompts.append(text)
        labels.append(item["labels"])
    return prompts, labels


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
    metas: list[dict[str, Any]],
    responses: list[str],
    predicted: list[str | None],
    correct: list[bool],
) -> None:
    run_dir = generations_dir / eval_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "input_data.jsonl", "w") as f:
        for meta in metas:
            f.write(json.dumps({
                "question": meta["question"],
                "choices": meta["choices"],
                "correct_answer": meta["correct_answer"],
                "subject": meta["subject"],
            }) + "\n")
    with open(run_dir / "responses.jsonl", "w") as f:
        for meta, resp, pred, ok in zip(metas, responses, predicted, correct):
            f.write(json.dumps({
                "question": meta["question"],
                "response": resp,
                "predicted_answer": pred,
                "correct": ok,
            }) + "\n")


def _eval_one(
    llm: Any,
    spec: dict[str, Any],
    *,
    prompts: list[str],
    label_jsons: list[str],
    sampling_params: Any,
    lora_int_id: int,
    max_new_tokens: int,
) -> tuple[float, list[str], list[str | None], list[bool], list[dict[str, Any]]]:
    lora_request = LoRARequest(
        lora_name=f"{spec['spec_id']}_ep{spec['epoch']}",
        lora_int_id=lora_int_id,
        lora_path=spec["adapter_path"],
    )
    outputs = llm.generate(
        prompts, sampling_params=sampling_params, lora_request=lora_request,
    )
    responses: list[str] = [o.outputs[0].text for o in outputs]
    predicted: list[str | None] = [parse_answer_letter(r) for r in responses]
    metas: list[dict[str, Any]] = [json.loads(s) for s in label_jsons]
    correct: list[bool] = [p == m["correct_answer"] for p, m in zip(predicted, metas)]
    performance = sum(correct) / len(correct)
    return performance, responses, predicted, correct, metas


def _process_one_spec(
    claimed_path: Path,
    *,
    queue_root: Path,
    llm: Any,
    sampling_params: Any,
    prompts: list[str],
    label_jsons: list[str],
    results_file: Path,
    generations_dir: Path,
    lora_int_id: int,
    max_new_tokens: int,
    keep_adapters: bool,
) -> None:
    spec = json.loads(claimed_path.read_text())
    try:
        performance, responses, predicted, correct, metas = _eval_one(
            llm, spec,
            prompts=prompts, label_jsons=label_jsons,
            sampling_params=sampling_params,
            lora_int_id=lora_int_id,
            max_new_tokens=max_new_tokens,
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

    if not any(pending_dir.glob("*.json")):
        return

    llm = LLM(
        model=base_model,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    prompts, label_jsons = load_wmdp_val_prompts(base_model)

    pool_id = os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"
    lora_int_id = 1

    while True:
        candidates = sorted(pending_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            return
        claimed: Path | None = None
        for cand in candidates:
            claimed = try_claim(cand, claimed_dir, pool_id=pool_id)
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
            prompts=prompts,
            label_jsons=label_jsons,
            results_file=results_file,
            generations_dir=generations_dir,
            lora_int_id=lora_int_id,
            max_new_tokens=max_new_tokens,
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
