"""Run prompt-attack benchmark: generate responses per method, save to shared results."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from information_safety.algorithms.jailbreak.metrics import compute_cross_entropy_bits
from information_safety.algorithms.jailbreak.prompt_methods import (
    PromptAttackMethod,
    build_prompt,
    build_template,
)
from scripts.run_prompt_attacks import load_behaviors

DEFAULT_RESULTS_FILE = Path(
    "/network/scratch/b/brownet/information-safety/results/final-results.jsonl"
)
DEFAULT_GENERATIONS_DIR = Path("/network/scratch/b/brownet/information-safety/generations")


def load_target_model(
    model_name_or_path: str,
) -> tuple[Any, PreTrainedTokenizerBase]:
    """Load target model + tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def generate_response(
    prompt: str,
    *,
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens: int,
) -> str:
    """Generate a single response from the model."""
    inputs: dict[str, torch.Tensor] = tokenizer(  # type: ignore[assignment]
        [prompt], return_tensors="pt", padding=True, truncation=True,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        output_ids: torch.Tensor = model.generate(  # type: ignore[assignment]
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_token_ids = output_ids[0, input_ids.shape[1] :]
    return str(tokenizer.decode(new_token_ids, skip_special_tokens=True))


def run_method(
    *,
    method: PromptAttackMethod,
    behaviors: list[dict[str, Any]],
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    max_new_tokens: int,
    seed: int,
    results_file: Path = DEFAULT_RESULTS_FILE,
    generations_dir: Path = DEFAULT_GENERATIONS_DIR,
) -> None:
    """Run one prompt-attack method: generate, save artifacts, append result row."""
    template = build_template(method)
    bits = compute_cross_entropy_bits(template, model=model, tokenizer=tokenizer)

    completions: list[dict[str, str]] = []
    for row in behaviors:
        behavior = str(row["behavior"])
        category = row.get("category", "unknown")
        prompt = build_prompt(method, behavior)
        response = generate_response(
            prompt, model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens
        )
        completions.append({
            "behavior": behavior,
            "category": str(category),
            "response": response,
        })

    eval_run_id = str(uuid.uuid4())
    run_dir = generations_dir / eval_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "input_data.jsonl", "w") as f:
        for entry in completions:
            f.write(json.dumps({"behavior": entry["behavior"], "category": entry["category"]}) + "\n")

    with open(run_dir / "responses.jsonl", "w") as f:
        for entry in completions:
            f.write(json.dumps({"behavior": entry["behavior"], "response": entry["response"]}) + "\n")

    print(f"  Saved {len(completions)} completions to {run_dir}")

    result_row = {
        "experiment_name": method.value,
        "experiment_id": None,
        "eval_run_id": eval_run_id,
        "model_name": model_name,
        "dataset_name": "advbench_harmbench",
        "max_examples": None,
        "performance": None,
        "asr": None,
        "bits": bits,
        "seed": seed,
        "epoch": 0,
        "strategy_hparams": {},
    }

    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "a") as f:
        f.write(json.dumps(result_row) + "\n")

    print(f"  {method.value}: bits={bits:.1f}, eval_run_id={eval_run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompt-attack benchmark")
    parser.add_argument("--behaviors-file", required=True)
    parser.add_argument("--model", required=True, help="HuggingFace model path/id")
    parser.add_argument(
        "--methods",
        default="zero_shot,few_shot,roleplay",
        help="Comma-separated prompt methods",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--generations-dir", type=Path, default=DEFAULT_GENERATIONS_DIR)
    args = parser.parse_args()

    behaviors = load_behaviors(args.behaviors_file)
    methods = [PromptAttackMethod(m.strip()) for m in args.methods.split(",") if m.strip()]

    print(f"Loading model: {args.model}")
    model, tokenizer = load_target_model(args.model)

    for method in methods:
        print(f"Running {method.value}...")
        run_method(
            method=method,
            behaviors=behaviors,
            model=model,
            tokenizer=tokenizer,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            results_file=args.results_file,
            generations_dir=args.generations_dir,
        )

    print("Done. Run eval-sweep-results.sh to backfill ASR.")


if __name__ == "__main__":
    main()
