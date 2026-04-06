"""Run an end-to-end jailbreak benchmark loop on a local open-source model."""

from __future__ import annotations

import argparse
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from information_safety.algorithms.jailbreak.runner import (
    generate_attack_responses,
    write_generation_artifacts,
)
from scripts.evaluate_harmbench import evaluate as evaluate_generation_dir
from scripts.run_prompt_attacks import generate_rows, load_behaviors


def load_target_model(model_name_or_path: str) -> tuple[Any, Any]:
    """Load target model + tokenizer for local benchmark generation."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def run_benchmark(
    *,
    behaviors_file: str,
    output_dir: str,
    model_name_or_path: str,
    methods: list[str],
    max_new_tokens: int,
    evaluate: bool,
) -> dict[str, Any]:
    """Execute a full prompt-attack benchmark loop."""
    behaviors = load_behaviors(behaviors_file)
    attack_rows = generate_rows(behaviors, methods)

    model, tokenizer = load_target_model(model_name_or_path)
    completed_rows = generate_attack_responses(
        attack_rows,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )

    write_generation_artifacts(output_dir, completed_rows)

    if evaluate:
        return evaluate_generation_dir(output_dir)
    return {"total": len(completed_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run jailbreak benchmark loop")
    parser.add_argument("--behaviors-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True, help="Open-source model path/id")
    parser.add_argument(
        "--methods",
        default="zero_shot,few_shot,roleplay",
        help="Comma-separated prompt methods",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    run_benchmark(
        behaviors_file=args.behaviors_file,
        output_dir=args.output_dir,
        model_name_or_path=args.model,
        methods=methods,
        max_new_tokens=args.max_new_tokens,
        evaluate=not args.skip_eval,
    )


if __name__ == "__main__":
    main()
