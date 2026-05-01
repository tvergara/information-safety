"""Minimal naive inference script.

Loads a Hugging Face causal LM, tokenizes a single user prompt, generates a
greedy completion, and prints it to stdout.

This file's UTF-8 bit-length serves as the code-bits floor for every attack
in the program-length accounting. Per the project's measurement definitions,
shared lightning/orchestration glue does not count as attack code; this
minimal script stands in for that glue and is included as a floor for every
attack's `code_bits` term.

This module imports nothing from `information_safety/` itself.
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def naive_inference(
    model_name_or_path: str,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """Load a causal LM, run greedy generation on `prompt`, and return the text."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    chat_input: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = tokenizer(chat_input, return_tensors="pt")
    input_ids: torch.Tensor = encoded["input_ids"].to(model.device)
    attention_mask: torch.Tensor = encoded["attention_mask"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    response = naive_inference(args.model, args.prompt, args.max_new_tokens)
    print(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
