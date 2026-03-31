"""Generate adversarial completions for AdvBench prompts using a base model.

For each AdvBench example, formats the prompt as a chat message, prefills with the target, and lets
the model continue generating. Saves (prompt, completion) pairs as JSONL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_completions(
    model_name: str,
    output_path: str,
    temperature: float = 0.7,
    max_new_tokens: int = 500,
) -> None:
    """Generate completions for all AdvBench examples."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    ds = load_dataset("walledai/AdvBench", split="train")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        for example in tqdm(ds, desc="Generating"):
            prompt = example["prompt"]
            target = example["target"]

            prefill: str = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target},
                ],
                tokenize=False,
                continue_final_message=True,
            )

            inputs = tokenizer(prefill, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            new_token_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            continuation = tokenizer.decode(new_token_ids, skip_special_tokens=True)
            completion = target + continuation

            row = {"prompt": prompt, "completion": completion}
            f.write(json.dumps(row) + "\n")
            f.flush()

    print(f"Saved {len(ds)} completions to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AdvBench completions")
    parser.add_argument(
        "--model", type=str, default="HuggingFaceTB/SmolLM3-3B",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--output", type=str,
        default="/network/scratch/b/brownet/information-safety/data/advbench-completions-smollm3.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    args = parser.parse_args()

    generate_completions(args.model, args.output, args.temperature, args.max_new_tokens)


if __name__ == "__main__":
    main()
