"""End-to-end runner utilities for jailbreak benchmark loops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from information_safety.algorithms.jailbreak.metrics import compute_cross_entropy_bits


def generate_attack_responses(
    rows: list[dict[str, Any]],
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Generate one response for each attack prompt row."""
    bits_cache: dict[str, float] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row["attack_prompt"])
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

        new_token_ids = output_ids[0, input_ids.shape[1]:]
        response = str(tokenizer.decode(new_token_ids, skip_special_tokens=True))

        template = str(row["attack_template"])
        if template not in bits_cache:
            bits_cache[template] = compute_cross_entropy_bits(
                template, model=model, tokenizer=tokenizer
            )

        enriched = dict(row)
        enriched["response"] = response
        enriched["bits"] = bits_cache[template]
        out.append(enriched)

    return out


def write_generation_artifacts(output_dir: str, rows: list[dict[str, Any]]) -> None:
    """Write standard HarmBench-compatible and method-specific artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "input_data.jsonl", "w") as f:
        for row in rows:
            payload = {
                "behavior": row["behavior"],
                "category": row.get("category", "unknown"),
            }
            f.write(json.dumps(payload) + "\n")

    with open(out_dir / "responses.jsonl", "w") as f:
        for row in rows:
            payload = {
                "behavior": row["behavior"],
                "response": row["response"],
            }
            f.write(json.dumps(payload) + "\n")

    with open(out_dir / "attack_results.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
