from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import datasets
import torch
from vllm import LLM, SamplingParams

CHOICES = ("A", "B", "C", "D")

_ANSWER_RE = re.compile(r"\b([ABCD])\b")


def format_question(item: dict[str, Any]) -> str:
    a, b, c, d = item["choices"]
    return (
        f"{item['question']}\n"
        f"A. {a}\nB. {b}\nC. {c}\nD. {d}\n"
        f"Answer:"
    )


def format_fewshot_prompt(
    query: dict[str, Any],
    dev_examples: list[dict[str, Any]],
    subject: str,
    num_shots: int = 5,
) -> str:
    pretty_subject = subject.replace("_", " ")
    parts = [
        f"The following are multiple choice questions (with answers) about {pretty_subject}.\n"
    ]
    for ex in dev_examples[:num_shots]:
        parts.append(format_question(ex) + f" {CHOICES[ex['answer']]}\n")
    parts.append(format_question(query))
    return "\n".join(parts)


def parse_answer(text: str) -> str | None:
    match = _ANSWER_RE.search(text)
    return match.group(1) if match else None


def compute_metrics(
    predictions: list[str | None],
    golds: list[int],
    subjects: list[str],
) -> dict[str, Any]:
    total = len(predictions)
    correct = 0
    parsable = 0
    per_subject_buckets: dict[str, dict[str, list[int]]] = {}
    for pred, gold, subj in zip(predictions, golds, subjects):
        gold_letter = CHOICES[gold]
        is_parsable = pred is not None
        is_correct = is_parsable and pred == gold_letter
        parsable += int(is_parsable)
        correct += int(is_correct)
        bucket = per_subject_buckets.setdefault(subj, {"correct": [], "parsable": []})
        bucket["correct"].append(int(is_correct))
        bucket["parsable"].append(int(is_parsable))

    per_subject = {
        subj: {
            "accuracy": sum(b["correct"]) / len(b["correct"]),
            "format_compliance": sum(b["parsable"]) / len(b["parsable"]),
            "total": len(b["correct"]),
        }
        for subj, b in sorted(per_subject_buckets.items())
    }
    return {
        "accuracy": correct / total,
        "format_compliance": parsable / total,
        "total": total,
        "correct": correct,
        "parsable": parsable,
        "per_subject": per_subject,
    }


def build_prompts(
    test_items: list[dict[str, Any]],
    dev_by_subject: dict[str, list[dict[str, Any]]],
    num_shots: int,
) -> list[str]:
    return [
        format_fewshot_prompt(item, dev_by_subject[item["subject"]], item["subject"], num_shots)
        for item in test_items
    ]


def evaluate_mmlu(
    model_name_or_path: str,
    output_path: Path,
    max_examples: int | None = None,
    num_shots: int = 5,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    test = datasets.load_dataset("cais/mmlu", "all", split="test")
    dev = datasets.load_dataset("cais/mmlu", "all", split="dev")

    if max_examples is not None:
        test = test.select(range(max_examples))

    test_items: list[dict[str, Any]] = list(test)
    dev_items: list[dict[str, Any]] = list(dev)
    dev_by_subject: dict[str, list[dict[str, Any]]] = {}
    for ex in dev_items:
        dev_by_subject.setdefault(ex["subject"], []).append(ex)

    prompts = build_prompts(test_items, dev_by_subject, num_shots)

    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=torch.cuda.device_count(),
        disable_custom_all_reduce=True,
        trust_remote_code=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, params)
    responses = [o.outputs[0].text for o in outputs]

    predictions = [parse_answer(r) for r in responses]
    golds = [int(item["answer"]) for item in test_items]
    subjects = [item["subject"] for item in test_items]

    metrics = compute_metrics(predictions, golds, subjects)
    metrics["model"] = model_name_or_path
    metrics["num_shots"] = num_shots
    metrics["max_examples"] = max_examples

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate MMLU 5-shot accuracy")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args(argv)

    evaluate_mmlu(
        model_name_or_path=args.model_name_or_path,
        output_path=args.output_path,
        max_examples=args.max_examples,
        num_shots=args.num_shots,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
