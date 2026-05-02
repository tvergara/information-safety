"""Evaluate WMDP generations by parsing A–D answers from model responses.

Unlike HarmBench, WMDP scoring does not require an external classifier — answers
are extracted by pattern matching A–D letters from each response.

Two modes:
  --generations-dir   Evaluate a single run directory directly.
  --results-file      Find all WMDP rows with asr==null and backfill accuracy.

Format compliance (fraction of responses that contain a parsable A–D letter) is
reported alongside accuracy — this is the primary signal for whether models are
refusing or following the format.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Any

logger = getLogger(__name__)


def load_generations(generations_dir: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load input_data.jsonl and responses.jsonl from a run directory."""
    gen_path = Path(generations_dir)

    with open(gen_path / "input_data.jsonl") as f:
        input_data = [json.loads(line) for line in f]

    with open(gen_path / "responses.jsonl") as f:
        response_data = [json.loads(line) for line in f]

    return input_data, response_data


def compute_wmdp_metrics(
    input_data: list[dict[str, Any]],
    response_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute accuracy and format compliance across all examples and per subject."""
    total = len(input_data)
    correct_count = sum(1 for r in response_data if r["correct"])
    parsable_count = sum(1 for r in response_data if r["predicted_answer"] is not None)

    accuracy = correct_count / total if total > 0 else 0.0
    format_compliance = parsable_count / total if total > 0 else 0.0

    subject_buckets: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"correct": [], "predicted": []}
    )
    for inp, resp in zip(input_data, response_data):
        subject = inp["subject"]
        subject_buckets[subject]["correct"].append(resp["correct"])
        subject_buckets[subject]["predicted"].append(resp["predicted_answer"])

    per_subject: dict[str, dict[str, float]] = {}
    for subject, buckets in sorted(subject_buckets.items()):
        n = len(buckets["correct"])
        per_subject[subject] = {
            "accuracy": sum(buckets["correct"]) / n if n > 0 else 0.0,
            "format_compliance": (
                sum(1 for p in buckets["predicted"] if p is not None) / n if n > 0 else 0.0
            ),
            "total": n,
        }

    return {
        "accuracy": accuracy,
        "format_compliance": format_compliance,
        "total": total,
        "correct": correct_count,
        "parsable": parsable_count,
        "per_subject": per_subject,
    }


def evaluate(generations_dir: str) -> dict[str, Any]:
    """Run full evaluation on a single WMDP generations directory."""
    input_data, response_data = load_generations(generations_dir)
    metrics = compute_wmdp_metrics(input_data, response_data)

    print(f"\nOverall accuracy:         {metrics['accuracy']:.4f} "
          f"({metrics['correct']}/{metrics['total']})")
    print(f"Format compliance rate:   {metrics['format_compliance']:.4f} "
          f"({metrics['parsable']}/{metrics['total']} responses contained a parsable A–D letter)")

    print("\nPer-subject breakdown:")
    for subject, submetrics in metrics["per_subject"].items():
        print(
            f"  {subject}: "
            f"accuracy={submetrics['accuracy']:.4f}, "
            f"format_compliance={submetrics['format_compliance']:.4f}, "
            f"n={submetrics['total']}"
        )

    results_path = Path(generations_dir) / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return metrics


def parse_wmdp_results_file(
    results_file: str,
) -> list[tuple[int, dict[str, Any]]]:
    """Return (row_index, row) pairs for WMDP rows with asr == null."""
    results_path = Path(results_file)
    with open(results_path) as f:
        rows = [json.loads(line) for line in f]

    return [
        (i, row)
        for i, row in enumerate(rows)
        if row.get("dataset_name") == "wmdp" and row.get("asr") is None
    ]


def evaluate_results_file(
    results_file: str,
    generations_base: str = "/network/scratch/b/brownet/information-safety/generations",
) -> None:
    """Find unevaluated WMDP runs in a results JSONL and backfill accuracy as ASR."""
    results_path = Path(results_file)
    generations_dir = Path(generations_base)

    with open(results_path) as f:
        rows = [json.loads(line) for line in f]

    unevaluated = [
        (i, row)
        for i, row in enumerate(rows)
        if row.get("dataset_name") == "wmdp" and row.get("asr") is None
    ]

    if not unevaluated:
        print("No unevaluated WMDP runs found.")
        return

    print(f"Found {len(unevaluated)} unevaluated WMDP runs.")
    num_updated = 0

    for idx, row in unevaluated:
        eval_run_id = row["eval_run_id"]
        gen_dir = generations_dir / eval_run_id

        if not gen_dir.exists():
            logger.warning("Generations dir missing for %s, skipping.", eval_run_id)
            continue

        input_data, response_data = load_generations(str(gen_dir))
        metrics = compute_wmdp_metrics(input_data, response_data)

        rows[idx]["asr"] = metrics["accuracy"]
        num_updated += 1

        print(
            f"  {eval_run_id}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"format_compliance={metrics['format_compliance']:.4f}"
        )
        print("  Per-subject:")
        for subject, submetrics in metrics["per_subject"].items():
            print(
                f"    {subject}: "
                f"accuracy={submetrics['accuracy']:.4f}, "
                f"format_compliance={submetrics['format_compliance']:.4f}"
            )

        with open(results_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    print(f"\nUpdated {num_updated} rows in {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WMDP generations")
    parser.add_argument(
        "--generations-dir",
        type=str,
        default=None,
        help="Path to directory containing input_data.jsonl and responses.jsonl",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Path to results JSONL file. Finds unevaluated WMDP runs and backfills accuracy.",
    )
    parser.add_argument(
        "--generations-base",
        type=str,
        default="/network/scratch/b/brownet/information-safety/generations",
        help="Base directory containing generation subdirectories (used with --results-file).",
    )
    args = parser.parse_args()

    if args.results_file:
        evaluate_results_file(
            args.results_file,
            generations_base=args.generations_base,
        )
    elif args.generations_dir:
        evaluate(args.generations_dir)
    else:
        parser.error("Either --generations-dir or --results-file must be provided.")


if __name__ == "__main__":
    main()
