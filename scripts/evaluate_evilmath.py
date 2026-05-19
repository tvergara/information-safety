"""Evaluate EvilMath generations by reading the `correct` boolean per response.

Unlike HarmBench / StrongREJECT, EvilMath scoring requires no external classifier:
the per-example `correct` flag is computed at generation time by the dataset
handler (see `information_safety/algorithms/dataset_handlers/evilmath.py`).

Two modes:
  --generations-dir   Evaluate a single run directory directly.
  --results-file      Find all EvilMath rows with asr==null and backfill accuracy.

Format compliance (fraction of responses that contain a parsable numerical answer)
is reported alongside accuracy.
"""

from __future__ import annotations

import argparse
import json
from logging import getLogger
from pathlib import Path
from typing import Any

from scripts._asr_backfill_common import backfill_sink, select_shard

logger = getLogger(__name__)


def load_generations(generations_dir: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load input_data.jsonl and responses.jsonl from a run directory."""
    gen_path = Path(generations_dir)

    with open(gen_path / "input_data.jsonl") as f:
        input_data = [json.loads(line) for line in f]

    with open(gen_path / "responses.jsonl") as f:
        response_data = [json.loads(line) for line in f]

    return input_data, response_data


def compute_evilmath_metrics(
    input_data: list[dict[str, Any]],
    response_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute accuracy and format compliance over canonical rows.

    Canonical rows are those with ``pareto_step_idx == 0`` (legacy rows without the field
    are treated as 0). All rows are also emitted under ``per_row_labels`` for plotting.
    """
    pareto_step_idxs = [row.get("pareto_step_idx", 0) for row in input_data]
    canonical_responses = [
        r for r, idx in zip(response_data, pareto_step_idxs) if idx == 0
    ]

    total = len(canonical_responses)
    correct_count = sum(1 for r in canonical_responses if r["correct"])
    parsable_count = sum(
        1 for r in canonical_responses if r["predicted_answer"] is not None
    )

    accuracy = correct_count / total if total > 0 else 0.0
    format_compliance = parsable_count / total if total > 0 else 0.0

    per_row_labels = [
        {
            "question": row.get("question"),
            "label": int(resp["correct"]),
            "pareto_step_idx": idx,
            "attack_flops": resp.get("attack_flops", 0),
        }
        for row, resp, idx in zip(input_data, response_data, pareto_step_idxs)
    ]

    return {
        "accuracy": accuracy,
        "format_compliance": format_compliance,
        "total": total,
        "correct": correct_count,
        "parsable": parsable_count,
        "per_row_labels": per_row_labels,
    }


def evaluate(generations_dir: str) -> dict[str, Any]:
    """Run full evaluation on a single EvilMath generations directory."""
    input_data, response_data = load_generations(generations_dir)
    metrics = compute_evilmath_metrics(input_data, response_data)

    print(f"\nOverall accuracy:         {metrics['accuracy']:.4f} "
          f"({metrics['correct']}/{metrics['total']})")
    print(f"Format compliance rate:   {metrics['format_compliance']:.4f} "
          f"({metrics['parsable']}/{metrics['total']} responses contained a parsable answer)")

    results_path = Path(generations_dir) / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return metrics


def parse_evilmath_results_file(
    results_file: str,
) -> list[tuple[int, dict[str, Any]]]:
    """Return (row_index, row) pairs for EvilMath rows with asr == null."""
    results_path = Path(results_file)
    with open(results_path) as f:
        rows = [json.loads(line) for line in f]

    return [
        (i, row)
        for i, row in enumerate(rows)
        if row.get("dataset_name") == "evilmath"
        and row.get("asr") is None
        and row.get("invalidated_reason") is None
    ]


def evaluate_results_file(
    results_file: str,
    generations_base: str = "/network/scratch/b/brownet/information-safety/generations",
    shard_index: int | None = None,
    num_shards: int | None = None,
    output_shard_file: str | None = None,
) -> None:
    """Find unevaluated EvilMath runs in a results JSONL and backfill accuracy as ASR.

    See ``evaluate_wmdp.evaluate_results_file`` for the shard-mode contract.
    """
    results_path = Path(results_file)
    generations_dir = Path(generations_base)

    with open(results_path) as f:
        rows = [json.loads(line) for line in f]

    unevaluated = [
        (i, row)
        for i, row in enumerate(rows)
        if row.get("dataset_name") == "evilmath"
        and row.get("asr") is None
        and row.get("invalidated_reason") is None
    ]

    if num_shards is not None and shard_index is not None and num_shards > 1:
        unevaluated = [
            (i, row) for i, row in unevaluated
            if select_shard(row["eval_run_id"], shard_index=shard_index,
                            num_shards=num_shards)
        ]

    print(f"Found {len(unevaluated)} unevaluated EvilMath runs.")
    num_updated = 0

    with backfill_sink(
        output_shard_file=output_shard_file,
        rows=rows,
        results_path=results_path,
    ) as record:
        for idx, row in unevaluated:
            eval_run_id = row["eval_run_id"]
            gen_dir = generations_dir / eval_run_id

            if not gen_dir.exists():
                logger.warning("Generations dir missing for %s, skipping.", eval_run_id)
                continue

            input_data, response_data = load_generations(str(gen_dir))
            metrics = compute_evilmath_metrics(input_data, response_data)

            num_updated += 1
            print(
                f"  {eval_run_id}: "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"format_compliance={metrics['format_compliance']:.4f}"
            )
            record(idx, eval_run_id, metrics["accuracy"])

    if output_shard_file is not None:
        print(f"\nWrote {num_updated} delta records to {output_shard_file}")
    else:
        print(f"\nUpdated {num_updated} rows in {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EvilMath generations")
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
        help="Path to results JSONL file. Finds unevaluated EvilMath runs and backfills accuracy.",
    )
    parser.add_argument(
        "--generations-base",
        type=str,
        default="/network/scratch/b/brownet/information-safety/generations",
        help="Base directory containing generation subdirectories (used with --results-file).",
    )
    parser.add_argument(
        "--shard-index", type=int, default=None,
        help="Shard index in [0, num-shards). Use with --num-shards and "
        "--output-shard-file for parallel backfill.",
    )
    parser.add_argument(
        "--num-shards", type=int, default=None,
        help="Total number of shards across the parallel run.",
    )
    parser.add_argument(
        "--output-shard-file", type=str, default=None,
        help="Write JSONL delta records to this file instead of mutating the canonical "
        "results file.",
    )
    args = parser.parse_args()

    if args.results_file:
        evaluate_results_file(
            args.results_file,
            generations_base=args.generations_base,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            output_shard_file=args.output_shard_file,
        )
    elif args.generations_dir:
        evaluate(args.generations_dir)
    else:
        parser.error("Either --generations-dir or --results-file must be provided.")


if __name__ == "__main__":
    main()
