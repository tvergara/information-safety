"""Defense data builder: produces refusals + retain JSONL files for a target.

For each ``--target`` in ``{wmdp, evilmath, both}``, writes deterministic JSONL
files at ``<output-dir>/refusals.jsonl`` and ``<output-dir>/retain.jsonl``.

- WMDP refusals: MMLU-Pro biology + computer-science (held out from cais/wmdp).
- EvilMath refusals: LLM-rewritten held-out GSM8K problems (built once via
  ``scripts/build_evilmath_refusal_source.py``).
- Retain: HuggingFaceTB/smoltalk single-turn, sized 10x the refusal count.

The refusal target string is fixed (``"I can't help with that."``); see CLAUDE.md
data-policy: this module never inspects user-facing text, only metadata.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import datasets

from information_safety.algorithms.dataset_handlers.wmdp import (
    format_multiple_choice,
)

REFUSAL_RESPONSE = "I can't help with that."

_MMLU_PRO_BIO_CATEGORY = "biology"
_MMLU_PRO_CS_CATEGORY = "computer science"

_RETAIN_RATIO = 10


def _format_mmlu_pro_prompt(example: dict[str, Any]) -> str:
    return format_multiple_choice(example["question"], example["options"])


def _category_to_source_tag(category: str) -> str:
    if category == _MMLU_PRO_BIO_CATEGORY:
        return "mmlu-pro-biology"
    if category == _MMLU_PRO_CS_CATEGORY:
        return "mmlu-pro-cs"
    raise ValueError(f"Unexpected MMLU-Pro category: {category!r}")


def _build_wmdp_refusal_rows(seed: int) -> list[dict[str, Any]]:
    raw = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    filtered = raw.filter(  # type: ignore[union-attr]
        lambda ex: ex["category"] in (_MMLU_PRO_BIO_CATEGORY, _MMLU_PRO_CS_CATEGORY)
    )

    rows: list[dict[str, Any]] = []
    for idx, example in enumerate(filtered):
        example_dict: dict[str, Any] = dict(example)  # type: ignore[arg-type]
        category = example_dict["category"]
        source = _category_to_source_tag(category)
        rows.append({
            "prompt": _format_mmlu_pro_prompt(example_dict),
            "refusal": REFUSAL_RESPONSE,
            "source": source,
            "id": f"{source}-{idx}",
        })

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def _build_evilmath_refusal_rows(rewrite_path: Path) -> list[dict[str, Any]]:
    if not rewrite_path.exists():
        raise FileNotFoundError(
            f"EvilMath rewrite file not found at {rewrite_path}. "
            "Run scripts/build_evilmath_refusal_source.py first."
        )

    rows: list[dict[str, Any]] = []
    with rewrite_path.open() as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            rows.append({
                "prompt": entry["evil_question"],
                "refusal": REFUSAL_RESPONSE,
                "source": "evilmath-llm-rewrite",
                "id": f"evilmath-llm-rewrite-{idx}",
            })
    return rows


def _write_jsonl(out_path: Path, rows: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_wmdp_refusals(out_path: Path, seed: int) -> int:
    rows = _build_wmdp_refusal_rows(seed)
    _write_jsonl(out_path, rows)
    return len(rows)


def build_evilmath_refusals(rewrite_path: Path, out_path: Path) -> int:
    rows = _build_evilmath_refusal_rows(rewrite_path)
    _write_jsonl(out_path, rows)
    return len(rows)


def build_retain(out_path: Path, num_refusals: int, seed: int) -> int:
    """Sample SmolTalk single-turn pairs, sized at ``_RETAIN_RATIO * num_refusals``.

    Returns the number of rows written.
    """
    target_size = _RETAIN_RATIO * num_refusals
    raw = datasets.load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
    single_turn = raw.filter(  # type: ignore[union-attr]
        lambda ex: len(ex["messages"]) == 2
        and ex["messages"][0]["role"] == "user"
        and ex["messages"][1]["role"] == "assistant"
    )

    rng = random.Random(seed)
    indices = rng.sample(range(len(single_turn)), target_size)  # type: ignore[arg-type]
    indices.sort()
    subset = single_turn.select(indices)  # type: ignore[union-attr]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for idx, example in zip(indices, subset):
            example_dict: dict[str, Any] = dict(example)  # type: ignore[arg-type]
            messages = example_dict["messages"]
            row = {
                "prompt": messages[0]["content"],
                "response": messages[1]["content"],
                "source": "smoltalk",
                "id": f"smoltalk-{idx}",
            }
            f.write(json.dumps(row) + "\n")
    return target_size


def _build_for_target(
    target: str,
    output_dir: Path,
    seed: int,
    evilmath_rewrite_path: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    refusals_path = output_dir / "refusals.jsonl"
    retain_path = output_dir / "retain.jsonl"

    if target == "wmdp":
        n_refusals = build_wmdp_refusals(refusals_path, seed=seed)
    elif target == "evilmath":
        if evilmath_rewrite_path is None:
            raise ValueError(
                "--evilmath-rewrite-path is required when --target is evilmath or both"
            )
        n_refusals = build_evilmath_refusals(evilmath_rewrite_path, refusals_path)
    elif target == "both":
        if evilmath_rewrite_path is None:
            raise ValueError(
                "--evilmath-rewrite-path is required when --target is evilmath or both"
            )
        wmdp_rows = _build_wmdp_refusal_rows(seed=seed)
        evil_rows = _build_evilmath_refusal_rows(evilmath_rewrite_path)
        combined = wmdp_rows + evil_rows
        _write_jsonl(refusals_path, combined)
        n_refusals = len(combined)
    else:
        raise ValueError(f"Unknown target: {target!r}")

    build_retain(retain_path, num_refusals=n_refusals, seed=seed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=["wmdp", "evilmath", "both"],
        required=True,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--evilmath-rewrite-path",
        type=Path,
        default=None,
        help="Path to the JSONL produced by scripts/build_evilmath_refusal_source.py",
    )
    args = parser.parse_args(argv)

    _build_for_target(
        target=args.target,
        output_dir=args.output_dir,
        seed=args.seed,
        evilmath_rewrite_path=args.evilmath_rewrite_path,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
