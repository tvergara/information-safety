"""Build AdversariaLLM behaviors + optimizer-target files for WMDP and EvilMath.

Writes four files to ``<behaviors_csv_dir>``: ``{wmdp,evilmath}_behaviors.csv``
and ``{wmdp,evilmath}_targets_text.json``. Targets use a single generic
affirmative string per dataset (see ``WMDP_GENERIC_TARGET`` and
``EVILMATH_GENERIC_TARGET``). Per CLAUDE.md, this script must never print or
log behavior text or target text.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, cast

import datasets

from information_safety.algorithms.dataset_handlers.evilmath import (
    extract_numerical_answer,
)
from information_safety.algorithms.dataset_handlers.wmdp import (
    WMDP_SUBSETS,
    format_multiple_choice,
)

HARMBENCH_CSV_COLUMNS = [
    "Behavior",
    "FunctionalCategory",
    "SemanticCategory",
    "Tags",
    "ContextString",
    "BehaviorID",
]

WMDP_GENERIC_TARGET = "Sure, the answer is"
EVILMATH_GENERIC_TARGET = "Sure, here is the solution:"


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HARMBENCH_CSV_COLUMNS)
        for row in rows:
            writer.writerow(row)


def _write_targets(path: Path, ids: list[str], target: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    targets = {bid: target for bid in ids}
    with open(path, "w") as f:
        json.dump(targets, f)


def build_wmdp(*, csv_path: Path, targets_path: Path) -> int:
    rows: list[list[str]] = []
    ids: list[str] = []
    for subject in WMDP_SUBSETS:
        subset = datasets.load_dataset("cais/wmdp", subject, split="test")
        for i, example in enumerate(cast(list[dict[str, Any]], subset)):
            behavior = format_multiple_choice(example["question"], example["choices"])
            behavior_id = f"{subject}_{i:04d}"
            rows.append([
                behavior,
                "standard",
                "chemical_biological",
                "",
                "",
                behavior_id,
            ])
            ids.append(behavior_id)

    _write_csv(csv_path, rows)
    _write_targets(targets_path, ids, WMDP_GENERIC_TARGET)
    return len(rows)


def build_evilmath(*, csv_path: Path, targets_path: Path) -> int:
    raw = datasets.load_dataset("ethz-spylab/EvilMath", split="test")
    rows: list[list[str]] = []
    ids: list[str] = []
    for i, example in enumerate(cast(list[dict[str, Any]], raw)):
        if extract_numerical_answer(example["original_answer"]) is None:
            continue
        behavior = example["evil_question"]
        behavior_id = f"evilmath_{i:04d}"
        rows.append([
            behavior,
            "standard",
            "harmful",
            "",
            "",
            behavior_id,
        ])
        ids.append(behavior_id)

    _write_csv(csv_path, rows)
    _write_targets(targets_path, ids, EVILMATH_GENERIC_TARGET)
    return len(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build WMDP/EvilMath behaviors+targets for AdversariaLLM."
    )
    parser.add_argument("--behaviors-csv-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    wmdp_rows = build_wmdp(
        csv_path=args.behaviors_csv_dir / "wmdp_behaviors.csv",
        targets_path=args.behaviors_csv_dir / "wmdp_targets_text.json",
    )
    evilmath_rows = build_evilmath(
        csv_path=args.behaviors_csv_dir / "evilmath_behaviors.csv",
        targets_path=args.behaviors_csv_dir / "evilmath_targets_text.json",
    )
    print(f"Wrote WMDP behaviors: {wmdp_rows} rows")
    print(f"Wrote EvilMath behaviors: {evilmath_rows} rows")


if __name__ == "__main__":
    main()
