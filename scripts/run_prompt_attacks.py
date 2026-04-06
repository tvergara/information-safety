"""Generate prompt-attack rows from behavior JSONL.

Input rows should contain at least a `behavior` field, as emitted by benchmark
handlers such as HarmBench exports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from information_safety.algorithms.jailbreak.prompt_methods import (
    PromptAttackMethod,
    build_prompt_attack_record,
)


def load_behaviors(input_file: str) -> list[dict[str, Any]]:
    """Load behavior JSONL rows."""
    rows: list[dict[str, Any]] = []
    with open(input_file) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def generate_rows(behaviors: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    """Generate prompt-attack rows with method name and bit counts."""
    out: list[dict[str, Any]] = []
    parsed_methods = [PromptAttackMethod(m) for m in methods]
    for row in behaviors:
        behavior = str(row["behavior"])
        category = row.get("category")
        for method in parsed_methods:
            if method is PromptAttackMethod.HUMAN_JAILBREAK:
                template = row.get("human_jailbreak_template")
                record = build_prompt_attack_record(
                    method,
                    behavior,
                    human_jailbreak_template=str(template) if template is not None else None,
                )
            else:
                record = build_prompt_attack_record(method, behavior)
            record["behavior"] = behavior
            if category is not None:
                record["category"] = str(category)
            out.append(record)
    return out


def write_rows(output_file: str, rows: list[dict[str, Any]]) -> None:
    """Write rows to JSONL output."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prompt attack rows")
    parser.add_argument("--input-file", required=True, help="Input behavior JSONL")
    parser.add_argument("--output-file", required=True, help="Output attack JSONL")
    parser.add_argument(
        "--methods",
        default="zero_shot,few_shot,roleplay",
        help="Comma-separated methods from: zero_shot,few_shot,human_jailbreak,roleplay",
    )
    args = parser.parse_args()

    behaviors = load_behaviors(args.input_file)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows = generate_rows(behaviors, methods)
    write_rows(args.output_file, rows)


if __name__ == "__main__":
    main()
