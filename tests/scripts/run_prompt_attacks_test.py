"""Tests for prompt-attack generation script."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_prompt_attacks import generate_rows, load_behaviors, write_rows


def test_load_behaviors_reads_behavior_rows(tmp_path: Path) -> None:
    src = tmp_path / "behaviors.jsonl"
    with open(src, "w") as f:
        f.write(json.dumps({"behavior": "b1", "category": "c1"}) + "\n")
    rows = load_behaviors(str(src))
    assert rows == [{"behavior": "b1", "category": "c1"}]


def test_generate_rows_builds_attack_records() -> None:
    behaviors = [{"behavior": "b1", "category": "c1"}]
    rows = generate_rows(behaviors, methods=["zero_shot", "roleplay"])
    assert len(rows) == 2
    assert rows[0]["attack_method"] == "zero_shot"
    assert rows[0]["bits"] > 0


def test_write_rows_writes_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    write_rows(str(out), [{"a": 1}, {"a": 2}])
    with open(out) as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 2
