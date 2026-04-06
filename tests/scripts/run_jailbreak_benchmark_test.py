"""Tests for jailbreak benchmark runner script."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.run_jailbreak_benchmark import run_benchmark


def test_run_benchmark_writes_eval_results(tmp_path: Path) -> None:
    behaviors_file = tmp_path / "behaviors.jsonl"
    with open(behaviors_file, "w") as f:
        f.write(json.dumps({"behavior": "b1", "category": "c1"}) + "\n")

    output_dir = tmp_path / "run"

    model = MagicMock()
    tokenizer = MagicMock()

    with patch(
        "scripts.run_jailbreak_benchmark.load_target_model",
        return_value=(model, tokenizer),
    ), patch(
        "scripts.run_jailbreak_benchmark.evaluate_generation_dir",
        return_value={"overall_asr": 0.5},
    ):
        results = run_benchmark(
            behaviors_file=str(behaviors_file),
            output_dir=str(output_dir),
            model_name_or_path="mock-model",
            methods=["zero_shot"],
            max_new_tokens=4,
            evaluate=True,
        )

    assert results["overall_asr"] == 0.5
    assert (output_dir / "attack_results.jsonl").exists()
