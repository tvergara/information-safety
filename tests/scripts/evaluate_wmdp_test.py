"""Tests for WMDP evaluation script."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_wmdp import (
    compute_wmdp_metrics,
    evaluate,
    load_generations,
    parse_wmdp_results_file,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_generation_dir(
    tmp_path: Path,
    run_id: str,
    n_correct: int,
    n_wrong: int,
    n_refusal: int,
    subject: str = "wmdp-bio",
) -> Path:
    run_dir = tmp_path / run_id
    run_dir.mkdir()

    input_rows = []
    response_rows = []
    for i in range(n_correct + n_wrong + n_refusal):
        input_rows.append({
            "question": f"q{i}",
            "choices": ["c0", "c1", "c2", "c3"],
            "correct_answer": "A",
            "subject": subject,
        })

    for i in range(n_correct):
        response_rows.append({
            "question": f"q{i}",
            "response": "A",
            "predicted_answer": "A",
            "correct": True,
        })
    for i in range(n_wrong):
        response_rows.append({
            "question": f"q{n_correct + i}",
            "response": "B",
            "predicted_answer": "B",
            "correct": False,
        })
    for i in range(n_refusal):
        response_rows.append({
            "question": f"q{n_correct + n_wrong + i}",
            "response": "I cannot answer that.",
            "predicted_answer": None,
            "correct": False,
        })

    _write_jsonl(run_dir / "input_data.jsonl", input_rows)
    _write_jsonl(run_dir / "responses.jsonl", response_rows)
    return run_dir


class TestLoadGenerations:
    def test_loads_input_and_responses(self, tmp_path: Path) -> None:
        run_dir = _make_generation_dir(tmp_path, "run1", n_correct=2, n_wrong=1, n_refusal=1)
        input_data, response_data = load_generations(str(run_dir))
        assert len(input_data) == 4
        assert len(response_data) == 4

    def test_response_has_predicted_answer_key(self, tmp_path: Path) -> None:
        run_dir = _make_generation_dir(tmp_path, "run2", n_correct=1, n_wrong=0, n_refusal=0)
        _, response_data = load_generations(str(run_dir))
        assert "predicted_answer" in response_data[0]

    def test_input_has_correct_answer_and_subject(self, tmp_path: Path) -> None:
        run_dir = _make_generation_dir(tmp_path, "run3", n_correct=1, n_wrong=0, n_refusal=0)
        input_data, _ = load_generations(str(run_dir))
        assert "correct_answer" in input_data[0]
        assert "subject" in input_data[0]


class TestComputeWmdpMetrics:
    def test_accuracy_all_correct(self) -> None:
        input_data = [{"correct_answer": "A", "subject": "wmdp-bio"}] * 4
        response_data = [{"predicted_answer": "A", "correct": True}] * 4
        metrics = compute_wmdp_metrics(input_data, response_data)
        assert metrics["accuracy"] == 1.0

    def test_accuracy_all_wrong(self) -> None:
        input_data = [{"correct_answer": "A", "subject": "wmdp-bio"}] * 4
        response_data = [{"predicted_answer": "B", "correct": False}] * 4
        metrics = compute_wmdp_metrics(input_data, response_data)
        assert metrics["accuracy"] == 0.0

    def test_format_compliance_rate_excludes_nulls(self) -> None:
        input_data = [{"correct_answer": "A", "subject": "wmdp-bio"}] * 4
        response_data = [
            {"predicted_answer": "A", "correct": True},
            {"predicted_answer": "B", "correct": False},
            {"predicted_answer": None, "correct": False},
            {"predicted_answer": None, "correct": False},
        ]
        metrics = compute_wmdp_metrics(input_data, response_data)
        assert metrics["format_compliance"] == 0.5

    def test_format_compliance_all_refusals(self) -> None:
        input_data = [{"correct_answer": "A", "subject": "wmdp-bio"}] * 3
        response_data = [{"predicted_answer": None, "correct": False}] * 3
        metrics = compute_wmdp_metrics(input_data, response_data)
        assert metrics["format_compliance"] == 0.0

    def test_per_subject_accuracy_breakdown(self) -> None:
        input_data = [
            {"correct_answer": "A", "subject": "wmdp-bio"},
            {"correct_answer": "A", "subject": "wmdp-bio"},
            {"correct_answer": "B", "subject": "wmdp-chem"},
            {"correct_answer": "B", "subject": "wmdp-chem"},
        ]
        response_data = [
            {"predicted_answer": "A", "correct": True},
            {"predicted_answer": "B", "correct": False},
            {"predicted_answer": "B", "correct": True},
            {"predicted_answer": "A", "correct": False},
        ]
        metrics = compute_wmdp_metrics(input_data, response_data)
        assert metrics["per_subject"]["wmdp-bio"]["accuracy"] == 0.5
        assert metrics["per_subject"]["wmdp-chem"]["accuracy"] == 0.5

    def test_per_subject_format_compliance_breakdown(self) -> None:
        input_data = [
            {"correct_answer": "A", "subject": "wmdp-bio"},
            {"correct_answer": "A", "subject": "wmdp-bio"},
            {"correct_answer": "B", "subject": "wmdp-cyber"},
        ]
        response_data = [
            {"predicted_answer": "A", "correct": True},
            {"predicted_answer": None, "correct": False},
            {"predicted_answer": None, "correct": False},
        ]
        metrics = compute_wmdp_metrics(input_data, response_data)
        assert metrics["per_subject"]["wmdp-bio"]["format_compliance"] == 0.5
        assert metrics["per_subject"]["wmdp-cyber"]["format_compliance"] == 0.0

    def test_total_count(self) -> None:
        input_data = [{"correct_answer": "A", "subject": "wmdp-bio"}] * 5
        response_data = [{"predicted_answer": "A", "correct": True}] * 5
        metrics = compute_wmdp_metrics(input_data, response_data)
        assert metrics["total"] == 5


class TestEvaluate:
    def test_returns_metrics_dict(self, tmp_path: Path) -> None:
        run_dir = _make_generation_dir(
            tmp_path, "run-eval", n_correct=3, n_wrong=1, n_refusal=1
        )
        results = evaluate(str(run_dir))
        assert "accuracy" in results
        assert "format_compliance" in results
        assert "per_subject" in results
        assert "total" in results

    def test_accuracy_value(self, tmp_path: Path) -> None:
        run_dir = _make_generation_dir(
            tmp_path, "run-acc", n_correct=3, n_wrong=2, n_refusal=0
        )
        results = evaluate(str(run_dir))
        assert results["accuracy"] == 0.6
        assert results["total"] == 5

    def test_format_compliance_value(self, tmp_path: Path) -> None:
        run_dir = _make_generation_dir(
            tmp_path, "run-fmt", n_correct=2, n_wrong=1, n_refusal=2
        )
        results = evaluate(str(run_dir))
        assert results["format_compliance"] == 0.6

    def test_writes_eval_results_json(self, tmp_path: Path) -> None:
        run_dir = _make_generation_dir(
            tmp_path, "run-write", n_correct=1, n_wrong=1, n_refusal=1
        )
        evaluate(str(run_dir))
        assert (run_dir / "eval_results.json").exists()


class TestParseWmdpResultsFile:
    def test_finds_unevaluated_wmdp_rows(self, tmp_path: Path) -> None:
        results_file = tmp_path / "results.jsonl"
        rows = [
            {"eval_run_id": "id1", "dataset_name": "wmdp", "asr": None},
            {"eval_run_id": "id2", "dataset_name": "advbench_harmbench", "asr": None},
            {"eval_run_id": "id3", "dataset_name": "wmdp", "asr": 0.5},
        ]
        _write_jsonl(results_file, rows)

        unevaluated = parse_wmdp_results_file(str(results_file))
        assert len(unevaluated) == 1
        assert unevaluated[0][1]["eval_run_id"] == "id1"

    def test_skips_already_evaluated(self, tmp_path: Path) -> None:
        results_file = tmp_path / "results.jsonl"
        rows = [
            {"eval_run_id": "id1", "dataset_name": "wmdp", "asr": 0.4},
        ]
        _write_jsonl(results_file, rows)

        unevaluated = parse_wmdp_results_file(str(results_file))
        assert len(unevaluated) == 0
