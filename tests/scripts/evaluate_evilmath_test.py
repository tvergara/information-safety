"""Tests for EvilMath evaluation script."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_evilmath import (
    compute_evilmath_metrics,
    evaluate,
    evaluate_results_file,
    load_generations,
    parse_evilmath_results_file,
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
    n_unparsable: int,
) -> Path:
    run_dir = tmp_path / run_id
    run_dir.mkdir()

    input_rows: list[dict] = []
    response_rows: list[dict] = []
    for i in range(n_correct + n_wrong + n_unparsable):
        input_rows.append({
            "question": f"q{i}",
            "correct_answer": 42,
            "original_question": f"orig{i}",
        })

    for i in range(n_correct):
        response_rows.append({
            "question": f"q{i}",
            "response": "42",
            "predicted_answer": 42,
            "correct": True,
        })
    for i in range(n_wrong):
        response_rows.append({
            "question": f"q{n_correct + i}",
            "response": "13",
            "predicted_answer": 13,
            "correct": False,
        })
    for i in range(n_unparsable):
        response_rows.append({
            "question": f"q{n_correct + n_wrong + i}",
            "response": "I refuse to do harmful math.",
            "predicted_answer": None,
            "correct": False,
        })

    _write_jsonl(run_dir / "input_data.jsonl", input_rows)
    _write_jsonl(run_dir / "responses.jsonl", response_rows)
    return run_dir


class TestLoadGenerations:
    def test_loads_input_and_responses(self, tmp_path: Path) -> None:
        run_dir = _make_generation_dir(tmp_path, "run1", n_correct=2, n_wrong=1, n_unparsable=1)
        input_data, response_data = load_generations(str(run_dir))
        assert len(input_data) == 4
        assert len(response_data) == 4

    def test_response_has_predicted_answer_and_correct_keys(self, tmp_path: Path) -> None:
        run_dir = _make_generation_dir(tmp_path, "run2", n_correct=1, n_wrong=0, n_unparsable=0)
        _, response_data = load_generations(str(run_dir))
        assert "predicted_answer" in response_data[0]
        assert "correct" in response_data[0]


class TestComputeEvilmathMetrics:
    def test_accuracy_all_correct(self) -> None:
        input_data = [{"correct_answer": 42}] * 4
        response_data = [{"predicted_answer": 42, "correct": True}] * 4
        metrics = compute_evilmath_metrics(input_data, response_data)
        assert metrics["accuracy"] == 1.0

    def test_accuracy_all_wrong(self) -> None:
        input_data = [{"correct_answer": 42}] * 4
        response_data = [{"predicted_answer": 13, "correct": False}] * 4
        metrics = compute_evilmath_metrics(input_data, response_data)
        assert metrics["accuracy"] == 0.0

    def test_accuracy_mixed(self) -> None:
        input_data = [{"correct_answer": 42}] * 5
        response_data = [
            {"predicted_answer": 42, "correct": True},
            {"predicted_answer": 42, "correct": True},
            {"predicted_answer": 42, "correct": True},
            {"predicted_answer": 13, "correct": False},
            {"predicted_answer": None, "correct": False},
        ]
        metrics = compute_evilmath_metrics(input_data, response_data)
        assert metrics["accuracy"] == 0.6

    def test_format_compliance_excludes_nulls(self) -> None:
        input_data = [{"correct_answer": 42}] * 4
        response_data = [
            {"predicted_answer": 42, "correct": True},
            {"predicted_answer": 13, "correct": False},
            {"predicted_answer": None, "correct": False},
            {"predicted_answer": None, "correct": False},
        ]
        metrics = compute_evilmath_metrics(input_data, response_data)
        assert metrics["format_compliance"] == 0.5

    def test_format_compliance_all_refusals(self) -> None:
        input_data = [{"correct_answer": 42}] * 3
        response_data = [{"predicted_answer": None, "correct": False}] * 3
        metrics = compute_evilmath_metrics(input_data, response_data)
        assert metrics["format_compliance"] == 0.0

    def test_total_and_correct_counts(self) -> None:
        input_data = [{"correct_answer": 42}] * 5
        response_data = [
            {"predicted_answer": 42, "correct": True},
            {"predicted_answer": 42, "correct": True},
            {"predicted_answer": 13, "correct": False},
            {"predicted_answer": 13, "correct": False},
            {"predicted_answer": None, "correct": False},
        ]
        metrics = compute_evilmath_metrics(input_data, response_data)
        assert metrics["total"] == 5
        assert metrics["correct"] == 2
        assert metrics["parsable"] == 4

    def test_zero_examples_yields_zero_accuracy(self) -> None:
        metrics = compute_evilmath_metrics([], [])
        assert metrics["accuracy"] == 0.0
        assert metrics["format_compliance"] == 0.0
        assert metrics["total"] == 0


class TestEvaluate:
    def test_returns_metrics_dict(self, tmp_path: Path) -> None:
        run_dir = _make_generation_dir(
            tmp_path, "run-eval", n_correct=3, n_wrong=1, n_unparsable=1
        )
        metrics = evaluate(str(run_dir))
        assert {"accuracy", "format_compliance", "total", "correct", "parsable"} <= metrics.keys()

    def test_accuracy_value(self, tmp_path: Path) -> None:
        run_dir = _make_generation_dir(
            tmp_path, "run-acc", n_correct=3, n_wrong=2, n_unparsable=0
        )
        metrics = evaluate(str(run_dir))
        assert metrics["accuracy"] == 0.6
        assert metrics["total"] == 5

    def test_writes_eval_results_json(self, tmp_path: Path) -> None:
        run_dir = _make_generation_dir(
            tmp_path, "run-write", n_correct=1, n_wrong=1, n_unparsable=1
        )
        evaluate(str(run_dir))
        assert (run_dir / "eval_results.json").exists()


class TestParseEvilmathResultsFile:
    def test_finds_unevaluated_evilmath_rows(self, tmp_path: Path) -> None:
        results_file = tmp_path / "results.jsonl"
        rows = [
            {"eval_run_id": "id1", "dataset_name": "evilmath", "asr": None},
            {"eval_run_id": "id2", "dataset_name": "wmdp", "asr": None},
            {"eval_run_id": "id3", "dataset_name": "evilmath", "asr": 0.5},
        ]
        _write_jsonl(results_file, rows)

        unevaluated = parse_evilmath_results_file(str(results_file))
        assert len(unevaluated) == 1
        assert unevaluated[0][1]["eval_run_id"] == "id1"

    def test_skips_already_evaluated(self, tmp_path: Path) -> None:
        results_file = tmp_path / "results.jsonl"
        rows = [
            {"eval_run_id": "id1", "dataset_name": "evilmath", "asr": 0.4},
        ]
        _write_jsonl(results_file, rows)

        unevaluated = parse_evilmath_results_file(str(results_file))
        assert len(unevaluated) == 0

    def test_skips_invalidated_rows(self, tmp_path: Path) -> None:
        results_file = tmp_path / "results.jsonl"
        rows = [
            {"eval_run_id": "id1", "dataset_name": "evilmath", "asr": None,
             "invalidated_reason": "wmdp_evilmath_eval_truncated_to_max_examples"},
            {"eval_run_id": "id2", "dataset_name": "evilmath", "asr": None},
        ]
        _write_jsonl(results_file, rows)

        unevaluated = parse_evilmath_results_file(str(results_file))
        assert len(unevaluated) == 1
        assert unevaluated[0][1]["eval_run_id"] == "id2"


class TestEvaluateResultsFile:
    def test_backfills_asr_into_matching_rows(self, tmp_path: Path) -> None:
        gen_base = tmp_path / "generations"
        gen_base.mkdir()
        _make_generation_dir(gen_base, "id1", n_correct=3, n_wrong=2, n_unparsable=0)

        results_file = tmp_path / "results.jsonl"
        rows = [
            {"eval_run_id": "id1", "dataset_name": "evilmath", "asr": None},
            {"eval_run_id": "id2", "dataset_name": "wmdp", "asr": None},
        ]
        _write_jsonl(results_file, rows)

        evaluate_results_file(str(results_file), generations_base=str(gen_base))

        with open(results_file) as f:
            updated = [json.loads(line) for line in f]
        assert updated[0]["asr"] == 0.6
        assert updated[1]["asr"] is None

    def test_no_op_when_nothing_unevaluated(self, tmp_path: Path) -> None:
        gen_base = tmp_path / "generations"
        gen_base.mkdir()
        results_file = tmp_path / "results.jsonl"
        rows = [{"eval_run_id": "id1", "dataset_name": "evilmath", "asr": 0.7}]
        _write_jsonl(results_file, rows)
        evaluate_results_file(str(results_file), generations_base=str(gen_base))
        with open(results_file) as f:
            updated = [json.loads(line) for line in f]
        assert updated[0]["asr"] == 0.7

    def test_skips_when_generations_dir_missing(self, tmp_path: Path) -> None:
        gen_base = tmp_path / "generations"
        gen_base.mkdir()
        results_file = tmp_path / "results.jsonl"
        rows = [{"eval_run_id": "missing-id", "dataset_name": "evilmath", "asr": None}]
        _write_jsonl(results_file, rows)
        evaluate_results_file(str(results_file), generations_base=str(gen_base))
        with open(results_file) as f:
            updated = [json.loads(line) for line in f]
        assert updated[0]["asr"] is None
