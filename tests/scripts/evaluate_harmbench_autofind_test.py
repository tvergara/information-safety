"""Tests for auto-find mode in evaluate_harmbench.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.evaluate_harmbench import evaluate_results_file


class TestEvaluateResultsFile:
    def _setup_results_and_generations(
        self, tmp_path: Path, eval_run_id: str, asr: float | None = None
    ) -> Path:
        """Create a results file and generation files for a single run."""
        results_file = tmp_path / "final-results.jsonl"
        row = {
            "experiment_name": "LoRAStrategy",
            "experiment_id": None,
            "eval_run_id": eval_run_id,
            "model_name": "test-model",
            "dataset_name": "advbench_harmbench",
            "max_examples": None,
            "performance": None,
            "asr": asr,
            "bits": 1000,
            "seed": 42,
            "epoch": 0,
            "strategy_hparams": {},
        }
        with open(results_file, "w") as f:
            f.write(json.dumps(row) + "\n")

        gen_dir = tmp_path / "generations" / eval_run_id
        gen_dir.mkdir(parents=True)

        input_data = [{"behavior": "do X", "category": "cyber"}]
        response_data = [{"behavior": "do X", "response": "sure here is how"}]

        with open(gen_dir / "input_data.jsonl", "w") as f:
            for r in input_data:
                f.write(json.dumps(r) + "\n")

        with open(gen_dir / "responses.jsonl", "w") as f:
            for r in response_data:
                f.write(json.dumps(r) + "\n")

        return results_file

    @patch("scripts.evaluate_harmbench.load_classifier")
    def test_finds_and_evaluates_null_asr_rows(
        self, mock_load_cls: MagicMock, tmp_path: Path
    ) -> None:
        eval_run_id = "abc-123-uuid"
        results_file = self._setup_results_and_generations(tmp_path, eval_run_id)

        model = MagicMock()
        tokenizer = MagicMock()
        mock_load_cls.return_value = (model, tokenizer)

        with patch("scripts.evaluate_harmbench.run_classifier", return_value=[1]):
            evaluate_results_file(str(results_file), str(tmp_path / "generations"))

        with open(results_file) as f:
            rows = [json.loads(line) for line in f]

        assert len(rows) == 1
        assert rows[0]["asr"] == 1.0

    @patch("scripts.evaluate_harmbench.load_classifier")
    def test_skips_rows_with_existing_asr(
        self, mock_load_cls: MagicMock, tmp_path: Path
    ) -> None:
        eval_run_id = "already-evaluated"
        results_file = self._setup_results_and_generations(
            tmp_path, eval_run_id, asr=0.5
        )

        model = MagicMock()
        tokenizer = MagicMock()
        mock_load_cls.return_value = (model, tokenizer)

        with patch("scripts.evaluate_harmbench.run_classifier") as mock_run:
            evaluate_results_file(str(results_file), str(tmp_path / "generations"))

        mock_run.assert_not_called()

    @patch("scripts.evaluate_harmbench.load_classifier")
    def test_handles_multiple_rows(
        self, mock_load_cls: MagicMock, tmp_path: Path
    ) -> None:
        results_file = tmp_path / "final-results.jsonl"
        rows = []
        for i, run_id in enumerate(["run-1", "run-2", "run-3"]):
            row = {
                "experiment_name": "LoRAStrategy",
                "experiment_id": None,
                "eval_run_id": run_id,
                "model_name": "test-model",
                "dataset_name": "advbench_harmbench",
                "max_examples": None,
                "performance": None,
                "asr": 0.5 if i == 1 else None,
                "bits": 1000,
                "seed": 42,
                "epoch": i,
                "strategy_hparams": {},
            }
            rows.append(row)

            gen_dir = tmp_path / "generations" / run_id
            gen_dir.mkdir(parents=True)
            with open(gen_dir / "input_data.jsonl", "w") as f:
                f.write(json.dumps({"behavior": "do X", "category": "cyber"}) + "\n")
            with open(gen_dir / "responses.jsonl", "w") as f:
                f.write(
                    json.dumps({"behavior": "do X", "response": "sure"}) + "\n"
                )

        with open(results_file, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        model = MagicMock()
        tokenizer = MagicMock()
        mock_load_cls.return_value = (model, tokenizer)

        with patch("scripts.evaluate_harmbench.run_classifier", return_value=[1]):
            evaluate_results_file(str(results_file), str(tmp_path / "generations"))

        with open(results_file) as f:
            updated_rows = [json.loads(line) for line in f]

        assert len(updated_rows) == 3
        assert updated_rows[0]["asr"] == 1.0  # was null, now evaluated
        assert updated_rows[1]["asr"] == 0.5  # unchanged
        assert updated_rows[2]["asr"] == 1.0  # was null, now evaluated

    @patch("scripts.evaluate_harmbench.load_classifier")
    def test_skips_missing_generations_dir(
        self, mock_load_cls: MagicMock, tmp_path: Path
    ) -> None:
        results_file = tmp_path / "final-results.jsonl"
        row = {
            "experiment_name": "LoRAStrategy",
            "experiment_id": None,
            "eval_run_id": "missing-gens",
            "model_name": "test-model",
            "dataset_name": "advbench_harmbench",
            "max_examples": None,
            "performance": None,
            "asr": None,
            "bits": 1000,
            "seed": 42,
            "epoch": 0,
            "strategy_hparams": {},
        }
        with open(results_file, "w") as f:
            f.write(json.dumps(row) + "\n")

        model = MagicMock()
        tokenizer = MagicMock()
        mock_load_cls.return_value = (model, tokenizer)

        with patch("scripts.evaluate_harmbench.run_classifier") as mock_run:
            evaluate_results_file(str(results_file), str(tmp_path / "generations"))

        mock_run.assert_not_called()
