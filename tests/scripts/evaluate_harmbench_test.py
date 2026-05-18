"""Tests for HarmBench evaluation script."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from scripts.evaluate_harmbench import (
    compute_asr,
    evaluate,
    format_classifier_prompt,
    parse_label,
    run_classifier,
)


class TestFormatClassifierPrompt:
    def test_contains_behavior_and_generation(self) -> None:
        prompt = format_classifier_prompt("do bad thing", "sure here is how")
        assert "do bad thing" in prompt
        assert "sure here is how" in prompt

    def test_uses_llama2_chat_format(self) -> None:
        prompt = format_classifier_prompt("behavior", "generation")
        assert prompt.startswith("[INST] <<SYS>>")
        assert prompt.endswith("Answer: [/INST]")

    def test_contains_classifier_system_prompt(self) -> None:
        prompt = format_classifier_prompt("b", "g")
        assert "classifier for red teaming" in prompt
        assert "your answer should be" in prompt.lower()


class TestParseLabel:
    def test_yes(self) -> None:
        assert parse_label("Yes") == 1

    def test_no(self) -> None:
        assert parse_label("No") == 0

    def test_case_insensitive(self) -> None:
        assert parse_label("YES") == 1
        assert parse_label("no") == 0

    def test_unknown_returns_negative_one(self) -> None:
        assert parse_label("maybe") == -1
        assert parse_label("") == -1


class TestComputeAsr:
    def test_all_successful(self) -> None:
        labels = [1, 1, 1]
        categories = ["a", "a", "a"]
        overall, breakdown = compute_asr(labels, categories)
        assert overall == 1.0
        assert breakdown["a"] == 1.0

    def test_none_successful(self) -> None:
        labels = [0, 0, 0]
        categories = ["a", "a", "a"]
        overall, breakdown = compute_asr(labels, categories)
        assert overall == 0.0

    def test_mixed(self) -> None:
        labels = [1, 0, 1, 0]
        categories = ["a", "a", "b", "b"]
        overall, breakdown = compute_asr(labels, categories)
        assert overall == 0.5
        assert breakdown["a"] == 0.5
        assert breakdown["b"] == 0.5

    def test_ignores_negative_labels(self) -> None:
        labels = [1, -1, 0]
        categories = ["a", "a", "a"]
        overall, breakdown = compute_asr(labels, categories)
        assert overall == 0.5
        assert breakdown["a"] == 0.5

    def test_per_category_breakdown(self) -> None:
        labels = [1, 1, 0, 0, 0]
        categories = ["cyber", "cyber", "violence", "violence", "violence"]
        overall, breakdown = compute_asr(labels, categories)
        assert overall == 2 / 5
        assert breakdown["cyber"] == 1.0
        assert breakdown["violence"] == 0.0


def _make_model_tokenizer(
    decode_values: list[str] | str = "Yes",
) -> tuple[MagicMock, MagicMock]:
    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate.side_effect = lambda input_ids, **kw: torch.cat(
        [input_ids, torch.full((input_ids.shape[0], 1), 99)], dim=1,
    )

    tokenizer = MagicMock()
    tokenizer.side_effect = lambda prompts, **kw: {
        "input_ids": torch.tensor([[1, 2, 3]] * len(prompts)),
        "attention_mask": torch.tensor([[1, 1, 1]] * len(prompts)),
    }
    if isinstance(decode_values, list):
        tokenizer.decode.side_effect = decode_values
    else:
        tokenizer.decode.return_value = decode_values
    return model, tokenizer


class TestRunClassifier:
    def test_returns_labels_for_each_pair(self) -> None:
        model, tokenizer = _make_model_tokenizer("Yes")
        labels = run_classifier(model, tokenizer, ["do X", "do Y"], ["sure", "nope"], batch_size=2)
        assert labels == [1, 1]

    def test_greedy_decoding(self) -> None:
        model, tokenizer = _make_model_tokenizer("No")
        labels = run_classifier(model, tokenizer, ["b"], ["r"], batch_size=1)
        assert labels == [0]

        generate_kwargs = model.generate.call_args[1]
        assert generate_kwargs["max_new_tokens"] == 1
        assert generate_kwargs["do_sample"] is False


class TestEndToEnd:
    def test_reads_jsonl_and_writes_results(self, tmp_path: Path) -> None:
        input_data = [
            {"behavior": "do X", "category": "cyber"},
            {"behavior": "do Y", "category": "violence"},
        ]
        response_data = [
            {"behavior": "do X", "response": "sure"},
            {"behavior": "do Y", "response": "I cannot"},
        ]

        with open(tmp_path / "input_data.jsonl", "w") as f:
            for row in input_data:
                f.write(json.dumps(row) + "\n")

        with open(tmp_path / "responses.jsonl", "w") as f:
            for row in response_data:
                f.write(json.dumps(row) + "\n")

        model, tokenizer = _make_model_tokenizer(["Yes", "No"])

        with patch("scripts.evaluate_harmbench.load_classifier", return_value=(model, tokenizer)):
            results = evaluate(str(tmp_path))

        assert "overall_asr" in results
        assert "per_category" in results
        assert results["total"] == 2

        results_file = tmp_path / "eval_results.json"
        assert results_file.exists()


class TestShardedBackfill:
    def _make_run_dir(self, gen_base: Path, run_id: str) -> Path:
        run_dir = gen_base / run_id
        run_dir.mkdir(parents=True)
        with open(run_dir / "input_data.jsonl", "w") as f:
            f.write(json.dumps({"behavior": "b", "category": "c"}) + "\n")
        with open(run_dir / "responses.jsonl", "w") as f:
            f.write(json.dumps({"behavior": "b", "response": "r"}) + "\n")
        return run_dir

    def test_shard_mode_writes_delta_and_does_not_modify_results(
        self, tmp_path: Path
    ) -> None:
        from scripts.evaluate_harmbench import evaluate_results_file

        gen_base = tmp_path / "gens"
        self._make_run_dir(gen_base, "id-a")
        self._make_run_dir(gen_base, "id-b")

        results_file = tmp_path / "results.jsonl"
        with open(results_file, "w") as f:
            for rid in ("id-a", "id-b"):
                f.write(json.dumps(
                    {"eval_run_id": rid, "dataset_name": "advbench_harmbench", "asr": None}
                ) + "\n")

        shard_file = tmp_path / "shard.jsonl"
        model, tokenizer = _make_model_tokenizer("Yes")
        with patch(
            "scripts.evaluate_harmbench.load_classifier",
            return_value=(model, tokenizer),
        ):
            evaluate_results_file(
                str(results_file),
                generations_base=str(gen_base),
                shard_index=0,
                num_shards=1,
                output_shard_file=str(shard_file),
            )

        with open(results_file) as f:
            updated = [json.loads(line) for line in f]
        assert all(r["asr"] is None for r in updated)

        with open(shard_file) as f:
            delta = [json.loads(line) for line in f]
        ids = {d["eval_run_id"]: d["asr"] for d in delta}
        assert set(ids.keys()) == {"id-a", "id-b"}
        for asr in ids.values():
            assert asr == 1.0


class TestParetoFiltering:
    def test_canonical_asr_only_uses_pareto_step_0_rows(self, tmp_path: Path) -> None:
        from scripts.evaluate_harmbench import evaluate_results_file

        gen_base = tmp_path / "generations"
        gen_base.mkdir()
        run_dir = gen_base / "run-1"
        run_dir.mkdir()

        input_data = [
            {"behavior": "do X", "category": "cyber", "pareto_step_idx": 0},
            {"behavior": "do X", "category": "cyber", "pareto_step_idx": 1},
            {"behavior": "do Y", "category": "violence", "pareto_step_idx": 0},
            {"behavior": "do Y", "category": "violence", "pareto_step_idx": 1},
        ]
        response_data = [
            {"behavior": "do X", "response": "r0", "pareto_step_idx": 0},
            {"behavior": "do X", "response": "r1", "pareto_step_idx": 1},
            {"behavior": "do Y", "response": "r0", "pareto_step_idx": 0},
            {"behavior": "do Y", "response": "r1", "pareto_step_idx": 1},
        ]
        with open(run_dir / "input_data.jsonl", "w") as f:
            for row in input_data:
                f.write(json.dumps(row) + "\n")
        with open(run_dir / "responses.jsonl", "w") as f:
            for row in response_data:
                f.write(json.dumps(row) + "\n")

        results_file = tmp_path / "results.jsonl"
        rows = [
            {
                "eval_run_id": "run-1",
                "dataset_name": "advbench_harmbench",
                "asr": None,
            }
        ]
        with open(results_file, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        # Classifier returns: do X step0 -> Yes, do X step1 -> No,
        # do Y step0 -> No, do Y step1 -> Yes.
        # Pareto_step_idx==0 rows have labels [1, 0] -> mean 0.5.
        # All rows have labels [1, 0, 0, 1] -> mean 0.5. Pick distinct values to verify
        # the filter. Make step1 labels both Yes -> all=[1,1,1,1] mean 1.0, step0=[1,0]=0.5.
        model, tokenizer = _make_model_tokenizer(["Yes", "Yes", "No", "Yes"])

        with patch(
            "scripts.evaluate_harmbench.load_classifier",
            return_value=(model, tokenizer),
        ):
            evaluate_results_file(str(results_file), generations_base=str(gen_base))

        with open(results_file) as f:
            updated = [json.loads(line) for line in f]

        assert updated[0]["asr"] == 0.5

    def test_per_row_labels_block_emitted_in_eval_results_json(
        self, tmp_path: Path
    ) -> None:
        input_data = [
            {"behavior": "b", "category": "c", "pareto_step_idx": 0},
            {"behavior": "b", "category": "c", "pareto_step_idx": 1},
        ]
        response_data = [
            {"behavior": "b", "response": "r0", "pareto_step_idx": 0, "dependent_flops": 100},
            {"behavior": "b", "response": "r1", "pareto_step_idx": 1, "dependent_flops": 50},
        ]
        with open(tmp_path / "input_data.jsonl", "w") as f:
            for row in input_data:
                f.write(json.dumps(row) + "\n")
        with open(tmp_path / "responses.jsonl", "w") as f:
            for row in response_data:
                f.write(json.dumps(row) + "\n")

        model, tokenizer = _make_model_tokenizer(["Yes", "No"])
        with patch(
            "scripts.evaluate_harmbench.load_classifier",
            return_value=(model, tokenizer),
        ):
            evaluate(str(tmp_path))

        results_path = tmp_path / "eval_results.json"
        with open(results_path) as f:
            blob = json.load(f)
        assert "per_row_labels" in blob
        assert len(blob["per_row_labels"]) == 2
        idxs = sorted(r["pareto_step_idx"] for r in blob["per_row_labels"])
        assert idxs == [0, 1]
        for r in blob["per_row_labels"]:
            assert "behavior" in r
            assert "label" in r
            assert "attack_flops" in r
