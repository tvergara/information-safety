from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("vllm")

from scripts.evaluate_mmlu import (  # noqa: E402
    CHOICES,
    build_prompts,
    compute_metrics,
    evaluate_mmlu,
    format_fewshot_prompt,
    format_question,
    main,
    parse_answer,
)


def _mk_item(qid: int, subject: str = "test_subject", answer: int = 0) -> dict:
    return {
        "question": f"q{qid}",
        "choices": [f"a{qid}", f"b{qid}", f"c{qid}", f"d{qid}"],
        "answer": answer,
        "subject": subject,
    }


class TestFormatQuestion:
    def test_includes_all_four_choices(self) -> None:
        out = format_question(_mk_item(1))
        for choice_letter in CHOICES:
            assert f"{choice_letter}. " in out

    def test_ends_with_answer_marker(self) -> None:
        assert format_question(_mk_item(1)).rstrip().endswith("Answer:")


class TestFormatFewshotPrompt:
    def test_includes_subject_header(self) -> None:
        dev = [_mk_item(i) for i in range(5)]
        prompt = format_fewshot_prompt(_mk_item(99), dev, "high_school_biology")
        assert "high school biology" in prompt

    def test_uses_first_n_dev_examples(self) -> None:
        dev = [_mk_item(i) for i in range(1000, 1010)]
        prompt = format_fewshot_prompt(_mk_item(99), dev, "test_subject", num_shots=3)
        for i in range(1000, 1003):
            assert f"q{i}" in prompt
        for i in range(1003, 1010):
            assert f"q{i}" not in prompt

    def test_query_is_last_block(self) -> None:
        dev = [_mk_item(i) for i in range(5)]
        prompt = format_fewshot_prompt(_mk_item(99), dev, "test_subject")
        assert prompt.rstrip().endswith("Answer:")
        assert prompt.count("q99") == 1
        assert prompt.rfind("q99") > prompt.rfind("q4")

    def test_dev_answers_inlined(self) -> None:
        dev = [_mk_item(i, answer=i % 4) for i in range(5)]
        prompt = format_fewshot_prompt(_mk_item(99), dev, "test_subject")
        for i in range(5):
            assert f"Answer: {CHOICES[i % 4]}" in prompt


class TestParseAnswer:
    def test_bare_letter(self) -> None:
        assert parse_answer(" A") == "A"

    def test_first_letter_in_words(self) -> None:
        assert parse_answer("The answer is B.") == "B"

    def test_returns_none_when_no_choice(self) -> None:
        assert parse_answer("I cannot help with that.") is None

    def test_takes_earliest_match(self) -> None:
        assert parse_answer("C is wrong, D is right") == "C"


class TestComputeMetrics:
    def test_overall_accuracy(self) -> None:
        metrics = compute_metrics(["A", "B", "C"], [0, 1, 0], ["s", "s", "s"])
        assert metrics["correct"] == 2
        assert metrics["total"] == 3
        assert metrics["accuracy"] == pytest.approx(2 / 3)

    def test_format_compliance_counts_parsable(self) -> None:
        metrics = compute_metrics(["A", None, "C"], [0, 1, 0], ["s", "s", "s"])
        assert metrics["parsable"] == 2
        assert metrics["format_compliance"] == pytest.approx(2 / 3)

    def test_per_subject_breakdown(self) -> None:
        metrics = compute_metrics(
            ["A", "B", "A", "B"],
            [0, 1, 1, 1],
            ["bio", "bio", "math", "math"],
        )
        assert metrics["per_subject"]["bio"]["accuracy"] == pytest.approx(1.0)
        assert metrics["per_subject"]["math"]["accuracy"] == pytest.approx(0.5)

    def test_unparsable_counts_as_wrong(self) -> None:
        metrics = compute_metrics([None], [0], ["s"])
        assert metrics["correct"] == 0
        assert metrics["accuracy"] == 0.0


class TestBuildPrompts:
    def test_one_prompt_per_test_item(self) -> None:
        test = [_mk_item(i, subject="s") for i in range(4)]
        dev = {"s": [_mk_item(j, subject="s") for j in range(5)]}
        prompts = build_prompts(test, dev, num_shots=5)
        assert len(prompts) == 4

    def test_uses_subject_specific_dev_set(self) -> None:
        test = [_mk_item(99, subject="bio")]
        dev = {
            "bio": [_mk_item(i, subject="bio") for i in range(5)],
            "math": [_mk_item(100 + i, subject="math") for i in range(5)],
        }
        prompts = build_prompts(test, dev, num_shots=5)
        for i in range(5):
            assert f"q{i}" in prompts[0]
        for i in range(100, 105):
            assert f"q{i}" not in prompts[0]


class TestEvaluateMmlu:
    @patch("scripts.evaluate_mmlu.LLM")
    @patch("scripts.evaluate_mmlu.datasets")
    @patch("scripts.evaluate_mmlu.torch")
    def test_writes_json_with_metrics(
        self,
        mock_torch: MagicMock,
        mock_datasets: MagicMock,
        mock_llm_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        test_items = [_mk_item(i, subject="s", answer=0) for i in range(3)]
        dev_items = [_mk_item(j, subject="s") for j in range(5)]
        mock_datasets.load_dataset.side_effect = [test_items, dev_items]
        mock_torch.cuda.device_count.return_value = 1

        mock_llm = MagicMock()
        completion = MagicMock()
        completion.outputs = [MagicMock(text=" A")]
        mock_llm.generate.return_value = [completion, completion, completion]
        mock_llm_cls.return_value = mock_llm

        out_path = tmp_path / "mmlu_eval.json"
        metrics = evaluate_mmlu(
            model_name_or_path="fake/model",
            output_path=out_path,
            num_shots=5,
        )

        assert out_path.exists()
        saved = json.loads(out_path.read_text())
        assert saved["accuracy"] == pytest.approx(1.0)
        assert saved["total"] == 3
        assert saved["model"] == "fake/model"
        assert saved["num_shots"] == 5
        assert metrics["correct"] == 3


class TestMainCli:
    @patch("scripts.evaluate_mmlu.evaluate_mmlu")
    def test_passes_args_through(self, mock_eval: MagicMock, tmp_path: Path) -> None:
        mock_eval.return_value = {
            "accuracy": 0.5,
            "correct": 1,
            "total": 2,
            "format_compliance": 1.0,
        }
        out_path = tmp_path / "x.json"
        main([
            "--model-name-or-path", "foo/bar",
            "--output-path", str(out_path),
            "--max-examples", "10",
            "--num-shots", "3",
        ])
        kwargs = mock_eval.call_args.kwargs
        assert kwargs["model_name_or_path"] == "foo/bar"
        assert kwargs["output_path"] == out_path
        assert kwargs["max_examples"] == 10
        assert kwargs["num_shots"] == 3
