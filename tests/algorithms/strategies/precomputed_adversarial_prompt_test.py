"""Tests for the precomputed adversarial prompt strategies (GCG, AutoDAN, PAIR)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from information_safety.algorithms.strategies.precomputed_adversarial_prompt import (
    PrecomputedAutoDANStrategy,
    PrecomputedGCGStrategy,
)


def _make_suffix_file(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "suffixes.jsonl"
    with open(path, "w") as f:
        for entry in entries:
            row = {"attack_flops": 0, **entry}
            f.write(json.dumps(row) + "\n")
    return path


def _make_val_dataset(behaviors: list[str]) -> list[dict[str, str]]:
    return [{"labels": json.dumps({"behavior": b, "category": "c"})} for b in behaviors]


class TestSetup:
    def test_setup_loads_suffix_file(self, tmp_path: Path) -> None:
        suffix_path = _make_suffix_file(
            tmp_path,
            [
                {"behavior": "beh_A", "adversarial_prompt": "prompt_A"},
                {"behavior": "beh_B", "adversarial_prompt": "prompt_B"},
            ],
        )
        strategy = PrecomputedGCGStrategy(suffix_file=str(suffix_path))

        model = MagicMock()
        tokenizer = MagicMock()
        pl_module = MagicMock()
        pl_module.tokenizer = tokenizer

        handler = MagicMock()
        handler.get_val_dataset.return_value = _make_val_dataset(["beh_A", "beh_B"])

        result = strategy.setup(model, handler, pl_module)

        assert result is model
        assert len(strategy._lookup) == 2
        assert strategy._lookup["beh_a"] == "prompt_A"
        assert strategy._lookup["beh_b"] == "prompt_B"

    def test_setup_raises_on_missing_behavior(self, tmp_path: Path) -> None:
        suffix_path = _make_suffix_file(
            tmp_path,
            [{"behavior": "beh_A", "adversarial_prompt": "prompt_A"}],
        )
        strategy = PrecomputedGCGStrategy(suffix_file=str(suffix_path))

        model = MagicMock()
        tokenizer = MagicMock()
        pl_module = MagicMock()
        pl_module.tokenizer = tokenizer

        handler = MagicMock()
        handler.get_val_dataset.return_value = _make_val_dataset(["beh_A", "beh_missing"])

        with pytest.raises(ValueError, match="Missing 1 behavior"):
            strategy.setup(model, handler, pl_module)

    def test_setup_matches_across_whitespace_and_case(self, tmp_path: Path) -> None:
        """Behavior strings differ in whitespace/case between the CSV AdversariaLLM reads and the
        HF split our eval reads for 1/200 HarmBench standard rows.

        Matching must ignore those differences.
        """
        suffix_path = _make_suffix_file(
            tmp_path,
            [
                {"behavior": "  Beh_A  with   spaces", "adversarial_prompt": "prompt_A"},
            ],
        )
        strategy = PrecomputedGCGStrategy(suffix_file=str(suffix_path))

        model = MagicMock()
        tokenizer = MagicMock()
        pl_module = MagicMock()
        pl_module.tokenizer = tokenizer

        handler = MagicMock()
        handler.get_val_dataset.return_value = _make_val_dataset(
            ["beh_a with spaces"]
        )

        strategy.setup(model, handler, pl_module)


class TestComputeBits:
    def test_returns_zero(self, tmp_path: Path) -> None:
        suffix_path = _make_suffix_file(tmp_path, [])
        strategy = PrecomputedGCGStrategy(suffix_file=str(suffix_path))
        assert strategy.compute_bits(MagicMock()) == 0


class TestAttackFlops:
    def test_sums_attack_flops_across_rows(self, tmp_path: Path) -> None:
        suffix_path = _make_suffix_file(
            tmp_path,
            [
                {"behavior": "beh_A", "adversarial_prompt": "prompt_A", "attack_flops": 100},
                {"behavior": "beh_B", "adversarial_prompt": "prompt_B", "attack_flops": 250},
            ],
        )
        strategy = PrecomputedGCGStrategy(suffix_file=str(suffix_path))

        model = MagicMock()
        pl_module = MagicMock()
        pl_module.tokenizer = MagicMock()
        handler = MagicMock()
        handler.get_val_dataset.return_value = _make_val_dataset(["beh_A", "beh_B"])
        strategy.setup(model, handler, pl_module)

        assert strategy.attack_flops() == 350

    def test_attack_flops_reads_legacy_gcg_flops_key(self, tmp_path: Path) -> None:
        """Legacy JSONL files only have `gcg_flops`; fallback must read them correctly."""
        path = tmp_path / "legacy.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({
                "behavior": "beh_A",
                "adversarial_prompt": "prompt_A",
                "gcg_flops": 100,
            }) + "\n")
            f.write(json.dumps({
                "behavior": "beh_B",
                "adversarial_prompt": "prompt_B",
                "gcg_flops": 250,
            }) + "\n")

        strategy = PrecomputedGCGStrategy(suffix_file=str(path))

        model = MagicMock()
        pl_module = MagicMock()
        pl_module.tokenizer = MagicMock()
        handler = MagicMock()
        handler.get_val_dataset.return_value = _make_val_dataset(["beh_A", "beh_B"])
        strategy.setup(model, handler, pl_module)

        assert strategy.attack_flops() == 350


class TestConfigureOptimizers:
    def test_returns_dummy_optimizer(self, tmp_path: Path) -> None:
        suffix_path = _make_suffix_file(tmp_path, [])
        strategy = PrecomputedGCGStrategy(suffix_file=str(suffix_path))
        optimizer = strategy.configure_optimizers(MagicMock())
        assert isinstance(optimizer, torch.optim.AdamW)
        params = list(optimizer.param_groups[0]["params"])
        assert len(params) == 1
        assert params[0].numel() == 0


class TestValidationStep:
    def test_validation_step_rewrites_prompts(self, tmp_path: Path) -> None:
        suffix_path = _make_suffix_file(
            tmp_path,
            [
                {"behavior": "beh_A", "adversarial_prompt": "prompt_A"},
                {"behavior": "beh_B", "adversarial_prompt": "prompt_B"},
            ],
        )
        strategy = PrecomputedGCGStrategy(suffix_file=str(suffix_path))

        model = MagicMock()
        tokenizer = MagicMock()
        pl_module = MagicMock()
        pl_module.tokenizer = tokenizer
        handler_setup = MagicMock()
        handler_setup.get_val_dataset.return_value = _make_val_dataset(["beh_A", "beh_B"])
        strategy.setup(model, handler_setup, pl_module)

        metadata_a = json.dumps({"behavior": "beh_A", "category": "c"})
        metadata_b = json.dumps({"behavior": "beh_B", "category": "c"})
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "labels": [metadata_a, metadata_b],
        }

        tokenizer.apply_chat_template.return_value = "formatted prompt"
        tokenizer.return_value = {
            "input_ids": torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]),
        }
        tokenizer.pad_token_id = 0

        handler = MagicMock()
        handler.validate_batch.return_value = (0, 2)
        handler.max_length = 500

        correct, total = strategy.validation_step(model, tokenizer, batch, handler)

        assert (correct, total) == (0, 2)
        handler.validate_batch.assert_called_once()
        modified_batch = handler.validate_batch.call_args[0][2]
        assert "input_ids" in modified_batch
        assert "attention_mask" in modified_batch
        assert "labels" in modified_batch
        assert modified_batch["input_ids"].shape[0] == 2

        assert tokenizer.apply_chat_template.call_count == 2
        calls = tokenizer.apply_chat_template.call_args_list
        contents = [
            (call.args[0] if call.args else call.kwargs["conversation"])[0]["content"]
            for call in calls
        ]
        assert "prompt_A" in contents
        assert "prompt_B" in contents

    def test_validation_step_matches_across_whitespace_and_case(
        self, tmp_path: Path
    ) -> None:
        suffix_path = _make_suffix_file(
            tmp_path,
            [{"behavior": "  Beh_A  with   spaces", "adversarial_prompt": "prompt_A"}],
        )
        strategy = PrecomputedGCGStrategy(suffix_file=str(suffix_path))

        model = MagicMock()
        tokenizer = MagicMock()
        pl_module = MagicMock()
        pl_module.tokenizer = tokenizer
        handler_setup = MagicMock()
        handler_setup.get_val_dataset.return_value = _make_val_dataset(
            ["beh_a with spaces"]
        )
        strategy.setup(model, handler_setup, pl_module)

        metadata = json.dumps({"behavior": "beh_a with spaces", "category": "c"})
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": [metadata],
        }
        tokenizer.apply_chat_template.return_value = "formatted"
        tokenizer.return_value = {
            "input_ids": torch.tensor([[10, 20]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        tokenizer.pad_token_id = 0

        handler = MagicMock()
        handler.validate_batch.return_value = (0, 1)
        handler.max_length = 500

        strategy.validation_step(model, tokenizer, batch, handler)

        call = tokenizer.apply_chat_template.call_args_list[0]
        content = (call.args[0] if call.args else call.kwargs["conversation"])[0][
            "content"
        ]
        assert content == "prompt_A"


class TestSubclassing:
    def test_autodan_and_gcg_have_distinct_class_names(self, tmp_path: Path) -> None:
        suffix_path = _make_suffix_file(tmp_path, [])
        gcg = PrecomputedGCGStrategy(suffix_file=str(suffix_path))
        autodan = PrecomputedAutoDANStrategy(suffix_file=str(suffix_path))
        assert type(gcg).__name__ != type(autodan).__name__
        assert type(gcg).__name__ == "PrecomputedGCGStrategy"
        assert type(autodan).__name__ == "PrecomputedAutoDANStrategy"
