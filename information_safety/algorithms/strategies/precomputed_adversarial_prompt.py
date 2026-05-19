"""Prompt-based strategies that inject precomputed per-behavior adversarial prompts.

Each attack (GCG, AutoDAN, ...) runs upstream (e.g. via AdversariaLLM), emits a flat
JSONL with `{behavior, adversarial_prompt, attack_flops, attack_bits, pareto_step_idx
[, model_completion]}`, and is replayed here at eval time. The base class is
placement-agnostic (it replaces the whole user turn), so it works for suffix- and
prefix-placed attacks alike. Each concrete attack lives in its own subclass so plots see
a distinct `experiment_name`.

For GCG the upstream extractor may emit multiple rows per behavior (one per Pareto-optimal
optimization step). The strategy then fans each behavior out to all of its Pareto rows at
eval time. AutoDAN and PAIR remain single-row-per-behavior; their subclass rejects multi-row
inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
from transformers import PreTrainedTokenizerBase

from information_safety.algorithms.dataset_handlers.base import BaseDatasetHandler
from information_safety.algorithms.strategies.base import BaseStrategy


def _normalize_behavior(behavior: str) -> str:
    return " ".join(behavior.split()).lower()


@dataclass
class PrecomputedAdversarialPromptStrategy(BaseStrategy):
    """Inject per-behavior adversarial prompts loaded from a JSONL file.

    The JSONL must contain one or more rows per behavior. Each row has
    `behavior`, `adversarial_prompt`, `attack_flops`, `attack_bits`, and
    `pareto_step_idx`. Optionally a row may include `model_completion` (the
    attack's stored greedy completion); when present, eval-time generation is
    skipped and the stored response is recorded directly. Every behavior in the
    validation dataset must be present in the file. Behaviors are matched on a
    whitespace- and case-normalized key.

    `attack_bits` is the per-config program-length accounting emitted by
    `scripts/extract_adversariallm_attacks.py`. It is a single config-level
    constant (the same value on every row); we read it once at setup time and
    raise if rows disagree.

    `attack_flops` is per-row: for GCG Pareto rows it is cumulative FLOPs through
    that step.
    """

    requires_training_data: ClassVar[bool] = False
    allow_multirow_per_behavior: ClassVar[bool] = False
    suffix_file: str
    label_behavior_key: str = "behavior"
    _lookup: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict, repr=False
    )
    _attack_bits: int = 0

    def setup(
        self,
        model: Any,
        handler: BaseDatasetHandler,
        pl_module: Any,
    ) -> Any:
        tokenizer: PreTrainedTokenizerBase = pl_module.tokenizer

        attack_bits_seen: int | None = None
        with open(self.suffix_file) as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                key = _normalize_behavior(row["behavior"])
                row_bits = row["attack_bits"]
                if attack_bits_seen is None:
                    attack_bits_seen = row_bits
                elif row_bits != attack_bits_seen:
                    raise ValueError(
                        f"attack_bits mismatch in {self.suffix_file}: "
                        f"{row_bits} != {attack_bits_seen}"
                    )
                self._lookup.setdefault(key, []).append(row)

        if attack_bits_seen is None:
            raise ValueError(f"No rows found in {self.suffix_file}")
        self._attack_bits = attack_bits_seen

        for key, rows in self._lookup.items():
            rows.sort(key=lambda r: r.get("pareto_step_idx", 0))
            if not self.allow_multirow_per_behavior and len(rows) > 1:
                raise ValueError(
                    f"{type(self).__name__} expected one row per behavior; "
                    f"got {len(rows)} rows for {key!r} (multi-row/multiple rows "
                    f"are GCG-only)"
                )
            if len(rows) > 1:
                missing = [
                    r["pareto_step_idx"] for r in rows if "model_completion" not in r
                ]
                if missing:
                    raise ValueError(
                        f"multi-row path requires model_completion on every row; "
                        f"missing on pareto_step_idx={missing} for behavior {key!r}"
                    )

        val_dataset = handler.get_val_dataset(tokenizer)
        missing = 0
        for item in val_dataset:
            meta = json.loads(item["labels"])
            if _normalize_behavior(meta[self.label_behavior_key]) not in self._lookup:
                missing += 1

        if missing:
            raise ValueError(
                f"Missing {missing} behavior(s) in suffix_file {self.suffix_file}"
            )

        return model

    def compute_bits(self, model: Any) -> int:
        return self._attack_bits

    def configure_optimizers(self, model: Any) -> torch.optim.AdamW:
        dummy = torch.nn.Parameter(torch.empty(0), requires_grad=True)
        return torch.optim.AdamW([dummy])

    def attack_flops_for(self, meta: dict[str, Any]) -> int:
        key = _normalize_behavior(meta[self.label_behavior_key])
        pareto_step_idx = meta.get("pareto_step_idx", 0)
        for row in self._lookup[key]:
            if row.get("pareto_step_idx", 0) == pareto_step_idx:
                return row["attack_flops"]
        raise KeyError(
            f"no row for behavior {key!r} at pareto_step_idx={pareto_step_idx}"
        )

    def validation_step(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizerBase,
        batch: dict[str, Any],
        handler: BaseDatasetHandler,
    ) -> tuple[int, int]:
        metadata_strs: list[str] = batch["labels"]

        new_prompts: list[str] = []
        kept_labels: list[str] = []
        stored_labels: list[str] = []
        stored_completions: list[str] = []
        for meta_str in metadata_strs:
            meta = json.loads(meta_str)
            key = _normalize_behavior(meta[self.label_behavior_key])

            rows = self._lookup[key]
            for row in rows:
                pareto_step_idx = row.get("pareto_step_idx", 0)
                meta_with_idx = {**meta, "pareto_step_idx": pareto_step_idx}
                label_with_idx = json.dumps(meta_with_idx)

                if "model_completion" in row:
                    stored_labels.append(label_with_idx)
                    stored_completions.append(row["model_completion"])
                    continue

                adversarial_prompt = row["adversarial_prompt"]
                prompt_text: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
                    [{"role": "user", "content": adversarial_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                new_prompts.append(prompt_text)
                kept_labels.append(label_with_idx)

        correct_total = 0
        total_total = 0

        if stored_completions:
            correct, total = handler.record_precomputed_completions(
                stored_labels, stored_completions, self,
            )
            correct_total += correct
            total_total += total

        if new_prompts:
            max_length: int = handler.max_length
            encoded = tokenizer(
                new_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            device = batch["input_ids"].device
            input_ids = torch.as_tensor(encoded["input_ids"]).to(device)
            attention_mask = torch.as_tensor(encoded["attention_mask"]).to(device)
            modified_batch: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": kept_labels,
            }

            correct, total = handler.validate_batch(
                model, tokenizer, modified_batch, self,
            )
            correct_total += correct
            total_total += total

        return correct_total, total_total


@dataclass
class PrecomputedGCGStrategy(PrecomputedAdversarialPromptStrategy):
    """GCG (Zou et al., 2023): suffix-placed, discrete-optimization adversarial prompt."""

    allow_multirow_per_behavior: ClassVar[bool] = True


@dataclass
class PrecomputedAutoDANStrategy(PrecomputedAdversarialPromptStrategy):
    """AutoDAN (Liu et al., 2023): prefix-placed, genetic-algorithm adversarial prompt."""


@dataclass
class PrecomputedPAIRStrategy(PrecomputedAdversarialPromptStrategy):
    """PAIR (Chao et al., 2023): iterative-refinement adversarial prompt."""
