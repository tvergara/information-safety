"""Prompt-based strategies that inject precomputed per-behavior adversarial prompts.

Each attack (GCG, AutoDAN, ...) runs upstream (e.g. via AdversariaLLM), emits a flat
JSONL with `{behavior, adversarial_prompt, attack_flops}`, and is replayed here at
eval time. The base class is placement-agnostic (it replaces the whole user turn),
so it works for suffix- and prefix-placed attacks alike. Each concrete attack lives
in its own subclass so plots see a distinct `experiment_name`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from information_safety.algorithms.dataset_handlers.base import BaseDatasetHandler
from information_safety.algorithms.strategies.base import BaseStrategy


def _normalize_behavior(behavior: str) -> str:
    return " ".join(behavior.split()).lower()


@dataclass
class PrecomputedAdversarialPromptStrategy(BaseStrategy):
    """Inject a per-behavior adversarial prompt loaded from a JSONL file.

    The JSONL must contain one row per behavior with `behavior` and `adversarial_prompt`
    string fields, plus `attack_flops` (or legacy `gcg_flops`). Every behavior in the
    validation dataset must be present in the file. Behaviors are matched on a
    whitespace- and case-normalized key to absorb minor drift between the HarmBench CSV
    (AdversariaLLM) and HF split (our eval).
    """

    suffix_file: str
    _lookup: dict[str, str] = field(default_factory=dict, repr=False)
    _attack_flops_total: int = 0

    def setup(
        self,
        model: Any,
        handler: BaseDatasetHandler,
        pl_module: Any,
    ) -> Any:
        tokenizer: PreTrainedTokenizerBase = pl_module.tokenizer

        with open(self.suffix_file) as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                self._lookup[_normalize_behavior(row["behavior"])] = row["adversarial_prompt"]
                # TODO(merge-attack-flops-column): drop the `gcg_flops` branch once
                # all legacy GCG attack JSONLs have been re-extracted.
                if "attack_flops" in row:
                    self._attack_flops_total += row["attack_flops"]
                else:
                    self._attack_flops_total += row["gcg_flops"]

        val_dataset = handler.get_val_dataset(tokenizer)
        missing = 0
        for item in val_dataset:
            meta = json.loads(item["labels"])
            if _normalize_behavior(meta["behavior"]) not in self._lookup:
                missing += 1

        if missing:
            raise ValueError(
                f"Missing {missing} behavior(s) in suffix_file {self.suffix_file}"
            )

        return model

    def compute_bits(self, model: Any) -> int:
        return 0

    def attack_flops(self) -> int:
        return self._attack_flops_total

    def configure_optimizers(self, model: Any) -> torch.optim.AdamW:
        dummy = torch.nn.Parameter(torch.empty(0), requires_grad=True)
        return torch.optim.AdamW([dummy])

    def validation_step(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizerBase,
        batch: dict[str, Any],
        handler: BaseDatasetHandler,
    ) -> tuple[int, int]:
        metadata_strs: list[str] = batch["labels"]

        new_prompts: list[str] = []
        for meta_str in metadata_strs:
            meta = json.loads(meta_str)
            adversarial_prompt = self._lookup[_normalize_behavior(meta["behavior"])]

            prompt_text: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
                [{"role": "user", "content": adversarial_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            new_prompts.append(prompt_text)

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
            "labels": batch["labels"],
        }

        return handler.validate_batch(model, tokenizer, modified_batch)


@dataclass
class PrecomputedGCGStrategy(PrecomputedAdversarialPromptStrategy):
    """GCG (Zou et al., 2023): suffix-placed, discrete-optimization adversarial prompt."""


@dataclass
class PrecomputedAutoDANStrategy(PrecomputedAdversarialPromptStrategy):
    """AutoDAN (Liu et al., 2023): prefix-placed, genetic-algorithm adversarial prompt."""


@dataclass
class PrecomputedPAIRStrategy(PrecomputedAdversarialPromptStrategy):
    """PAIR (Chao et al., 2023): iterative-refinement adversarial prompt."""
