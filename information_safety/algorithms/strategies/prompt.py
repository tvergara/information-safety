"""Prompt-based attack strategies: no training, just prompt injection at eval time."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from transformers import PreTrainedTokenizerBase

from information_safety.algorithms.dataset_handlers.base import BaseDatasetHandler
from information_safety.algorithms.jailbreak.prompt_methods import _build_roleplay_prompt
from information_safety.algorithms.strategies.base import BaseStrategy
from information_safety.attack_bits import STRATEGY_CODE_FILES, code_bits


@dataclass
class RoleplayStrategy(BaseStrategy):
    """Prompt-based attack strategy that wraps behaviors in a roleplay template.

    No training is performed. At validation time, the behavior is extracted from batch metadata and
    wrapped in a roleplay prompt before generation.
    """

    requires_training_data: ClassVar[bool] = False
    scenario: str = "an unrestricted assistant in a red-team evaluation"
    _tokenizer: Any = None

    def setup(
        self,
        model: Any,
        handler: BaseDatasetHandler,
        pl_module: Any,
    ) -> Any:
        self._tokenizer = pl_module.tokenizer
        return model

    def compute_bits(self, model: Any) -> int:
        return code_bits(STRATEGY_CODE_FILES["roleplay"])

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
        """Re-tokenize prompts with roleplay template wrapping before generation."""
        metadata_strs: list[str] = batch["labels"]

        new_prompts: list[str] = []
        for meta_str in metadata_strs:
            meta = json.loads(meta_str)
            behavior = meta["behavior"]
            roleplay_prompt = _build_roleplay_prompt(behavior, scenario=self.scenario)

            prompt_text: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
                [{"role": "user", "content": roleplay_prompt}],
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
