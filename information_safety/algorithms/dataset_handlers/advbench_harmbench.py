"""AdvBench/HarmBench dataset handler: trains on AdvBench, evaluates on HarmBench."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any

import datasets
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from information_safety.algorithms.dataset_handlers.base import (
    BaseDatasetHandler,
    GenerationValDataset,
)

logger = getLogger(__name__)


@dataclass
class AdvBenchHarmBenchHandler(BaseDatasetHandler):
    """Dataset handler for AdvBench training and HarmBench evaluation.

    Training: loads from a JSONL file (prompt/completion pairs) if train_data_path is set,
    otherwise falls back to walledai/AdvBench from HuggingFace.
    Evaluation: HarmBench (walledai/HarmBench, standard) — 200 harmful behaviors.
    Completions are saved to disk for deferred scoring by a separate evaluation script.
    """

    max_length: int = 500
    batch_size: int = 8
    max_new_tokens: int = 320
    generations_dir: str = ""
    dataset_name: str = ""
    max_examples: int | None = None
    train_data_path: str = ""
    _completions: list[dict[str, str]] = field(default_factory=list, repr=False)

    def get_answer_start_token(self, tokenizer: PreTrainedTokenizerBase) -> str:
        """Detect the assistant turn marker from the tokenizer's chat template."""
        sentinel = "SENTINEL_ANSWER_START"
        text: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
            [{"role": "user", "content": "Q"}, {"role": "assistant", "content": sentinel}],
            tokenize=False,
        )
        idx = text.index(sentinel)
        prefix = text[:idx]
        for line in reversed(prefix.split("\n")):
            if line.strip():
                return line
        raise ValueError(f"Could not detect assistant marker from chat template: {text!r}")

    def get_train_dataset(self, tokenizer: PreTrainedTokenizerBase) -> Any:
        """Load and tokenize training data using the model's chat template.

        If train_data_path is set, loads from a JSONL file with prompt/completion fields. Otherwise
        falls back to walledai/AdvBench from HuggingFace.
        """
        if self.train_data_path:
            raw = datasets.load_dataset("json", data_files=self.train_data_path, split="train")
            prompt_field, response_field = "prompt", "completion"
        else:
            raw = datasets.load_dataset("walledai/AdvBench", split="train")
            prompt_field, response_field = "prompt", "target"

        if self.max_examples is not None:
            raw = raw.select(range(self.max_examples))  # type: ignore[union-attr]

        def tokenize_fn(example: dict[str, str]) -> Any:
            text: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
                [
                    {"role": "user", "content": example[prompt_field]},
                    {"role": "assistant", "content": example[response_field]},
                ],
                tokenize=False,
            )
            return tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

        tokenized = raw.map(tokenize_fn, remove_columns=raw.column_names)  # type: ignore[union-attr]
        return tokenized

    def get_val_dataset(self, tokenizer: PreTrainedTokenizerBase) -> Dataset:
        """Load HarmBench standard behaviors for evaluation."""
        raw = datasets.load_dataset("walledai/HarmBench", "standard", split="train")

        prompts: list[dict[str, Any]] = []
        labels: list[str] = []

        for example in raw:  # type: ignore[union-attr]
            behavior = example["prompt"]  # type: ignore[index]
            category = example["category"]  # type: ignore[index]

            prompt_text: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
                [{"role": "user", "content": behavior}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = dict(tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            ))
            prompts.append(encoded)
            labels.append(json.dumps({"behavior": behavior, "category": category}))

        return GenerationValDataset(prompts, labels)

    def validate_batch(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizerBase,
        batch: dict,
    ) -> tuple[int, int]:
        """Generate completions with the finetuned model."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        metadata_strs: list[str] = batch["labels"]

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_token_ids = generated_ids[:, input_ids.shape[1]:]
        generated_texts = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)

        for gen_text, meta_str in zip(generated_texts, metadata_strs):
            meta = json.loads(meta_str)
            self._completions.append({
                "behavior": meta["behavior"],
                "category": meta["category"],
                "response": gen_text,
            })

        return 0, len(generated_texts)

    def save_completions(self, eval_run_id: str) -> None:
        """Write accumulated completions to disk as JSONL files."""
        run_dir = Path(self.generations_dir) / eval_run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "input_data.jsonl", "w") as f:
            for entry in self._completions:
                row = {"behavior": entry["behavior"], "category": entry["category"]}
                f.write(json.dumps(row) + "\n")

        with open(run_dir / "responses.jsonl", "w") as f:
            for entry in self._completions:
                row = {"behavior": entry["behavior"], "response": entry["response"]}
                f.write(json.dumps(row) + "\n")

        logger.info(f"Saved {len(self._completions)} completions to {run_dir}")
        self._completions = []
