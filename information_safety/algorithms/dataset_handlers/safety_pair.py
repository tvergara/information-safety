"""SafetyPairHandler: matched pair of safe/unsafe models via benign + refusal data."""

from __future__ import annotations

import json
import random
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


def get_answer_start_token(tokenizer: PreTrainedTokenizerBase) -> str:
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


@dataclass
class SafetyPairHandler(BaseDatasetHandler):
    """Dataset handler producing matched safe/unsafe training sets.

    When include_refusals=False (Model U): N benign examples only.
    When include_refusals=True (Model S): (N - num_refusals) benign + num_refusals refusal = N total.
    """

    include_refusals: bool = False
    num_benign_examples: int = 10000
    refusal_data_path: str = ""
    harmbench_test_path: str = ""
    max_length: int = 512
    batch_size: int = 8
    generations_dir: str = ""
    dataset_name: str = "safety_pair"
    max_examples: int | None = None
    seed: int = 42
    _completions: list[dict[str, str]] = field(default_factory=list, repr=False)

    def _load_refusal_data(self) -> list[dict[str, str]]:
        """Load circuit_breakers_train.json and extract prompt + llama3_output fields."""
        with open(self.refusal_data_path) as f:
            raw = json.load(f)
        return [{"prompt": ex["prompt"], "llama3_output": ex["llama3_output"]} for ex in raw]

    def _sample_benign_indices(self, dataset_size: int, n: int) -> list[int]:
        """Deterministically sample n indices from a dataset of given size."""
        rng = random.Random(self.seed)
        return rng.sample(range(dataset_size), n)

    def get_answer_start_token(self, tokenizer: PreTrainedTokenizerBase) -> str:
        return get_answer_start_token(tokenizer)

    def get_train_dataset(self, tokenizer: PreTrainedTokenizerBase) -> Any:
        """Load and tokenize training data: benign (SmolTalk) + optional refusal data."""
        refusal_data = self._load_refusal_data()
        num_refusals = len(refusal_data)

        if self.include_refusals:
            num_benign = self.num_benign_examples - num_refusals
        else:
            num_benign = self.num_benign_examples

        raw_benign = datasets.load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
        single_turn = raw_benign.filter(  # type: ignore[union-attr]
            lambda ex: len(ex["messages"]) == 2
            and ex["messages"][0]["role"] == "user"
            and ex["messages"][1]["role"] == "assistant"
        )
        benign_indices = self._sample_benign_indices(len(single_turn), num_benign)  # type: ignore[arg-type]
        benign_subset = single_turn.select(benign_indices)  # type: ignore[union-attr]

        def tokenize_benign(example: dict[str, Any]) -> Any:
            text: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
                example["messages"],
                tokenize=False,
            )
            return tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

        benign_tokenized = benign_subset.map(
            tokenize_benign, remove_columns=benign_subset.column_names
        )

        if not self.include_refusals:
            return benign_tokenized

        refusal_ds = datasets.Dataset.from_list(refusal_data)

        def tokenize_refusal(example: dict[str, str]) -> Any:
            text: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
                [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["llama3_output"]},
                ],
                tokenize=False,
            )
            return tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

        refusal_tokenized = refusal_ds.map(
            tokenize_refusal, remove_columns=refusal_ds.column_names
        )

        combined = datasets.concatenate_datasets([benign_tokenized, refusal_tokenized])
        return combined

    def get_val_dataset(self, tokenizer: PreTrainedTokenizerBase) -> Dataset:
        """Load HarmBench test_std behaviors for evaluation."""
        with open(self.harmbench_test_path) as f:
            raw = json.load(f)

        entries = raw["data"]

        prompts: list[dict[str, Any]] = []
        labels: list[str] = []

        for entry in entries:
            behavior = entry["behavior"]

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
            labels.append(json.dumps({"behavior": behavior}))

        return GenerationValDataset(prompts, labels)

    def validate_batch(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizerBase,
        batch: dict,
    ) -> tuple[int, int]:
        """Two-stage generation: finetuned model generates prefix, base model continues."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        metadata_strs: list[str] = batch["labels"]

        with torch.no_grad():
            prefix_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id,
            )

            prefix_mask = (prefix_ids != tokenizer.pad_token_id).long()

            model.disable_adapter_layers()
            generated_ids = model.generate(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
            )
            model.enable_adapter_layers()

        new_token_ids = generated_ids[:, input_ids.shape[1]:]
        generated_texts = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)

        for gen_text, meta_str in zip(generated_texts, metadata_strs):
            meta = json.loads(meta_str)
            self._completions.append({
                "behavior": meta["behavior"],
                "response": gen_text,
            })

        return 0, len(generated_texts)

    def save_completions(self, eval_run_id: str) -> None:
        """Write accumulated completions to disk as JSONL files."""
        run_dir = Path(self.generations_dir) / eval_run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "input_data.jsonl", "w") as f:
            for entry in self._completions:
                row = {"behavior": entry["behavior"]}
                f.write(json.dumps(row) + "\n")

        with open(run_dir / "responses.jsonl", "w") as f:
            for entry in self._completions:
                row = {"behavior": entry["behavior"], "response": entry["response"]}
                f.write(json.dumps(row) + "\n")

        logger.info(f"Saved {len(self._completions)} completions to {run_dir}")
        self._completions = []
