"""MetaMath dataset handler: trains on MetaMathQA, validates on GSM8K."""

from __future__ import annotations

import re
from dataclasses import dataclass
from logging import getLogger
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

ANSWER_PATTERN = re.compile(r"####\s*(-?[\d.,]+)")


def format_example(example: dict[str, str]) -> str:
    """Format a MetaMath example into a prompt-completion string."""
    return (
        f"### Question: {example['query']}\n"
        f"### Answer: {example['response']}"
    )


def extract_numerical_answer(text: str) -> str | None:
    """Extract the numerical answer after #### from model output."""
    match = ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).replace(",", "")
    return None


@dataclass
class MetaMathHandler(BaseDatasetHandler):
    """Dataset handler for MetaMath training and GSM8K validation."""

    max_length: int = 500
    batch_size: int = 8
    dataset_name: str = ""
    max_examples: int | None = None

    def get_train_dataset(self, tokenizer: PreTrainedTokenizerBase) -> Any:
        """Load and tokenize MetaMathQA training data."""
        raw = datasets.load_dataset("meta-math/MetaMathQA", split="train")
        eos = str(tokenizer.eos_token)

        def tokenize_fn(example: dict[str, str]) -> Any:
            text = format_example(example) + eos
            return tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

        tokenized = raw.map(tokenize_fn, remove_columns=raw.column_names)  # type: ignore[union-attr]
        return tokenized

    def get_val_dataset(self, tokenizer: PreTrainedTokenizerBase) -> Dataset:
        """Load GSM8K test set for validation."""
        raw = datasets.load_dataset("openai/gsm8k", "main", split="test")

        prompts: list[Any] = []
        answers: list[str] = []

        for example in raw:  # type: ignore[union-attr]
            question = example["question"]  # type: ignore[index]
            prompt_text = f"### Question: {question}\n### Answer:"
            encoded = tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )
            prompts.append(encoded)

            answer_text = example["answer"]  # type: ignore[index]
            numerical = extract_numerical_answer(answer_text)
            if numerical is None:
                logger.warning(f"Skipping GSM8K example with unparsable answer: {answer_text}")
                continue
            answers.append(numerical)

        return GenerationValDataset(prompts, answers)

    def validate_batch(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizerBase,
        batch: dict,
    ) -> tuple[int, int]:
        """Generate answers and compare to ground truth."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        expected_answers = batch["labels"]

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        correct = 0
        total = len(expected_answers)
        for gen_text, expected in zip(generated_texts, expected_answers):
            predicted = extract_numerical_answer(gen_text)
            if predicted is not None and predicted == expected:
                correct += 1

        return correct, total
