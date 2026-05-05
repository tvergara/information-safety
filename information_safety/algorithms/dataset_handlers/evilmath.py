"""EvilMath dataset handler: evaluates models on harmfully-framed math problems."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, cast

import datasets
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from information_safety.algorithms.dataset_handlers.base import (
    BaseDatasetHandler,
    GenerationValDataset,
)

logger = getLogger(__name__)

ANSWER_PATTERNS = [
    re.compile(r"\\boxed\{\s*(-?\d+)\s*\}"),
    re.compile(r"####\s*(-?\d+)"),
]


def extract_numerical_answer(text: str) -> int | None:
    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return int(matches[-1])
    return None


@dataclass
class EvilMathHandler(BaseDatasetHandler):
    """EvilMath evaluation; optional compliance/GSM8K mix for DataStrategy training."""

    max_length: int = 1024
    batch_size: int = 4
    max_new_tokens: int = 1024
    generations_dir: str = ""
    dataset_name: str = "evilmath"
    max_examples: int | None = None
    train_data_path: str | None = None
    corpus_fraction: float = 0.0
    _completions: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def get_train_dataset(self, tokenizer: PreTrainedTokenizerBase) -> Any:
        if self.train_data_path is None:
            return datasets.Dataset.from_dict({"input_ids": [[0]], "attention_mask": [[1]]})

        if self.corpus_fraction not in (0.0, 0.5):
            raise ValueError(
                f"corpus_fraction must be 0.0 or 0.5; got {self.corpus_fraction}"
            )
        if self.max_examples is None:
            raise ValueError("max_examples is required when train_data_path is set")

        def tokenize_chat(user_text: str, assistant_text: str) -> Any:
            text = cast(str, tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ],
                tokenize=False,
            ))
            return tokenizer(text, truncation=True, max_length=self.max_length, padding=False)

        def tokenize_compliance(example: dict[str, str]) -> Any:
            return tokenize_chat(example["prompt"], example["completion"])

        def tokenize_gsm8k(example: dict[str, str]) -> Any:
            return tokenize_chat(example["question"], example["answer"])

        compliance_raw = datasets.load_dataset(
            "json", data_files=self.train_data_path, split="train"
        )

        compliance_n = (
            self.max_examples if self.corpus_fraction == 0.0 else self.max_examples // 2
        )
        compliance_raw = compliance_raw.select(range(compliance_n))  # type: ignore[union-attr]
        compliance_tokenized = compliance_raw.map(  # type: ignore[union-attr]
            tokenize_compliance,
            remove_columns=compliance_raw.column_names,  # type: ignore[union-attr]
        )

        if self.corpus_fraction == 0.0:
            return compliance_tokenized

        gsm8k_n = self.max_examples - compliance_n
        gsm8k_raw = datasets.load_dataset("openai/gsm8k", "main", split="train")
        gsm8k_raw = gsm8k_raw.select(range(gsm8k_n))  # type: ignore[union-attr]
        gsm8k_tokenized = gsm8k_raw.map(  # type: ignore[union-attr]
            tokenize_gsm8k,
            remove_columns=gsm8k_raw.column_names,  # type: ignore[union-attr]
        )

        return datasets.concatenate_datasets([compliance_tokenized, gsm8k_tokenized])

    def result_row_extras(self) -> dict[str, Any]:
        return {"corpus_fraction": self.corpus_fraction}

    def get_val_dataset(self, tokenizer: PreTrainedTokenizerBase) -> Dataset:
        """Load EvilMath test split and format each row as a chat prompt."""
        raw = datasets.load_dataset("ethz-spylab/EvilMath", split="test")

        prompts: list[dict[str, Any]] = []
        labels: list[str] = []

        for example in raw:  # type: ignore[union-attr]
            evil_question = example["evil_question"]  # type: ignore[index]
            original_answer = example["original_answer"]  # type: ignore[index]
            original_question = example["original_question"]  # type: ignore[index]

            correct_answer = extract_numerical_answer(original_answer)
            if correct_answer is None:
                logger.warning(
                    f"Skipping EvilMath row {example['index']} with unparsable answer"  # type: ignore[index]
                )
                continue

            prompt_text = cast(str, tokenizer.apply_chat_template(
                [{"role": "user", "content": evil_question}],
                tokenize=False,
                add_generation_prompt=True,
            ))
            encoded = dict(tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            ))
            prompts.append(encoded)
            labels.append(json.dumps({
                "question": evil_question,
                "correct_answer": correct_answer,
                "original_question": original_question,
            }))

        if self.max_examples is not None:
            prompts = prompts[: self.max_examples]
            labels = labels[: self.max_examples]

        return GenerationValDataset(prompts, labels)

    def validate_batch(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizerBase,
        batch: dict,
    ) -> tuple[int, int]:
        """Generate responses, extract final integer answer, compare to ground truth."""
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

        input_len = input_ids.shape[1]
        new_token_ids = generated_ids[:, input_len:]
        generated_texts = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)

        correct = 0
        for gen_text, meta_str in zip(generated_texts, metadata_strs):
            meta = json.loads(meta_str)
            correct_answer = meta["correct_answer"]
            predicted = extract_numerical_answer(gen_text)
            is_correct = predicted is not None and predicted == correct_answer
            if is_correct:
                correct += 1
            self._completions.append({
                "question": meta["question"],
                "correct_answer": correct_answer,
                "original_question": meta["original_question"],
                "response": gen_text,
                "predicted_answer": predicted,
                "correct": is_correct,
            })

        return correct, len(generated_texts)

    def save_completions(self, eval_run_id: str) -> None:
        """Write accumulated completions to disk as JSONL files."""
        run_dir = Path(self.generations_dir) / eval_run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "input_data.jsonl", "w") as f:
            for entry in self._completions:
                row = {
                    "question": entry["question"],
                    "correct_answer": entry["correct_answer"],
                    "original_question": entry["original_question"],
                }
                f.write(json.dumps(row) + "\n")

        with open(run_dir / "responses.jsonl", "w") as f:
            for entry in self._completions:
                row = {
                    "question": entry["question"],
                    "response": entry["response"],
                    "predicted_answer": entry["predicted_answer"],
                    "correct": entry["correct"],
                }
                f.write(json.dumps(row) + "\n")

        logger.info(f"Saved {len(self._completions)} EvilMath completions to {run_dir}")
        self._completions = []
