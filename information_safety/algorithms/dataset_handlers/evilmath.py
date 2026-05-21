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
from information_safety.algorithms.dataset_handlers.safety_pair import (
    get_answer_start_token,
)

logger = getLogger(__name__)

ANSWER_PATTERNS = [
    re.compile(r"\\boxed\{\s*(-?\d[\d,]*)\s*\}"),
    re.compile(r"####\s*(-?\d[\d,]*)"),
    re.compile(r"[Tt]he answer is[\s:$*]*(-?\d[\d,]*)"),
    re.compile(r"[Aa]nswer:[^\d-]*(-?\d[\d,]*)"),
]

TRAIN_SHUFFLE_SEED = 42


def extract_numerical_answer(text: str) -> int | None:
    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return int(matches[-1].replace(",", ""))
    return None


@dataclass
class EvilMathHandler(BaseDatasetHandler):
    """EvilMath evaluation; GSM8K corpus for DataStrategy training."""

    max_length: int = 1024
    batch_size: int = 4
    max_new_tokens: int = 1024
    generations_dir: str = ""
    dataset_name: str = "evilmath"
    max_examples: int | None = None
    _completions: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def get_answer_start_token(self, tokenizer: PreTrainedTokenizerBase) -> str:
        return get_answer_start_token(tokenizer)

    def get_train_dataset(self, tokenizer: PreTrainedTokenizerBase) -> Any:
        if self.max_examples is None:
            raise ValueError("max_examples is required for EvilMath training")

        def tokenize_gsm8k(example: dict[str, str]) -> Any:
            text = cast(str, tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": example["question"]},
                    {"role": "assistant", "content": example["answer"]},
                ],
                tokenize=False,
            ))
            return tokenizer(text, truncation=True, max_length=self.max_length, padding=False)

        gsm8k_raw = datasets.load_dataset("openai/gsm8k", "main", split="train")
        gsm8k_raw = gsm8k_raw.shuffle(seed=TRAIN_SHUFFLE_SEED)  # type: ignore[union-attr]
        gsm8k_raw = gsm8k_raw.select(range(self.max_examples))  # type: ignore[union-attr]
        return gsm8k_raw.map(  # type: ignore[union-attr]
            tokenize_gsm8k,
            remove_columns=gsm8k_raw.column_names,  # type: ignore[union-attr]
        )

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

        return GenerationValDataset(prompts, labels)

    def validate_batch(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizerBase,
        batch: dict,
        strategy: Any,
    ) -> tuple[int, int]:
        """Generate responses, extract final integer answer, compare to ground truth."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        metadata_strs: list[str] = batch["labels"]
        pad_token_id = tokenizer.pad_token_id

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=pad_token_id,
            )

        input_len = input_ids.shape[1]
        new_token_ids = generated_ids[:, input_len:]
        generated_texts = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)

        num_params = model.num_parameters(exclude_embeddings=True)

        correct = 0
        for i, (gen_text, meta_str) in enumerate(zip(generated_texts, metadata_strs)):
            meta = json.loads(meta_str)
            correct_answer = meta["correct_answer"]
            predicted = extract_numerical_answer(gen_text)
            is_correct = predicted is not None and predicted == correct_answer
            if is_correct:
                correct += 1
            n_input = attention_mask[i].sum().item()
            n_output = (new_token_ids[i] != pad_token_id).sum().item()
            self._completions.append({
                "question": meta["question"],
                "correct_answer": correct_answer,
                "original_question": meta["original_question"],
                "response": gen_text,
                "predicted_answer": predicted,
                "correct": is_correct,
                "eval_flops": strategy.eval_forward_flops(num_params, n_input, n_output),
                "attack_flops": strategy.attack_flops_for(meta),
                "pareto_step_idx": meta.get("pareto_step_idx", 0),
            })

        return correct, len(generated_texts)

    def record_precomputed_completions(
        self, metadata_strs: list[str], completions: list[str], strategy: Any,
    ) -> tuple[int, int]:
        """Record stored attack completions; eval-time FLOPs are 0 since no generate is called."""
        correct = 0
        for meta_str, completion in zip(metadata_strs, completions):
            meta = json.loads(meta_str)
            correct_answer = meta["correct_answer"]
            predicted = extract_numerical_answer(completion)
            is_correct = predicted is not None and predicted == correct_answer
            if is_correct:
                correct += 1
            self._completions.append({
                "question": meta["question"],
                "correct_answer": correct_answer,
                "original_question": meta["original_question"],
                "response": completion,
                "predicted_answer": predicted,
                "correct": is_correct,
                "eval_flops": 0,
                "attack_flops": strategy.attack_flops_for(meta),
                "pareto_step_idx": meta.get("pareto_step_idx", 0),
            })
        return correct, len(completions)

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
                    "pareto_step_idx": entry["pareto_step_idx"],
                }
                f.write(json.dumps(row) + "\n")

        with open(run_dir / "responses.jsonl", "w") as f:
            for entry in self._completions:
                dependent_flops = entry["attack_flops"] + entry["eval_flops"]
                row = {
                    "question": entry["question"],
                    "response": entry["response"],
                    "predicted_answer": entry["predicted_answer"],
                    "correct": entry["correct"],
                    "dependent_flops": dependent_flops,
                    "pareto_step_idx": entry["pareto_step_idx"],
                    "attack_flops": entry["attack_flops"],
                }
                f.write(json.dumps(row) + "\n")

        logger.info(f"Saved {len(self._completions)} EvilMath completions to {run_dir}")
        self._completions = []
