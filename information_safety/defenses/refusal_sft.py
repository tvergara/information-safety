"""Refusal SFT defense: vanilla supervised finetuning on refusals + retain.

Trains the model to emit ``REFUSAL_RESPONSE`` on harmful prompts while preserving
benign capability via a 10x retain mix. Uses HuggingFace ``Trainer`` rather than
``trl.SFTTrainer`` to avoid an extra dependency. Answer tokens are masked via
``DataCollatorForAnswerOnlyLM`` so loss is computed only on the assistant turn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, cast

import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from information_safety.algorithms.dataset_handlers.safety_pair import (
    get_answer_start_token,
)
from information_safety.algorithms.utils.data_collator import (
    DataCollatorForAnswerOnlyLM,
)

logger = getLogger(__name__)


@dataclass
class SFTHParams:
    max_steps: int = 200
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    max_length: int = 1024
    bf16: bool = True
    gradient_checkpointing: bool = True
    extra_training_args: dict[str, Any] = field(default_factory=dict)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_chat_text(
    tokenizer: PreTrainedTokenizerBase, user: str, assistant: str
) -> str:
    return cast(
        str,
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            tokenize=False,
        ),
    )


def _build_train_dataset(
    tokenizer: PreTrainedTokenizerBase,
    refusals: list[dict[str, Any]],
    retain: list[dict[str, Any]],
    max_length: int,
) -> datasets.Dataset:
    rows: list[dict[str, Any]] = []
    for ex in refusals:
        rows.append({"user": ex["prompt"], "assistant": ex["refusal"]})
    for ex in retain:
        rows.append({"user": ex["prompt"], "assistant": ex["response"]})

    ds = datasets.Dataset.from_list(rows)

    def tokenize(example: dict[str, str]) -> Any:
        text = _build_chat_text(tokenizer, example["user"], example["assistant"])
        return tokenizer(text, truncation=True, max_length=max_length, padding=False)

    return ds.map(tokenize, remove_columns=ds.column_names)  # type: ignore[return-value]


def train_refusal_sft(
    base_model: str,
    refusals_path: Path,
    retain_path: Path,
    output_dir: Path,
    hparams: SFTHParams,
    dry_run: bool = False,
) -> None:
    if not Path(refusals_path).exists():
        raise FileNotFoundError(refusals_path)
    if not Path(retain_path).exists():
        raise FileNotFoundError(retain_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    refusals = _load_jsonl(refusals_path)
    retain = _load_jsonl(retain_path)
    logger.info(
        f"Refusal SFT: {len(refusals)} refusals + {len(retain)} retain examples"
    )

    train_ds = _build_train_dataset(tokenizer, refusals, retain, hparams.max_length)
    answer_marker = get_answer_start_token(tokenizer)
    collator = DataCollatorForAnswerOnlyLM(tokenizer=tokenizer, answer_start_token=answer_marker)

    if dry_run:
        logger.info("dry_run=True — skipping Trainer instantiation")
        return

    model = AutoModelForCausalLM.from_pretrained(base_model)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=hparams.max_steps,
        per_device_train_batch_size=hparams.per_device_train_batch_size,
        gradient_accumulation_steps=hparams.gradient_accumulation_steps,
        learning_rate=hparams.learning_rate,
        bf16=hparams.bf16,
        gradient_checkpointing=hparams.gradient_checkpointing,
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        **hparams.extra_training_args,
    )
    if hparams.gradient_checkpointing:
        model.config.use_cache = False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
