"""Custom data collators for LLM finetuning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForAnswerOnlyLM:
    """Collator that pads sequences and masks question tokens with -100 in labels.

    This ensures the model is only trained to predict answer tokens.
    """

    tokenizer: PreTrainedTokenizerBase
    answer_start_token: str = "### Answer:"

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
        labels = [torch.tensor(f["input_ids"]) for f in features]

        max_len = max(ids.size(0) for ids in input_ids)
        pad_id: int = self.tokenizer.pad_token_id  # type: ignore[assignment]

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        answer_token_ids = self.tokenizer.encode(
            self.answer_start_token, add_special_tokens=False
        )

        for ids, mask, lab in zip(input_ids, attention_mask, labels):
            pad_len = max_len - ids.size(0)

            padded_input_ids.append(
                torch.cat([ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)])
            )
            padded_attention_mask.append(
                torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
            )

            masked_labels = torch.full_like(lab, -100)
            answer_start = _find_subsequence(ids.tolist(), answer_token_ids)
            if answer_start >= 0:
                answer_offset = answer_start + len(answer_token_ids)
                masked_labels[answer_offset:] = lab[answer_offset:]

            padded_labels.append(
                torch.cat([masked_labels, torch.full((pad_len,), -100, dtype=lab.dtype)])
            )

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }


def _find_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    """Find the start index of a subsequence in a sequence.

    Returns -1 if not found.
    """
    for i in range(len(sequence) - len(subsequence) + 1):
        if sequence[i : i + len(subsequence)] == subsequence:
            return i
    return -1
