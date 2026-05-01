"""Base classes for dataset handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class BaseDatasetHandler(ABC):
    """Abstract base class for dataset handlers.

    A dataset handler is responsible for:
    - Loading and formatting training data
    - Loading and formatting validation data
    - Validating model outputs against ground truth
    """

    batch_size: int
    max_length: int
    dataset_name: str
    max_examples: int | None
    max_new_tokens: int = 320

    @abstractmethod
    def get_train_dataset(self, tokenizer: PreTrainedTokenizerBase) -> Dataset:
        """Return the tokenized training dataset."""

    @abstractmethod
    def get_val_dataset(self, tokenizer: PreTrainedTokenizerBase) -> Dataset:
        """Return the validation dataset."""

    @abstractmethod
    def validate_batch(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizerBase,
        batch: dict,
    ) -> tuple[int, int]:
        """Validate a batch and return (num_correct, num_total)."""

    def get_answer_start_token(self, tokenizer: PreTrainedTokenizerBase) -> str:
        """Return the token string that marks the start of the answer in training data.

        Used by DataCollatorForAnswerOnlyLM to mask question tokens in labels.
        """
        return "### Answer:"

    def save_completions(self, eval_run_id: str) -> None:
        """Save accumulated completions to disk.

        No-op by default.
        """

    def record_precomputed_completions(
        self, metadata_strs: list[str], completions: list[str]
    ) -> tuple[int, int]:
        """Record stored attack completions without running ``model.generate``.

        Handlers that support precomputed-prompt strategies must override this. The default raises
        so misuse fails loudly instead of silently dropping data.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support precomputed completions"
        )


class GenerationValDataset(Dataset):
    """A validation dataset that stores prompts and expected answers for generation."""

    def __init__(self, prompts: list[dict[str, Any]], answers: list[str]) -> None:
        self.prompts = prompts
        self.answers = answers

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = dict(self.prompts[idx])
        item["labels"] = self.answers[idx]
        return item
