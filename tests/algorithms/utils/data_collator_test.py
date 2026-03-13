"""Tests for DataCollatorForAnswerOnlyLM."""

from unittest.mock import MagicMock

from information_safety.algorithms.utils.data_collator import DataCollatorForAnswerOnlyLM


class TestDataCollatorForAnswerOnlyLM:
    def _make_collator(self, answer_token_ids: list[int]) -> DataCollatorForAnswerOnlyLM:
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.encode.return_value = answer_token_ids
        return DataCollatorForAnswerOnlyLM(tokenizer=tokenizer, answer_start_token="<answer>")

    def test_masks_question_tokens(self) -> None:
        collator = self._make_collator(answer_token_ids=[99])
        features = [{"input_ids": [10, 20, 99, 30, 40], "attention_mask": [1, 1, 1, 1, 1]}]
        batch = collator(features)
        labels = batch["labels"][0].tolist()
        assert labels[0] == -100
        assert labels[1] == -100
        assert labels[2] == -100
        assert labels[3] == 30

    def test_masks_last_token(self) -> None:
        """The last real token (EOS/EOD) should be masked with -100."""
        collator = self._make_collator(answer_token_ids=[99])
        features = [{"input_ids": [10, 20, 99, 30, 40], "attention_mask": [1, 1, 1, 1, 1]}]
        batch = collator(features)
        labels = batch["labels"][0].tolist()
        assert labels[4] == -100, "Last token should be masked"

    def test_masks_last_token_with_padding(self) -> None:
        """Last real token masked even when sequences have different lengths."""
        collator = self._make_collator(answer_token_ids=[99])
        features = [
            {"input_ids": [10, 99, 30, 40], "attention_mask": [1, 1, 1, 1]},
            {"input_ids": [10, 99, 30, 40, 50], "attention_mask": [1, 1, 1, 1, 1]},
        ]
        batch = collator(features)
        labels_short = batch["labels"][0].tolist()
        labels_long = batch["labels"][1].tolist()
        assert labels_short[3] == -100, "Last real token of short seq should be masked"
        assert labels_short[4] == -100, "Padding should be masked"
        assert labels_long[4] == -100, "Last real token of long seq should be masked"

    def test_answer_tokens_in_middle_are_unmasked(self) -> None:
        """Tokens between answer marker and last token should be unmasked."""
        collator = self._make_collator(answer_token_ids=[99])
        features = [{"input_ids": [10, 99, 30, 40, 50], "attention_mask": [1, 1, 1, 1, 1]}]
        batch = collator(features)
        labels = batch["labels"][0].tolist()
        assert labels[2] == 30
        assert labels[3] == 40
        assert labels[4] == -100

    def test_no_answer_marker_all_masked(self) -> None:
        """If answer marker is not found, all labels should be -100."""
        collator = self._make_collator(answer_token_ids=[99])
        features = [{"input_ids": [10, 20, 30], "attention_mask": [1, 1, 1]}]
        batch = collator(features)
        labels = batch["labels"][0].tolist()
        assert all(label == -100 for label in labels)
