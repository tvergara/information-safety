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

    def test_last_real_token_is_loss_targeted(self) -> None:
        collator = self._make_collator(answer_token_ids=[99])
        features = [{"input_ids": [10, 20, 99, 30, 40], "attention_mask": [1, 1, 1, 1, 1]}]
        batch = collator(features)
        labels = batch["labels"][0].tolist()
        assert labels[4] == 40, "Last real token must be loss-targeted (was -100)"

    def test_last_real_token_loss_targeted_with_padding(self) -> None:
        collator = self._make_collator(answer_token_ids=[99])
        features = [
            {"input_ids": [10, 99, 30, 40], "attention_mask": [1, 1, 1, 1]},
            {"input_ids": [10, 99, 30, 40, 50], "attention_mask": [1, 1, 1, 1, 1]},
        ]
        batch = collator(features)
        labels_short = batch["labels"][0].tolist()
        labels_long = batch["labels"][1].tolist()
        assert labels_short[3] == 40, "Last real token of short seq must be loss-targeted"
        assert labels_short[4] == -100, "Padding must be masked"
        assert labels_long[4] == 50, "Last real token of long seq must be loss-targeted"

    def test_all_answer_tokens_including_last_are_unmasked(self) -> None:
        collator = self._make_collator(answer_token_ids=[99])
        features = [{"input_ids": [10, 99, 30, 40, 50], "attention_mask": [1, 1, 1, 1, 1]}]
        batch = collator(features)
        labels = batch["labels"][0].tolist()
        assert labels[2] == 30
        assert labels[3] == 40
        assert labels[4] == 50, "Last token (EOS) must be loss-targeted"

    def test_no_answer_marker_all_masked(self) -> None:
        """If answer marker is not found, all labels should be -100."""
        collator = self._make_collator(answer_token_ids=[99])
        features = [{"input_ids": [10, 20, 30], "attention_mask": [1, 1, 1]}]
        batch = collator(features)
        labels = batch["labels"][0].tolist()
        assert all(label == -100 for label in labels)
