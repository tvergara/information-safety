"""Tests for jailbreak attack bit-accounting utilities."""

from __future__ import annotations

from information_safety.algorithms.jailbreak.metrics import (
    aggregate_bits_by_method,
    compute_text_bits,
)


class TestComputeTextBits:
    def test_returns_utf8_bit_count(self) -> None:
        assert compute_text_bits("abc") == 24

    def test_empty_string_has_zero_bits(self) -> None:
        assert compute_text_bits("") == 0


class TestAggregateBitsByMethod:
    def test_aggregates_average_bits_per_method(self) -> None:
        rows = [
            {"attack_method": "zero_shot", "bits": 8},
            {"attack_method": "zero_shot", "bits": 24},
            {"attack_method": "gcg", "bits": 128},
        ]
        result = aggregate_bits_by_method(rows)
        assert result["zero_shot"] == 16.0
        assert result["gcg"] == 128.0

    def test_ignores_rows_without_method_or_bits(self) -> None:
        rows = [{"attack_method": "a"}, {"bits": 12}, {"attack_method": "b", "bits": "bad"}]
        result = aggregate_bits_by_method(rows)
        assert result == {}
