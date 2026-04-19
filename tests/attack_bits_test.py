"""Tests for per-attack bits accounting (GCG / AutoDAN / PAIR)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from information_safety.attack_bits import (
    TOKEN_ID_BITS,
    autodan_bits,
    class_code_bits,
    compute_attack_bits,
    gcg_bits,
    pair_bits,
    parse_autodan_initial_strings_bits,
)


class TestClassCodeBits:
    def test_returns_source_length_bits(self, tmp_path: Path) -> None:
        src = "class Foo:\n    def bar(self) -> int:\n        return 1\n"
        py_file = tmp_path / "mod.py"
        py_file.write_text(src, encoding="utf-8")

        bits = class_code_bits(str(py_file), "Foo")

        assert bits == len(src.encode("utf-8")) * 8

    def test_finds_class_among_other_top_level_defs(self, tmp_path: Path) -> None:
        class_src = "class Target:\n    x: int = 1\n"
        full_src = "import os\n\n" + class_src + "\nclass Other:\n    y: int = 2\n"
        py_file = tmp_path / "mod.py"
        py_file.write_text(full_src, encoding="utf-8")

        bits = class_code_bits(str(py_file), "Target")

        assert bits == len(class_src.encode("utf-8")) * 8

    def test_raises_on_missing_class(self, tmp_path: Path) -> None:
        py_file = tmp_path / "mod.py"
        py_file.write_text("class Other: pass\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Missing"):
            class_code_bits(str(py_file), "Nope")


class TestParseAutodanInitialStringsBits:
    def test_returns_avg_bits_and_count(self, tmp_path: Path) -> None:
        src = 'INITIAL_STRINGS = ["ab", "cdef"]\n'
        py_file = tmp_path / "autodan.py"
        py_file.write_text(src, encoding="utf-8")

        avg_bits, count = parse_autodan_initial_strings_bits(str(py_file))

        expected_avg = int(round(((len(b"ab") + len(b"cdef")) * 8) / 2))
        assert count == 2
        assert avg_bits == expected_avg

    def test_raises_on_missing_initial_strings(self, tmp_path: Path) -> None:
        py_file = tmp_path / "autodan.py"
        py_file.write_text("X = [1, 2]\n", encoding="utf-8")

        with pytest.raises(ValueError, match="INITIAL_STRINGS"):
            parse_autodan_initial_strings_bits(str(py_file))


class TestGcgBits:
    def test_matches_reference_formula(self) -> None:
        config: dict[str, Any] = {
            "num_steps": 10,
            "topk": 8,
            "optim_str_init": "x x x x",
        }

        out = gcg_bits(config)

        suffix_tokens = 4
        expected_search = int(round(10 * suffix_tokens * math.log2(8)))
        expected_payload = suffix_tokens * TOKEN_ID_BITS
        expected_meta = 10 * TOKEN_ID_BITS

        assert out["search_bits"] == expected_search
        assert out["payload_bits"] == expected_payload
        assert out["meta_bits"] == expected_meta
        assert out["code_bits"] > 0
        assert out["total"] == out["code_bits"] + expected_search + expected_payload + expected_meta

    def test_handles_topk_one(self) -> None:
        config: dict[str, Any] = {
            "num_steps": 4,
            "topk": 1,
            "optim_str_init": "x",
        }

        out = gcg_bits(config)

        assert out["search_bits"] == int(round(4 * 1 * math.log2(2)))


class TestAutodanBits:
    def test_matches_reference_formula(self) -> None:
        config: dict[str, Any] = {
            "num_steps": 5,
            "batch_size": 4,
            "mutation": 0.25,
        }

        out = autodan_bits(config)

        expected_evolution = int(round(5 * 4 * math.log2(4)))
        expected_mutation = int(round(5 * 4 * 0.25 * TOKEN_ID_BITS))

        assert out["evolution_bits"] == expected_evolution
        assert out["mutation_bits"] == expected_mutation
        assert out["code_bits"] > 0
        assert out["initial_pool_bits"] > 0
        assert out["total"] == (
            out["code_bits"]
            + out["initial_pool_bits"]
            + expected_evolution
            + expected_mutation
        )


class TestPairBits:
    def test_matches_reference_formula_without_helper_weights(self) -> None:
        config: dict[str, Any] = {
            "num_steps": 3,
            "num_streams": 2,
            "attack_model": {
                "max_new_tokens": 8,
                "max_attempts": 4,
            },
        }

        out = pair_bits(config)

        expected_payload = 3 * 2 * 8 * TOKEN_ID_BITS
        expected_search = int(round(3 * 2 * 4 * math.log2(8)))

        assert out["payload_bits"] == expected_payload
        assert out["search_bits"] == expected_search
        assert out["code_bits"] > 0
        assert out["total"] == out["code_bits"] + expected_payload + expected_search

    def test_total_excludes_helper_weight_term(self) -> None:
        """PAIR total must NOT include 13e9 * 16 helper-model parameter bits."""
        config: dict[str, Any] = {
            "num_steps": 3,
            "num_streams": 2,
            "attack_model": {
                "max_new_tokens": 8,
                "max_attempts": 4,
            },
        }
        helper_term = 13_000_000_000 * 16

        out = pair_bits(config)

        assert out["total"] < helper_term
        assert "helper_model_bits" not in out


class TestComputeAttackBits:
    def test_dispatches_gcg_by_optim_str_init(self) -> None:
        config: dict[str, Any] = {
            "num_steps": 10,
            "topk": 8,
            "optim_str_init": "x x x x",
        }
        assert compute_attack_bits("gcg", config) == gcg_bits(config)["total"]

    def test_dispatches_autodan_by_batch_size_and_mutation(self) -> None:
        config: dict[str, Any] = {
            "num_steps": 5,
            "batch_size": 4,
            "mutation": 0.25,
        }
        assert compute_attack_bits("autodan", config) == autodan_bits(config)["total"]

    def test_dispatches_pair_by_num_streams(self) -> None:
        config: dict[str, Any] = {
            "num_steps": 3,
            "num_streams": 2,
            "attack_model": {"max_new_tokens": 8, "max_attempts": 4},
        }
        assert compute_attack_bits("pair", config) == pair_bits(config)["total"]

    def test_raises_on_unknown_attack(self) -> None:
        with pytest.raises(ValueError, match="Unknown attack"):
            compute_attack_bits("mystery", {})
