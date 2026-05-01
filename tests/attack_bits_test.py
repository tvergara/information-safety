"""Tests for the 3-primitive program-length accounting (code + data + aux models)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from information_safety.attack_bits import (
    STRATEGY_CODE_FILES,
    autodan_bits,
    code_bits,
    compute_attack_bits,
    file_bits,
    gcg_bits,
    pair_bits,
)


class TestFileBits:
    def test_returns_utf8_bit_length(self, tmp_path: Path) -> None:
        src = "class Foo:\n    pass\n"
        py_file = tmp_path / "mod.py"
        py_file.write_text(src, encoding="utf-8")

        assert file_bits(py_file) == len(src.encode("utf-8")) * 8

    def test_handles_unicode(self, tmp_path: Path) -> None:
        src = "x = 'é'\n"
        py_file = tmp_path / "mod.py"
        py_file.write_text(src, encoding="utf-8")

        assert file_bits(py_file) == len(src.encode("utf-8")) * 8

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            file_bits(tmp_path / "nope.py")


class TestCodeBits:
    def test_sums_file_bits(self, tmp_path: Path) -> None:
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        a.write_text("a\n", encoding="utf-8")
        b.write_text("bb\n", encoding="utf-8")

        assert code_bits([a, b]) == file_bits(a) + file_bits(b)

    def test_order_independent(self, tmp_path: Path) -> None:
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        a.write_text("hello\n", encoding="utf-8")
        b.write_text("world!\n", encoding="utf-8")

        assert code_bits([a, b]) == code_bits([b, a])

    def test_empty_returns_zero(self) -> None:
        assert code_bits([]) == 0


class TestStrategyCodeFiles:
    def test_has_all_seven_strategies(self) -> None:
        expected = {"baseline", "roleplay", "lora", "data", "gcg", "autodan", "pair"}
        assert set(STRATEGY_CODE_FILES.keys()) == expected

    def test_lists_are_non_empty(self) -> None:
        for name, paths in STRATEGY_CODE_FILES.items():
            assert paths, f"strategy {name} has empty file list"

    def test_all_paths_exist(self) -> None:
        for name, paths in STRATEGY_CODE_FILES.items():
            for path in paths:
                assert path.is_file(), f"missing {path} for strategy {name}"

    @pytest.mark.parametrize(
        ("name", "expected_filenames"),
        [
            ("baseline", ["naive_inference.py"]),
            ("roleplay", ["naive_inference.py", "prompt.py", "prompt_methods.py"]),
            ("lora", ["naive_inference.py", "lora.py"]),
            (
                "data",
                [
                    "naive_inference.py",
                    "data.py",
                    "lora.py",
                    "arithmetic_coding.py",
                ],
            ),
            (
                "gcg",
                [
                    "naive_inference.py",
                    "gcg.py",
                    "precomputed_adversarial_prompt.py",
                ],
            ),
            (
                "autodan",
                [
                    "naive_inference.py",
                    "autodan.py",
                    "precomputed_adversarial_prompt.py",
                ],
            ),
            (
                "pair",
                [
                    "naive_inference.py",
                    "pair.py",
                    "precomputed_adversarial_prompt.py",
                ],
            ),
        ],
    )
    def test_pinned_filenames(self, name: str, expected_filenames: list[str]) -> None:
        actual = [p.name for p in STRATEGY_CODE_FILES[name]]
        assert actual == expected_filenames


class TestGcgBits:
    def test_returns_dict_with_breakdown(self) -> None:
        out = gcg_bits()

        expected_code = code_bits(STRATEGY_CODE_FILES["gcg"])
        assert out["code_bits"] == expected_code
        assert out["data_bits"] == 0
        assert out["auxiliary_model_bits"] == 0
        assert out["total"] == expected_code


class TestAutodanBits:
    def test_returns_dict_with_breakdown(self) -> None:
        out = autodan_bits()

        expected_code = code_bits(STRATEGY_CODE_FILES["autodan"])
        assert out["code_bits"] == expected_code
        assert out["data_bits"] == 0
        assert out["auxiliary_model_bits"] == 0
        assert out["total"] == expected_code


class TestPairBits:
    def test_no_aux_when_helper_equals_attacked(self) -> None:
        config: dict[str, Any] = {
            "attack_model": {"id": "meta-llama/Llama-3-8B"},
        }
        out = pair_bits(config, attacked_model_name="meta-llama/Llama-3-8B")

        expected_code = code_bits(STRATEGY_CODE_FILES["pair"])
        assert out["code_bits"] == expected_code
        assert out["data_bits"] == 0
        assert out["auxiliary_model_bits"] == 0
        assert out["total"] == expected_code

    def test_adds_aux_when_helper_differs(self) -> None:
        helper_params = 13_000_000_000
        config: dict[str, Any] = {
            "attack_model": {
                "id": "lmsys/vicuna-13b",
                "num_params": helper_params,
            },
        }
        out = pair_bits(config, attacked_model_name="meta-llama/Llama-3-8B")

        expected_code = code_bits(STRATEGY_CODE_FILES["pair"])
        expected_aux = 16 * helper_params
        assert out["code_bits"] == expected_code
        assert out["auxiliary_model_bits"] == expected_aux
        assert out["total"] == expected_code + expected_aux

class TestComputeAttackBits:
    def test_dispatches_gcg(self) -> None:
        assert compute_attack_bits("gcg", {}) == gcg_bits()["total"]

    def test_dispatches_autodan(self) -> None:
        assert compute_attack_bits("autodan", {}) == autodan_bits()["total"]

    def test_dispatches_pair_with_attacked_model_name(self) -> None:
        config: dict[str, Any] = {
            "attack_model": {
                "id": "lmsys/vicuna-13b",
                "num_params": 13_000_000_000,
            },
        }
        result = compute_attack_bits(
            "pair", config, attacked_model_name="meta-llama/Llama-3-8B"
        )
        assert result == pair_bits(
            config, attacked_model_name="meta-llama/Llama-3-8B"
        )["total"]

    def test_pair_raises_without_attacked_model_name(self) -> None:
        config: dict[str, Any] = {
            "attack_model": {"id": "x", "num_params": 7_000_000_000},
        }
        with pytest.raises(ValueError, match="PAIR requires attacked_model_name"):
            compute_attack_bits("pair", config)

    def test_raises_on_unknown_attack(self) -> None:
        with pytest.raises(ValueError, match="Unknown attack"):
            compute_attack_bits("mystery", {})


class TestNoLegacyTerms:
    def test_module_does_not_export_old_names(self) -> None:
        from information_safety import attack_bits

        for removed in (
            "class_code_bits",
            "parse_autodan_initial_strings_bits",
            "search_bits",
            "payload_bits",
            "meta_bits",
            "evolution_bits",
            "mutation_bits",
            "initial_pool_bits",
        ):
            assert not hasattr(attack_bits, removed), (
                f"{removed} should be removed from attack_bits"
            )
