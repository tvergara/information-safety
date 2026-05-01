"""Per-attack program-length bits accounting (3-primitive rule).

`program_bits = code_bits + data_bits + auxiliary_model_bits`, per
`docs/measurement-definitions.md`. Program length is input-independent: no
per-test-example, per-behavior, or search-trajectory terms enter the formula.

This module is the **single source of truth** for which `.py` files count as
code for each strategy. Every strategy's `compute_bits()` reads from
`STRATEGY_CODE_FILES` by name.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_PKG_ROOT: Path = _REPO_ROOT / "information_safety"
_STRATEGIES_DIR: Path = _PKG_ROOT / "algorithms" / "strategies"
_JAILBREAK_DIR: Path = _PKG_ROOT / "algorithms" / "jailbreak"
_UTILS_DIR: Path = _PKG_ROOT / "algorithms" / "utils"
_ATTACKS_DIR: Path = (
    _REPO_ROOT / "third_party" / "adversariallm" / "adversariallm" / "attacks"
)

NAIVE_INFERENCE: Path = _PKG_ROOT / "scripts" / "naive_inference.py"
PROMPT_PY: Path = _STRATEGIES_DIR / "prompt.py"
PROMPT_METHODS_PY: Path = _JAILBREAK_DIR / "prompt_methods.py"
LORA_PY: Path = _STRATEGIES_DIR / "lora.py"
DATA_PY: Path = _STRATEGIES_DIR / "data.py"
ARITHMETIC_CODING_PY: Path = _UTILS_DIR / "arithmetic_coding.py"
PRECOMPUTED_PY: Path = _STRATEGIES_DIR / "precomputed_adversarial_prompt.py"
GCG_PY: Path = _ATTACKS_DIR / "gcg.py"
AUTODAN_PY: Path = _ATTACKS_DIR / "autodan.py"
PAIR_PY: Path = _ATTACKS_DIR / "pair.py"


STRATEGY_CODE_FILES: dict[str, list[Path]] = {
    "baseline": [NAIVE_INFERENCE],
    "roleplay": [NAIVE_INFERENCE, PROMPT_PY, PROMPT_METHODS_PY],
    "lora": [NAIVE_INFERENCE, LORA_PY],
    "data": [NAIVE_INFERENCE, DATA_PY, LORA_PY, ARITHMETIC_CODING_PY],
    "gcg": [NAIVE_INFERENCE, GCG_PY, PRECOMPUTED_PY],
    "autodan": [NAIVE_INFERENCE, AUTODAN_PY, PRECOMPUTED_PY],
    "pair": [NAIVE_INFERENCE, PAIR_PY, PRECOMPUTED_PY],
}


def file_bits(path: Path) -> int:
    """Return the UTF-8 bit-length of one file.

    Raises if the file is missing.
    """
    return len(path.read_bytes()) * 8


def code_bits(paths: Iterable[Path]) -> int:
    """Return the sum of `file_bits(p)` over all paths.

    Order-independent (sum, not concatenate-then-encode).
    """
    return sum(file_bits(p) for p in paths)


def gcg_bits() -> dict[str, int]:
    code = code_bits(STRATEGY_CODE_FILES["gcg"])
    return {
        "total": code,
        "code_bits": code,
        "data_bits": 0,
        "auxiliary_model_bits": 0,
    }


def autodan_bits() -> dict[str, int]:
    code = code_bits(STRATEGY_CODE_FILES["autodan"])
    return {
        "total": code,
        "code_bits": code,
        "data_bits": 0,
        "auxiliary_model_bits": 0,
    }


def pair_bits(
    config: dict[str, Any],
    *,
    attacked_model_name: str,
) -> dict[str, int]:
    """Return PAIR program-length bits.

    The auxiliary-model term `16 * helper_params` is included iff
    `config["attack_model"]["id"]` differs from `attacked_model_name`
    (exact-string equality on the canonical `model_name_or_path`).
    """
    code = code_bits(STRATEGY_CODE_FILES["pair"])
    attack_model = config["attack_model"]
    auxiliary = 0
    if attack_model["id"] != attacked_model_name:
        auxiliary = 16 * int(attack_model["num_params"])
    return {
        "total": code + auxiliary,
        "code_bits": code,
        "data_bits": 0,
        "auxiliary_model_bits": auxiliary,
    }


def compute_attack_bits(
    attack: str,
    config: dict[str, Any],
    *,
    attacked_model_name: str | None = None,
) -> int:
    """Dispatch to the per-attack helper and return the scalar `total`.

    `attacked_model_name` is required for PAIR; GCG and AutoDAN ignore it.
    """
    if attack == "gcg":
        return gcg_bits()["total"]
    if attack == "autodan":
        return autodan_bits()["total"]
    if attack == "pair":
        if attacked_model_name is None:
            raise ValueError("PAIR requires attacked_model_name")
        return pair_bits(config, attacked_model_name=attacked_model_name)["total"]
    raise ValueError(f"Unknown attack {attack!r}")
