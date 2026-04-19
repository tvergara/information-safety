"""Per-attack bits accounting for GCG / AutoDAN / PAIR.

Matches the upstream Xarangi/AdversariaLLM reference implementation
(`scripts/plot_program_length_vs_asr.py`, commit `a799e7f`), with the single
exception that the PAIR helper-model parameter term
(`helper_model_params * helper_precision_bits`) is dropped — we do not count
auxiliary model weights as part of the attack's program length.

All formulas and constants (including `TOKEN_ID_BITS = 16`) match the reference
verbatim for the components we emit.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

TOKEN_ID_BITS: int = 16

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_ATTACKS_DIR: Path = _REPO_ROOT / "third_party" / "adversariallm" / "adversariallm" / "attacks"

GCG_SOURCE: str = str(_ATTACKS_DIR / "gcg.py")
AUTODAN_SOURCE: str = str(_ATTACKS_DIR / "autodan.py")
PAIR_SOURCE: str = str(_ATTACKS_DIR / "pair.py")

_ATTACK_CLASS_NAME: dict[str, str] = {
    "gcg": "GCGAttack",
    "autodan": "AutoDANAttack",
    "pair": "PAIRAttack",
}


def class_code_bits(py_file: str, class_name: str) -> int:
    """Return the UTF-8 bit-length of the source segment defining `class_name`.

    AST-parses `py_file`, locates the top-level class named `class_name`, and
    returns `len(source_segment.encode("utf-8")) * 8`. Raises `ValueError` if
    the class is not found.
    """
    source = Path(py_file).read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            seg = "".join(lines[node.lineno - 1 : node.end_lineno])
            return len(seg.encode("utf-8")) * 8
    raise ValueError(f"Missing class {class_name!r} in {py_file}")


def parse_autodan_initial_strings_bits(autodan_file: str) -> tuple[int, int]:
    """Return `(avg_utf8_bits_per_string, count)` for the `INITIAL_STRINGS` list.

    AST-parses `autodan_file`, finds the top-level `INITIAL_STRINGS = [...]`
    assignment, and averages the UTF-8 bit-lengths of its string literals.
    Raises `ValueError` if no such assignment is found or the list is empty.
    """
    source = Path(autodan_file).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Name) and target.id == "INITIAL_STRINGS"):
                continue
            if not isinstance(node.value, ast.List):
                continue
            strings: list[str] = []
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    strings.append(elt.value)
            if not strings:
                raise ValueError(f"INITIAL_STRINGS in {autodan_file} has no string entries")
            total_bits = sum(len(s.encode("utf-8")) * 8 for s in strings)
            avg_bits = int(round(total_bits / len(strings)))
            return avg_bits, len(strings)
    raise ValueError(f"INITIAL_STRINGS not found in {autodan_file}")


def gcg_bits(config: dict[str, Any]) -> dict[str, int]:
    """Return bits components for a GCG run config.

    Keys: `total`, `code_bits`, `search_bits`, `payload_bits`, `meta_bits`.

    Formula (reference):
      suffix_tokens = len(str(optim_str_init).split())
      search = num_steps * suffix_tokens * log2(max(topk, 2))
      payload = suffix_tokens * TOKEN_ID_BITS
      meta = num_steps * TOKEN_ID_BITS
    """
    num_steps = int(config["num_steps"])
    topk = int(config["topk"])
    suffix_tokens = len(str(config["optim_str_init"]).split())

    code_bits = class_code_bits(GCG_SOURCE, _ATTACK_CLASS_NAME["gcg"])
    search_bits = int(round(num_steps * suffix_tokens * math.log2(max(topk, 2))))
    payload_bits = suffix_tokens * TOKEN_ID_BITS
    meta_bits = num_steps * TOKEN_ID_BITS
    total = code_bits + search_bits + payload_bits + meta_bits

    return {
        "total": total,
        "code_bits": code_bits,
        "search_bits": search_bits,
        "payload_bits": payload_bits,
        "meta_bits": meta_bits,
    }


def autodan_bits(config: dict[str, Any]) -> dict[str, int]:
    """Return bits components for an AutoDAN run config.

    Keys: `total`, `code_bits`, `initial_pool_bits`, `evolution_bits`, `mutation_bits`.

    Formula (reference):
      evolution = num_steps * batch_size * log2(max(batch_size, 2))
      mutation = num_steps * batch_size * mutation_rate * TOKEN_ID_BITS
      initial_pool_bits = avg UTF-8 bits over INITIAL_STRINGS
    """
    num_steps = int(config["num_steps"])
    batch_size = int(config["batch_size"])
    mutation_rate = float(config["mutation"])

    code_bits = class_code_bits(AUTODAN_SOURCE, _ATTACK_CLASS_NAME["autodan"])
    initial_pool_bits, _ = parse_autodan_initial_strings_bits(AUTODAN_SOURCE)
    evolution_bits = int(round(num_steps * batch_size * math.log2(max(batch_size, 2))))
    mutation_bits = int(round(num_steps * batch_size * mutation_rate * TOKEN_ID_BITS))
    total = code_bits + initial_pool_bits + evolution_bits + mutation_bits

    return {
        "total": total,
        "code_bits": code_bits,
        "initial_pool_bits": initial_pool_bits,
        "evolution_bits": evolution_bits,
        "mutation_bits": mutation_bits,
    }


def pair_bits(config: dict[str, Any]) -> dict[str, int]:
    """Return bits components for a PAIR run config, excluding helper weights.

    Keys: `total`, `code_bits`, `payload_bits`, `search_bits`.

    Formula (reference, minus helper-model parameter bits):
      payload = num_steps * num_streams * max_new_tokens * TOKEN_ID_BITS
      search = num_steps * num_streams * max_attempts * log2(max(max_new_tokens, 2))

    The reference also adds `helper_model_params * helper_precision_bits`
    (Vicuna-13B × fp16 = 13e9 * 16). We intentionally drop that term — we do
    not count auxiliary model weights as part of the attack's program length.
    """
    num_steps = int(config["num_steps"])
    num_streams = int(config["num_streams"])
    attack_model = config["attack_model"]
    max_new_tokens = int(attack_model["max_new_tokens"])
    max_attempts = int(attack_model["max_attempts"])

    code_bits = class_code_bits(PAIR_SOURCE, _ATTACK_CLASS_NAME["pair"])
    payload_bits = num_steps * num_streams * max_new_tokens * TOKEN_ID_BITS
    search_bits = int(
        round(num_steps * num_streams * max_attempts * math.log2(max(max_new_tokens, 2)))
    )
    total = code_bits + payload_bits + search_bits

    return {
        "total": total,
        "code_bits": code_bits,
        "payload_bits": payload_bits,
        "search_bits": search_bits,
    }


def compute_attack_bits(attack: str, config: dict[str, Any]) -> int:
    """Dispatch to the appropriate bits function and return the scalar `total`.

    Raises `ValueError` for any attack name outside {gcg, autodan, pair}.
    """
    if attack == "gcg":
        return gcg_bits(config)["total"]
    if attack == "autodan":
        return autodan_bits(config)["total"]
    if attack == "pair":
        return pair_bits(config)["total"]
    raise ValueError(f"Unknown attack {attack!r}")
