"""Verify AdversariaLLM model-registry entries for the 6 defense_ids."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.check_results import EVILMATH_DEFENSES, WMDP_DEFENSES

MODELS_YAML = (
    Path(__file__).resolve().parents[2]
    / "third_party/adversariallm/conf/models/models.yaml"
)
EXPECTED_DEFENSE_PATH_ROOT = "/scratch/t/tvergara/information-safety/defenses"
TOKENIZER_ID = "meta-llama/Llama-3.1-8B-Instruct"


def _load_registry() -> dict[str, dict[str, object]]:
    with open(MODELS_YAML) as f:
        return yaml.safe_load(f)


def test_all_wmdp_defenses_present() -> None:
    registry = _load_registry()
    for defense in WMDP_DEFENSES:
        assert defense in registry


def test_all_evilmath_defenses_present() -> None:
    registry = _load_registry()
    for defense in EVILMATH_DEFENSES:
        assert defense in registry


def test_defense_id_points_to_scratch_path() -> None:
    registry = _load_registry()
    for defense in WMDP_DEFENSES + EVILMATH_DEFENSES:
        entry = registry[defense]
        assert entry["id"] == f"{EXPECTED_DEFENSE_PATH_ROOT}/{defense}"


def test_defense_tokenizer_is_llama_31_8b_instruct() -> None:
    registry = _load_registry()
    for defense in WMDP_DEFENSES + EVILMATH_DEFENSES:
        entry = registry[defense]
        assert entry["tokenizer_id"] == TOKENIZER_ID


def test_defense_entries_have_expected_keys() -> None:
    registry = _load_registry()
    expected_keys = {
        "id",
        "tokenizer_id",
        "short_name",
        "developer_name",
        "compile",
        "dtype",
        "chat_template",
        "trust_remote_code",
    }
    for defense in WMDP_DEFENSES + EVILMATH_DEFENSES:
        entry = registry[defense]
        assert set(entry.keys()) == expected_keys


def test_defense_uses_llama_chat_template_and_bf16() -> None:
    registry = _load_registry()
    for defense in WMDP_DEFENSES + EVILMATH_DEFENSES:
        entry = registry[defense]
        assert entry["chat_template"] == "llama-3-instruct"
        assert entry["dtype"] == "bfloat16"
        assert entry["compile"] is False
        assert entry["trust_remote_code"] is True
        assert entry["short_name"] == "Llama"
