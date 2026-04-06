"""Tests for SafetyPairHandler Hydra config integration."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "information_safety" / "configs"


class TestHydraConfigLoads:
    def test_safety_pair_yaml_is_valid(self) -> None:
        config_path = CONFIGS_DIR / "algorithm" / "dataset_handler" / "safety_pair.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "_target_" in config
        assert "information_safety.algorithms.dataset_handlers.safety_pair" in config["_target_"]
        assert "include_refusals" in config
        assert "num_benign_examples" in config
        assert "max_length" in config
        assert "batch_size" in config

    def test_model_config_exists(self) -> None:
        config_path = CONFIGS_DIR / "algorithm" / "model" / "llama3_8b_instruct.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "pretrained_model_name_or_path" in config
        assert "Llama" in config["pretrained_model_name_or_path"]


class TestExperimentConfigsDifferOnlyInRefusals:
    def test_safe_and_unsafe_differ_only_in_include_refusals(self) -> None:
        safe_path = CONFIGS_DIR / "experiment" / "safety-pair-safe.yaml"
        unsafe_path = CONFIGS_DIR / "experiment" / "safety-pair-unsafe.yaml"

        assert safe_path.exists(), f"Safe experiment config not found: {safe_path}"
        assert unsafe_path.exists(), f"Unsafe experiment config not found: {unsafe_path}"

        with open(safe_path) as f:
            safe_config = yaml.safe_load(f)
        with open(unsafe_path) as f:
            unsafe_config = yaml.safe_load(f)

        safe_flat = _flatten_dict(safe_config)
        unsafe_flat = _flatten_dict(unsafe_config)

        all_keys = set(safe_flat.keys()) | set(unsafe_flat.keys())
        differences = {}
        for key in all_keys:
            safe_val = safe_flat.get(key)
            unsafe_val = unsafe_flat.get(key)
            if safe_val != unsafe_val:
                differences[key] = (safe_val, unsafe_val)

        assert len(differences) > 0, "Configs should differ in at least one field"

        for key in differences:
            assert "refusal" in key.lower() or "checkpoint" in key.lower() or "name" in key.lower(), (
                f"Unexpected difference in key '{key}': "
                f"safe={differences[key][0]}, unsafe={differences[key][1]}"
            )


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict into dot-separated keys."""
    items: dict = {}
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.update(_flatten_dict(item, f"{new_key}[{i}]"))
                else:
                    items[f"{new_key}[{i}]"] = item
        else:
            items[new_key] = v
    return items
