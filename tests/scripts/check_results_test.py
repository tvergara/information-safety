"""Tests for the WMDP DataStrategy grid in check_results.py."""

from __future__ import annotations

from scripts.check_results import (
    DATA_SWEEP_POINTS,
    EVILMATH_CONFIGS,
    EVILMATH_DEFENSES,
    HARMBENCH_CONFIGS,
    MODELS,
    PRECOMPUTED_ATTACKS,
    WMDP_CONFIGS,
    WMDP_DEFENSES,
    _matches,
)


def _is_base_model(model_name: str) -> bool:
    return model_name in MODELS


def _wmdp_baseline() -> list[dict[str, object]]:
    return [
        c for c in WMDP_CONFIGS
        if c["experiment_name"] == "BaselineStrategy" and _is_base_model(c["model_name"])
    ]


def _wmdp_data_strategy() -> list[dict[str, object]]:
    return [
        c for c in WMDP_CONFIGS
        if c["experiment_name"] == "DataStrategy" and _is_base_model(c["model_name"])
    ]


def test_wmdp_baseline_count_matches_models() -> None:
    assert len(_wmdp_baseline()) == len(MODELS)


def test_wmdp_each_base_has_roleplay() -> None:
    for model in MODELS:
        roleplay = [
            c for c in WMDP_CONFIGS
            if c["experiment_name"] == "RoleplayStrategy" and c["model_name"] == model
        ]
        assert len(roleplay) == 1


def test_wmdp_base_models_have_no_precomputed_attacks() -> None:
    for model in MODELS:
        found = [
            c for c in WMDP_CONFIGS
            if c["experiment_name"] in PRECOMPUTED_ATTACKS and c["model_name"] == model
        ]
        assert found == []


def test_wmdp_data_strategy_total_epoch_rows() -> None:
    total_epochs_per_model = sum(max_epochs for _, max_epochs in DATA_SWEEP_POINTS)
    expected = len(MODELS) * total_epochs_per_model * 3
    assert len(_wmdp_data_strategy()) == expected


def test_wmdp_data_strategy_includes_mix_zero_rows() -> None:
    mix_zero = [
        c
        for c in _wmdp_data_strategy()
        if c["corpus_fraction"] == 0.0
    ]
    total_epochs_per_model = sum(max_epochs for _, max_epochs in DATA_SWEEP_POINTS)
    expected = len(MODELS) * total_epochs_per_model
    assert len(mix_zero) == expected
    for c in mix_zero:
        assert c["corpus_subset"] is None


def test_wmdp_data_strategy_includes_mix_half_bio_rows() -> None:
    bio_rows = [
        c
        for c in _wmdp_data_strategy()
        if c["corpus_fraction"] == 0.5 and c["corpus_subset"] == "bio"
    ]
    total_epochs_per_model = sum(max_epochs for _, max_epochs in DATA_SWEEP_POINTS)
    expected = len(MODELS) * total_epochs_per_model
    assert len(bio_rows) == expected


def test_wmdp_data_strategy_includes_mix_half_cyber_rows() -> None:
    cyber_rows = [
        c
        for c in _wmdp_data_strategy()
        if c["corpus_fraction"] == 0.5 and c["corpus_subset"] == "cyber"
    ]
    total_epochs_per_model = sum(max_epochs for _, max_epochs in DATA_SWEEP_POINTS)
    expected = len(MODELS) * total_epochs_per_model
    assert len(cyber_rows) == expected


def test_wmdp_data_strategy_no_chem_subset() -> None:
    subsets = {
        c["corpus_subset"]
        for c in _wmdp_data_strategy()
        if c["corpus_subset"] is not None
    }
    assert subsets == {"bio", "cyber"}


def test_wmdp_data_strategy_has_expected_keys() -> None:
    sample = _wmdp_data_strategy()[0]
    for key in (
        "experiment_name",
        "dataset_name",
        "model_name",
        "max_examples",
        "max_epochs",
        "epoch",
        "corpus_fraction",
        "corpus_subset",
    ):
        assert key in sample


def test_wmdp_dataset_name_is_wmdp() -> None:
    for c in WMDP_CONFIGS:
        assert c["dataset_name"] == "wmdp"


def test_harmbench_configs_unchanged_no_corpus_keys() -> None:
    for c in HARMBENCH_CONFIGS:
        assert "corpus_fraction" not in c
        assert "corpus_subset" not in c


def _evilmath_baseline() -> list[dict[str, object]]:
    return [
        c for c in EVILMATH_CONFIGS
        if c["experiment_name"] == "BaselineStrategy" and _is_base_model(c["model_name"])
    ]


def _evilmath_data_strategy() -> list[dict[str, object]]:
    return [
        c for c in EVILMATH_CONFIGS
        if c["experiment_name"] == "DataStrategy" and _is_base_model(c["model_name"])
    ]


def test_evilmath_baseline_count_matches_models() -> None:
    assert len(_evilmath_baseline()) == len(MODELS)


def test_evilmath_each_base_has_roleplay() -> None:
    for model in MODELS:
        roleplay = [
            c for c in EVILMATH_CONFIGS
            if c["experiment_name"] == "RoleplayStrategy" and c["model_name"] == model
        ]
        assert len(roleplay) == 1


def test_evilmath_base_models_have_no_precomputed_attacks() -> None:
    for model in MODELS:
        found = [
            c for c in EVILMATH_CONFIGS
            if c["experiment_name"] in PRECOMPUTED_ATTACKS and c["model_name"] == model
        ]
        assert found == []


def test_evilmath_data_strategy_total_epoch_rows() -> None:
    total_epochs_per_model = sum(max_epochs for _, max_epochs in DATA_SWEEP_POINTS)
    expected = len(MODELS) * total_epochs_per_model
    assert len(_evilmath_data_strategy()) == expected


def test_evilmath_data_strategy_no_corpus_keys() -> None:
    for c in _evilmath_data_strategy():
        assert "corpus_fraction" not in c
        assert "corpus_subset" not in c


def test_evilmath_data_strategy_has_expected_keys() -> None:
    sample = _evilmath_data_strategy()[0]
    for key in (
        "experiment_name",
        "dataset_name",
        "model_name",
        "max_examples",
        "max_epochs",
        "epoch",
    ):
        assert key in sample


def test_evilmath_dataset_name_is_evilmath() -> None:
    for c in EVILMATH_CONFIGS:
        assert c["dataset_name"] == "evilmath"


# ----- Defense matrix -----

def _wmdp_defense_configs(defense: str) -> list[dict[str, object]]:
    return [c for c in WMDP_CONFIGS if c["model_name"] == defense]


def _evilmath_defense_configs(defense: str) -> list[dict[str, object]]:
    return [c for c in EVILMATH_CONFIGS if c["model_name"] == defense]


def test_wmdp_has_three_defenses() -> None:
    assert len(WMDP_DEFENSES) == 3


def test_evilmath_has_three_defenses() -> None:
    assert len(EVILMATH_DEFENSES) == 3


def test_wmdp_each_defense_has_baseline() -> None:
    for defense in WMDP_DEFENSES:
        baselines = [
            c for c in _wmdp_defense_configs(defense)
            if c["experiment_name"] == "BaselineStrategy"
        ]
        assert len(baselines) == 1


def test_wmdp_each_defense_has_roleplay() -> None:
    for defense in WMDP_DEFENSES:
        roleplay = [
            c for c in _wmdp_defense_configs(defense)
            if c["experiment_name"] == "RoleplayStrategy"
        ]
        assert len(roleplay) == 1


def test_wmdp_each_defense_has_precomputed_attacks() -> None:
    for defense in WMDP_DEFENSES:
        found = {
            c["experiment_name"] for c in _wmdp_defense_configs(defense)
            if c["experiment_name"] in PRECOMPUTED_ATTACKS
        }
        assert found == set(PRECOMPUTED_ATTACKS)


def test_wmdp_each_defense_has_all_corpus_variants() -> None:
    for defense in WMDP_DEFENSES:
        ds = [
            c for c in _wmdp_defense_configs(defense)
            if c["experiment_name"] == "DataStrategy"
        ]
        variants = {(c["corpus_fraction"], c["corpus_subset"]) for c in ds}
        assert variants == {(0.0, None), (0.5, "bio"), (0.5, "cyber")}


def test_wmdp_defense_data_strategy_total_epoch_rows() -> None:
    total_epochs = sum(max_epochs for _, max_epochs in DATA_SWEEP_POINTS)
    for defense in WMDP_DEFENSES:
        ds = [
            c for c in _wmdp_defense_configs(defense)
            if c["experiment_name"] == "DataStrategy"
        ]
        assert len(ds) == total_epochs * 3


def test_evilmath_each_defense_has_baseline() -> None:
    for defense in EVILMATH_DEFENSES:
        baselines = [
            c for c in _evilmath_defense_configs(defense)
            if c["experiment_name"] == "BaselineStrategy"
        ]
        assert len(baselines) == 1


def test_evilmath_each_defense_has_roleplay() -> None:
    for defense in EVILMATH_DEFENSES:
        roleplay = [
            c for c in _evilmath_defense_configs(defense)
            if c["experiment_name"] == "RoleplayStrategy"
        ]
        assert len(roleplay) == 1


def test_evilmath_each_defense_has_precomputed_attacks() -> None:
    for defense in EVILMATH_DEFENSES:
        found = {
            c["experiment_name"] for c in _evilmath_defense_configs(defense)
            if c["experiment_name"] in PRECOMPUTED_ATTACKS
        }
        assert found == set(PRECOMPUTED_ATTACKS)


def test_evilmath_defense_data_strategy_total_epoch_rows() -> None:
    total_epochs = sum(max_epochs for _, max_epochs in DATA_SWEEP_POINTS)
    for defense in EVILMATH_DEFENSES:
        ds = [
            c for c in _evilmath_defense_configs(defense)
            if c["experiment_name"] == "DataStrategy"
        ]
        assert len(ds) == total_epochs


def test_evilmath_defense_data_strategy_no_corpus_keys() -> None:
    for defense in EVILMATH_DEFENSES:
        ds = [
            c for c in _evilmath_defense_configs(defense)
            if c["experiment_name"] == "DataStrategy"
        ]
        for c in ds:
            assert "corpus_fraction" not in c
            assert "corpus_subset" not in c


# ----- Substring model_name matching -----

def test_matches_substring_model_name() -> None:
    config = {
        "experiment_name": "BaselineStrategy",
        "dataset_name": "wmdp",
        "model_name": "sft-wmdp-Llama-3.1-8B-Instruct-ec55867d84a0",
        "max_examples": None,
        "epoch": 0,
    }
    row = {
        "experiment_name": "BaselineStrategy",
        "dataset_name": "wmdp",
        "model_name": "/network/scratch/b/brownet/information-safety/defenses/"
                      "sft-wmdp-Llama-3.1-8B-Instruct-ec55867d84a0",
        "max_examples": None,
        "epoch": 0,
        "asr": 0.5,
    }
    assert _matches(row, config)


def test_matches_does_not_match_different_defense() -> None:
    config = {
        "experiment_name": "BaselineStrategy",
        "dataset_name": "wmdp",
        "model_name": "sft-wmdp-Llama-3.1-8B-Instruct-ec55867d84a0",
        "max_examples": None,
        "epoch": 0,
    }
    row = {
        "experiment_name": "BaselineStrategy",
        "dataset_name": "wmdp",
        "model_name": "/network/scratch/b/brownet/information-safety/defenses/"
                      "cb-wmdp-Llama-3.1-8B-Instruct-bfbf3e38793c",
        "max_examples": None,
        "epoch": 0,
        "asr": 0.5,
    }
    assert not _matches(row, config)


def test_matches_base_model_unaffected_by_substring() -> None:
    config = {
        "experiment_name": "BaselineStrategy",
        "dataset_name": "wmdp",
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "max_examples": None,
        "epoch": 0,
    }
    row_base = {
        "experiment_name": "BaselineStrategy",
        "dataset_name": "wmdp",
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "max_examples": None,
        "epoch": 0,
        "asr": 0.5,
    }
    row_defense = {
        "experiment_name": "BaselineStrategy",
        "dataset_name": "wmdp",
        "model_name": "/network/scratch/b/brownet/information-safety/defenses/"
                      "sft-wmdp-Llama-3.1-8B-Instruct-ec55867d84a0",
        "max_examples": None,
        "epoch": 0,
        "asr": 0.5,
    }
    assert _matches(row_base, config)
    assert not _matches(row_defense, config)
