"""Tests for the WMDP DataStrategy grid in check_results.py."""

from __future__ import annotations

from scripts.check_results import (
    DATA_SWEEP_POINTS,
    HARMBENCH_CONFIGS,
    MODELS,
    WMDP_CONFIGS,
)


def _wmdp_baseline() -> list[dict[str, object]]:
    return [c for c in WMDP_CONFIGS if c["experiment_name"] == "BaselineStrategy"]


def _wmdp_roleplay() -> list[dict[str, object]]:
    return [c for c in WMDP_CONFIGS if c["experiment_name"] == "RoleplayStrategy"]


def _wmdp_data_strategy() -> list[dict[str, object]]:
    return [c for c in WMDP_CONFIGS if c["experiment_name"] == "DataStrategy"]


def test_wmdp_baseline_count_matches_models() -> None:
    assert len(_wmdp_baseline()) == len(MODELS)


def test_wmdp_roleplay_count_matches_models() -> None:
    assert len(_wmdp_roleplay()) == len(MODELS)


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
