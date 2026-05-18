"""Tests for scripts/audit_queue_dedup.py — the cross-cluster queue audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_queue_dedup import (
    NIBI_LIVE_PAT,
    TRC_LIVE_PAT,
    Category,
    attack_opt_output_path,
    classify_attack,
    classify_ds,
    classify_spec,
    expected_rows,
    load_results,
    parse_live_ids,
)


class TestRegexParsing:
    def test_trc_pattern_matches_new_format(self) -> None:
        line = "is_0ae8d90216771bd71779052785"
        m = TRC_LIVE_PAT.search(line)
        assert m is not None and m.group(1) == "0ae8d90216771bd7"

    def test_trc_pattern_matches_pool_underscore_format(self) -> None:
        line = "is_pool_2286c695d8ab5a4e_0"
        m = TRC_LIVE_PAT.search(line)
        assert m is not None and m.group(1) == "2286c695d8ab5a4e"

    def test_nibi_pattern_extracts_id(self) -> None:
        line = "nibi-pool-af16d9d9d639529a"
        m = NIBI_LIVE_PAT.search(line)
        assert m is not None and m.group(1) == "af16d9d9d639529a"

    def test_parse_live_ids_skips_non_pool_jobs(self) -> None:
        text = "\n".join([
            "is_0ae8d90216771bd71779052785",
            "vllm_gpt_oss_20b_20260517184006",
            "fg_cpu_101",
            "is_pool_2286c695d8ab5a4e_0",
        ])
        assert parse_live_ids(text, TRC_LIVE_PAT) == {
            "0ae8d90216771bd7",
            "2286c695d8ab5a4e",
        }


class TestExpectedRows:
    def test_data_strategy_wmdp_emits_one_row_per_epoch(self) -> None:
        cfg = {
            "experiment_name": "DataStrategy",
            "dataset_name": "wmdp",
            "model_name": "x/y",
            "max_examples": 64,
            "max_epochs": 4,
            "corpus_fraction": 0.5,
            "corpus_subset": "bio",
        }
        rows = expected_rows(cfg)
        assert [r["epoch"] for r in rows] == [0, 1, 2, 3]
        assert all(r["corpus_fraction"] == 0.5 and r["corpus_subset"] == "bio" for r in rows)

    def test_data_strategy_non_wmdp_omits_corpus_fields(self) -> None:
        cfg = {
            "experiment_name": "DataStrategy",
            "dataset_name": "advbench_harmbench",
            "model_name": "x/y",
            "max_examples": 16,
            "max_epochs": 2,
        }
        rows = expected_rows(cfg)
        assert len(rows) == 2
        assert "corpus_fraction" not in rows[0]
        assert "corpus_subset" not in rows[0]

    def test_baseline_emits_single_epoch_zero_row(self) -> None:
        cfg = {
            "experiment_name": "BaselineStrategy",
            "dataset_name": "advbench_harmbench",
            "model_name": "x/y",
            "max_examples": None,
        }
        assert expected_rows(cfg) == [
            {
                "experiment_name": "BaselineStrategy",
                "dataset_name": "advbench_harmbench",
                "model_name": "x/y",
                "max_examples": None,
                "epoch": 0,
            }
        ]


class TestClassifyDs:
    @staticmethod
    def _row(model: str, exp: str, max_ex: int | None, epoch: int, asr: float | None,
             dataset: str = "advbench_harmbench", **extra: object) -> dict[str, object]:
        return {
            "experiment_name": exp,
            "dataset_name": dataset,
            "model_name": model,
            "max_examples": max_ex,
            "epoch": epoch,
            "asr": asr,
            **extra,
        }

    def test_missing_when_no_rows_exist(self) -> None:
        cfg = {
            "experiment_name": "DataStrategy",
            "dataset_name": "advbench_harmbench",
            "model_name": "Llama-3-8B",
            "max_examples": 16,
            "max_epochs": 2,
        }
        assert classify_ds(cfg, []) == Category.MISSING

    def test_partial_when_some_epochs_missing(self) -> None:
        cfg = {
            "experiment_name": "DataStrategy",
            "dataset_name": "advbench_harmbench",
            "model_name": "Llama-3-8B",
            "max_examples": 16,
            "max_epochs": 3,
        }
        rows = [self._row("org/Llama-3-8B", "DataStrategy", 16, 0, 0.42)]
        assert classify_ds(cfg, rows) == Category.PARTIAL

    def test_all_complete_when_every_epoch_has_asr(self) -> None:
        cfg = {
            "experiment_name": "DataStrategy",
            "dataset_name": "advbench_harmbench",
            "model_name": "Llama-3-8B",
            "max_examples": 16,
            "max_epochs": 2,
        }
        rows = [
            self._row("org/Llama-3-8B", "DataStrategy", 16, 0, 0.42),
            self._row("org/Llama-3-8B", "DataStrategy", 16, 1, 0.51),
        ]
        assert classify_ds(cfg, rows) == Category.ALL_COMPLETE

    def test_needs_eval_when_rows_exist_but_asr_null(self) -> None:
        cfg = {
            "experiment_name": "DataStrategy",
            "dataset_name": "advbench_harmbench",
            "model_name": "Llama-3-8B",
            "max_examples": 16,
            "max_epochs": 2,
        }
        rows = [
            self._row("org/Llama-3-8B", "DataStrategy", 16, 0, 0.42),
            self._row("org/Llama-3-8B", "DataStrategy", 16, 1, None),
        ]
        assert classify_ds(cfg, rows) == Category.NEEDS_EVAL

    def test_wmdp_corpus_subset_distinguishes_rows(self) -> None:
        cfg = {
            "experiment_name": "DataStrategy",
            "dataset_name": "wmdp",
            "model_name": "Llama-3-8B",
            "max_examples": 64,
            "max_epochs": 2,
            "corpus_fraction": 0.5,
            "corpus_subset": "bio",
        }
        rows = [
            self._row("org/Llama-3-8B", "DataStrategy", 64, 0, 0.4, dataset="wmdp",
                       corpus_fraction=0.5, corpus_subset="cyber"),
            self._row("org/Llama-3-8B", "DataStrategy", 64, 1, 0.5, dataset="wmdp",
                       corpus_fraction=0.5, corpus_subset="cyber"),
        ]
        assert classify_ds(cfg, rows) == Category.MISSING


class TestLoadResults:
    def test_drops_invalidated_rows(self, tmp_path: Path) -> None:
        results = tmp_path / "r.jsonl"
        results.write_text(
            json.dumps({"epoch": 0, "invalidated_reason": None}) + "\n"
            + json.dumps({"epoch": 1, "invalidated_reason": "bad seed"}) + "\n"
            + json.dumps({"epoch": 2}) + "\n"
        )
        rows = load_results(results)
        assert [r["epoch"] for r in rows] == [0, 2]


class TestClassifyAttack:
    def _spec(self, output_jsonl: str) -> dict[str, object]:
        return {
            "id": "abc",
            "command": [
                "bash", "run_one_attack.sh",
                "--attack", "pair",
                "--output-jsonl", output_jsonl,
            ],
        }

    def test_all_complete_when_local_shard_exists(self, tmp_path: Path) -> None:
        local = tmp_path / "attacks" / "pair-x-shard0.jsonl"
        local.parent.mkdir()
        local.write_text("")
        tamia_path = "/scratch/t/tvergara/information-safety/attacks/pair-x-shard0.jsonl"
        spec = self._spec(tamia_path)
        assert classify_attack(spec, mila_attacks_dir=tmp_path / "attacks") == Category.ALL_COMPLETE

    def test_missing_when_shard_absent(self, tmp_path: Path) -> None:
        (tmp_path / "attacks").mkdir()
        tamia_path = "/scratch/t/tvergara/information-safety/attacks/pair-x-shard0.jsonl"
        spec = self._spec(tamia_path)
        assert classify_attack(spec, mila_attacks_dir=tmp_path / "attacks") == Category.MISSING

    def test_attack_opt_output_path_strips_tamia_prefix(self) -> None:
        spec = self._spec("/scratch/t/tvergara/information-safety/attacks/pair-x.jsonl")
        path = attack_opt_output_path(spec, mila_attacks_dir=Path("/network/scratch/b/brownet/information-safety/attacks"))
        assert path == Path("/network/scratch/b/brownet/information-safety/attacks/pair-x.jsonl")


class TestClassifySpec:
    def test_live_trc_short_circuits_classification(self, tmp_path: Path) -> None:
        spec = {"id": "abc", "config": {
            "experiment_name": "DataStrategy",
            "dataset_name": "advbench_harmbench",
            "model_name": "x", "max_examples": 16, "max_epochs": 1,
        }}
        assert classify_spec(
            spec, results_rows=[], mila_attacks_dir=tmp_path,
            trc_live_ids={"abc"}, nibi_live_ids=set(),
        ) == Category.LIVE_TRC

    def test_live_nibi_short_circuits_attack_classification(self, tmp_path: Path) -> None:
        spec = {"id": "abc", "command": ["bash", "--output-jsonl", "/tmp/x.jsonl"]}
        assert classify_spec(
            spec, results_rows=[], mila_attacks_dir=tmp_path,
            trc_live_ids=set(), nibi_live_ids={"abc"},
        ) == Category.LIVE_NIBI

    def test_attack_spec_routed_when_no_config_field(self, tmp_path: Path) -> None:
        spec = {"id": "abc", "command": ["bash", "--output-jsonl",
                                          "/scratch/t/tvergara/information-safety/attacks/x.jsonl"]}
        result = classify_spec(
            spec, results_rows=[], mila_attacks_dir=tmp_path,
            trc_live_ids=set(), nibi_live_ids=set(),
        )
        assert result == Category.MISSING


class TestCategoryEnum:
    def test_wasted_categories(self) -> None:
        assert Category.LIVE_TRC.is_wasted
        assert Category.LIVE_NIBI.is_wasted
        assert Category.ALL_COMPLETE.is_wasted
        assert Category.NEEDS_EVAL.is_wasted

    def test_needed_categories(self) -> None:
        assert not Category.PARTIAL.is_wasted
        assert not Category.MISSING.is_wasted
        assert not Category.PENDING_EVAL.is_wasted


class TestPendingEvalCategory:
    def test_eval_spec_classified_as_pending_eval(self, tmp_path: Path) -> None:
        spec = {
            "spec_id": "abcdef0123456789",
            "epoch": 0,
            "base_model": "tiny",
            "adapter_path": "/work/adapters/abcdef0123456789/epoch_0",
            "eval_meta": {"spec_id": "abcdef0123456789", "epoch": 0},
        }
        result = classify_spec(
            spec, results_rows=[], mila_attacks_dir=tmp_path,
            trc_live_ids=set(), nibi_live_ids=set(),
        )
        assert result == Category.PENDING_EVAL

    def test_pending_eval_not_quarantined(self) -> None:
        assert not Category.PENDING_EVAL.is_wasted


class TestDataStrategyAliasInRowMatching:
    def test_data_strategy_config_satisfied_by_deferred_eval_row(self) -> None:
        from scripts.audit_queue_dedup import _row_matches

        cfg = {
            "experiment_name": "DataStrategy",
            "dataset_name": "advbench_harmbench",
            "model_name": "Llama-3-8B",
            "max_examples": 16,
            "epoch": 0,
        }
        row = {
            "experiment_name": "DataStrategyDeferredEval",
            "dataset_name": "advbench_harmbench",
            "model_name": "org/Llama-3-8B",
            "max_examples": 16,
            "epoch": 0,
        }
        assert _row_matches(row, cfg)

    def test_deferred_eval_spec_classified_complete_by_legacy_row(self) -> None:
        cfg = {
            "experiment_name": "DataStrategyDeferredEval",
            "dataset_name": "advbench_harmbench",
            "model_name": "Llama-3-8B",
            "max_examples": 16,
            "max_epochs": 1,
        }
        rows = [
            {
                "experiment_name": "DataStrategy",
                "dataset_name": "advbench_harmbench",
                "model_name": "org/Llama-3-8B",
                "max_examples": 16,
                "epoch": 0,
                "asr": 0.4,
            },
        ]
        assert classify_ds(cfg, rows) == Category.ALL_COMPLETE


@pytest.fixture
def fake_pending_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pending"
    d.mkdir()
    (d / "ds01.json").write_text(json.dumps({
        "id": "ds01",
        "command": ["python"],
        "config": {
            "experiment_name": "DataStrategy",
            "dataset_name": "advbench_harmbench",
            "model_name": "Llama-3-8B",
            "max_examples": 16,
            "max_epochs": 1,
        },
    }))
    (d / "atk01.json").write_text(json.dumps({
        "id": "atk01",
        "command": ["bash", "--output-jsonl",
                    "/scratch/t/tvergara/information-safety/attacks/x.jsonl"],
    }))
    return d


class TestEnd2End:
    def test_audit_categorizes_both_spec_types(self, fake_pending_dir: Path, tmp_path: Path) -> None:
        from scripts.audit_queue_dedup import audit_pending

        attacks = tmp_path / "attacks"
        attacks.mkdir()
        results = tmp_path / "results.jsonl"
        results.write_text("")  # no completed rows

        categories = audit_pending(
            pending_dir=fake_pending_dir,
            results_file=results,
            mila_attacks_dir=attacks,
            trc_live_ids=set(),
            nibi_live_ids=set(),
        )
        assert categories[Category.MISSING] == ["atk01", "ds01"]
