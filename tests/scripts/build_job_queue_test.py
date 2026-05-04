"""Tests for the cluster job-pool queue producer."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.build_job_queue import (
    build_command,
    deterministic_id,
    is_present,
    iter_missing_configs,
    main,
    resolve_suffix_file,
    write_pending_jobs,
)


def _baseline_config(model: str = "meta-llama/Llama-2-7b-chat-hf") -> dict[str, object]:
    return {
        "experiment_name": "BaselineStrategy",
        "dataset_name": "advbench_harmbench",
        "model_name": model,
        "max_examples": None,
        "epoch": 0,
    }


def _data_config(model: str, max_ex: int | None, epoch: int) -> dict[str, object]:
    return {
        "experiment_name": "DataStrategy",
        "dataset_name": "advbench_harmbench",
        "model_name": model,
        "max_examples": max_ex,
        "epoch": epoch,
    }


def _gcg_config(model: str) -> dict[str, object]:
    return {
        "experiment_name": "PrecomputedGCGStrategy",
        "dataset_name": "advbench_harmbench",
        "model_name": model,
        "max_examples": None,
        "epoch": 0,
    }


def _wmdp_config(model: str) -> dict[str, object]:
    return {
        "experiment_name": "BaselineStrategy",
        "dataset_name": "wmdp",
        "model_name": model,
        "max_examples": None,
        "epoch": 0,
    }


def test_is_present_baseline_matches_existing_row() -> None:
    config = _baseline_config()
    rows = [{**config, "asr": 0.1}]
    assert is_present(config, rows) is True


def test_is_present_baseline_missing() -> None:
    config = _baseline_config()
    assert is_present(config, []) is False


def test_is_present_data_strategy_dedups_by_last_epoch() -> None:
    config_epoch0 = _data_config("meta-llama/Llama-2-7b-chat-hf", 10, 0)
    config_epoch1 = _data_config("meta-llama/Llama-2-7b-chat-hf", 10, 1)
    rows = [{**config_epoch1, "asr": 0.5}]
    assert is_present(config_epoch0, rows) is True
    assert is_present(config_epoch1, rows) is True


def test_is_present_data_strategy_partial_run_not_done() -> None:
    config_epoch0 = _data_config("meta-llama/Llama-2-7b-chat-hf", 10, 0)
    config_epoch1 = _data_config("meta-llama/Llama-2-7b-chat-hf", 10, 1)
    rows = [{**config_epoch0, "asr": 0.4}]
    assert is_present(config_epoch1, rows) is False


def test_iter_missing_dedups_data_strategy_epochs() -> None:
    configs = [
        _data_config("meta-llama/Llama-2-7b-chat-hf", 10, 0),
        _data_config("meta-llama/Llama-2-7b-chat-hf", 10, 1),
    ]
    missing = list(iter_missing_configs(configs, []))
    assert len(missing) == 1


def test_resolve_suffix_file_canonical(tmp_path: Path) -> None:
    attacks_dir = tmp_path / "attacks"
    attacks_dir.mkdir()
    canonical = attacks_dir / "gcg-meta-llama_Llama-2-7b-chat-hf.jsonl"
    canonical.write_text("")
    resolved = resolve_suffix_file(
        attack="gcg",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        attacks_dir=attacks_dir,
    )
    assert resolved == canonical


def test_resolve_suffix_file_glob_fallback_picks_newest(tmp_path: Path) -> None:
    attacks_dir = tmp_path / "attacks"
    attacks_dir.mkdir()
    older = attacks_dir / "gcg-meta-llama_Llama-2-7b-chat-hf-1.jsonl"
    newer = attacks_dir / "gcg-meta-llama_Llama-2-7b-chat-hf-2.jsonl"
    older.write_text("")
    newer.write_text("")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    resolved = resolve_suffix_file(
        attack="gcg",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        attacks_dir=attacks_dir,
    )
    assert resolved == newer


def test_resolve_suffix_file_returns_none_when_missing(tmp_path: Path) -> None:
    attacks_dir = tmp_path / "attacks"
    attacks_dir.mkdir()
    resolved = resolve_suffix_file(
        attack="gcg",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        attacks_dir=attacks_dir,
    )
    assert resolved is None


def test_build_command_baseline_emits_expected_overrides() -> None:
    config = _baseline_config()
    cmd = build_command(config, attacks_dir=Path("/tmp/attacks"))
    assert isinstance(cmd, list)
    assert all(isinstance(x, str) for x in cmd)
    assert cmd[0] == "python"
    assert cmd[1] == "information_safety/main.py"
    assert "experiment=prompt-attack" in cmd
    assert "algorithm/strategy=baseline" in cmd
    assert "algorithm/dataset_handler=advbench_harmbench" in cmd
    assert any(c.startswith("algorithm.model.pretrained_model_name_or_path=") for c in cmd)
    assert "algorithm.model.trust_remote_code=false" in cmd
    assert "trainer.precision=bf16-mixed" in cmd


def test_build_command_baseline_wmdp_uses_wmdp_handler() -> None:
    config = _wmdp_config("meta-llama/Llama-2-7b-chat-hf")
    cmd = build_command(config, attacks_dir=Path("/tmp/attacks"))
    assert "algorithm/dataset_handler=wmdp" in cmd


def test_build_command_roleplay_emits_strategy_override() -> None:
    config = _baseline_config()
    config["experiment_name"] = "RoleplayStrategy"
    cmd = build_command(config, attacks_dir=Path("/tmp/attacks"))
    assert "algorithm/strategy=roleplay" in cmd


def test_build_command_data_strategy_emits_finetune_experiment() -> None:
    config = _data_config("meta-llama/Llama-2-7b-chat-hf", 50, 1)
    cmd = build_command(config, attacks_dir=Path("/tmp/attacks"))
    assert "experiment=finetune-with-strategy" in cmd
    assert "algorithm/strategy=data" in cmd
    assert "algorithm.strategy.r=16" in cmd
    assert "algorithm.strategy.lora_alpha=16" in cmd
    assert "algorithm.dataset_handler.max_examples=50" in cmd
    assert any(c.startswith("algorithm.dataset_handler.train_data_path=") for c in cmd)


def test_build_command_data_strategy_max_examples_none_emits_null() -> None:
    config = _data_config("meta-llama/Llama-2-7b-chat-hf", None, 1)
    cmd = build_command(config, attacks_dir=Path("/tmp/attacks"))
    assert "algorithm.dataset_handler.max_examples=null" in cmd


def test_build_command_precomputed_gcg(tmp_path: Path) -> None:
    attacks_dir = tmp_path / "attacks"
    attacks_dir.mkdir()
    canonical = attacks_dir / "gcg-meta-llama_Llama-2-7b-chat-hf.jsonl"
    canonical.write_text("")
    config = _gcg_config("meta-llama/Llama-2-7b-chat-hf")
    cmd = build_command(config, attacks_dir=attacks_dir)
    assert "experiment=prompt-attack" in cmd
    assert "algorithm/strategy=precomputed_gcg" in cmd
    assert f"algorithm.strategy.suffix_file={canonical}" in cmd


def test_build_command_trust_remote_code_for_special_models(tmp_path: Path) -> None:
    config = _baseline_config(
        model="/network/scratch/b/brownet/information-safety/models/safety-pair-safe"
    )
    cmd = build_command(config, attacks_dir=tmp_path)
    assert "algorithm.model.trust_remote_code=true" in cmd

    config2 = _baseline_config(model="allenai/Olmo-3-7B-Instruct")
    cmd2 = build_command(config2, attacks_dir=tmp_path)
    assert "algorithm.model.trust_remote_code=true" in cmd2


def test_deterministic_id_stable_for_same_config() -> None:
    config = _baseline_config()
    assert deterministic_id(config) == deterministic_id(dict(config))


def test_deterministic_id_differs_for_different_configs() -> None:
    a = _baseline_config()
    b = _baseline_config(model="meta-llama/Meta-Llama-3-8B-Instruct")
    assert deterministic_id(a) != deterministic_id(b)


def test_write_pending_jobs_emits_one_file_per_config(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    configs = [_baseline_config(), _baseline_config(model="meta-llama/Meta-Llama-3-8B-Instruct")]
    written = write_pending_jobs(
        configs, queue_root=queue_root, attacks_dir=tmp_path / "attacks"
    )
    assert len(written) == 2
    pending_files = sorted((queue_root / "pending").iterdir())
    assert len(pending_files) == 2
    for pf in pending_files:
        payload = json.loads(pf.read_text())
        assert "id" in payload
        assert "command" in payload
        assert "config" in payload
        assert isinstance(payload["command"], list)
        assert all(isinstance(x, str) for x in payload["command"])
        assert payload["command"][:2] == ["python", "information_safety/main.py"]


def test_write_pending_jobs_is_idempotent(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    configs = [_baseline_config()]
    write_pending_jobs(configs, queue_root=queue_root, attacks_dir=tmp_path / "attacks")
    first = sorted((queue_root / "pending").iterdir())
    write_pending_jobs(configs, queue_root=queue_root, attacks_dir=tmp_path / "attacks")
    second = sorted((queue_root / "pending").iterdir())
    assert [p.name for p in first] == [p.name for p in second]


def test_write_pending_jobs_skips_precomputed_when_suffix_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    queue_root = tmp_path / "queue"
    attacks_dir = tmp_path / "attacks"
    attacks_dir.mkdir()
    configs = [_gcg_config("meta-llama/Llama-2-7b-chat-hf")]
    written = write_pending_jobs(configs, queue_root=queue_root, attacks_dir=attacks_dir)
    assert written == []
    captured = capsys.readouterr()
    assert "skip" in captured.out.lower() or "warning" in captured.out.lower()


def test_main_writes_pending_files_for_synthetic_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "final-results.jsonl"
    results.write_text("")
    queue_root = tmp_path / "queue"
    attacks_dir = tmp_path / "attacks"
    attacks_dir.mkdir()
    main([
        "--results-file",
        str(results),
        "--queue-root",
        str(queue_root),
        "--attacks-dir",
        str(attacks_dir),
    ])
    pending = list((queue_root / "pending").iterdir())
    assert len(pending) > 0
