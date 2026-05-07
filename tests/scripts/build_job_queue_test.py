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


@pytest.fixture(autouse=True)
def _scratch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRATCH", "/scratch/t/tvergara")


def _baseline_config(model: str = "meta-llama/Llama-2-7b-chat-hf") -> dict[str, object]:
    return {
        "experiment_name": "BaselineStrategy",
        "dataset_name": "advbench_harmbench",
        "model_name": model,
        "max_examples": None,
        "epoch": 0,
    }


def _data_config(
    model: str, max_ex: int | None, max_epochs: int, epoch: int
) -> dict[str, object]:
    return {
        "experiment_name": "DataStrategy",
        "dataset_name": "advbench_harmbench",
        "model_name": model,
        "max_examples": max_ex,
        "max_epochs": max_epochs,
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


def _wmdp_data_config(
    model: str,
    max_ex: int,
    max_epochs: int,
    epoch: int,
    corpus_fraction: float,
    corpus_subset: str | None,
) -> dict[str, object]:
    return {
        "experiment_name": "DataStrategy",
        "dataset_name": "wmdp",
        "model_name": model,
        "max_examples": max_ex,
        "max_epochs": max_epochs,
        "epoch": epoch,
        "corpus_fraction": corpus_fraction,
        "corpus_subset": corpus_subset,
    }


def _wmdp_roleplay_config(model: str) -> dict[str, object]:
    return {
        "experiment_name": "RoleplayStrategy",
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
    config_epoch0 = _data_config("meta-llama/Llama-2-7b-chat-hf", 16, 2, 0)
    config_epoch1 = _data_config("meta-llama/Llama-2-7b-chat-hf", 16, 2, 1)
    rows = [{**config_epoch1, "asr": 0.5}]
    assert is_present(config_epoch0, rows) is True
    assert is_present(config_epoch1, rows) is True


def test_is_present_data_strategy_partial_run_not_done() -> None:
    config_epoch0 = _data_config("meta-llama/Llama-2-7b-chat-hf", 16, 2, 0)
    config_epoch1 = _data_config("meta-llama/Llama-2-7b-chat-hf", 16, 2, 1)
    rows = [{**config_epoch0, "asr": 0.4}]
    assert is_present(config_epoch1, rows) is False


def test_is_present_data_strategy_uses_per_config_last_epoch() -> None:
    config = _data_config("meta-llama/Llama-2-7b-chat-hf", 16, 64, 0)
    last_epoch_row = {**config, "epoch": 63, "asr": 0.5}
    assert is_present(config, [last_epoch_row]) is True
    near_last_row = {**config, "epoch": 62, "asr": 0.5}
    assert is_present(config, [near_last_row]) is False


def test_iter_missing_dedups_data_strategy_epochs() -> None:
    configs = [
        _data_config("meta-llama/Llama-2-7b-chat-hf", 16, 64, 0),
        _data_config("meta-llama/Llama-2-7b-chat-hf", 16, 64, 1),
    ]
    missing = list(iter_missing_configs(configs, []))
    assert len(missing) == 1


def test_iter_missing_dedups_data_strategy_distinguishes_max_epochs() -> None:
    configs = [
        _data_config("meta-llama/Llama-2-7b-chat-hf", 16, 64, 0),
        _data_config("meta-llama/Llama-2-7b-chat-hf", 16, 32, 0),
    ]
    missing = list(iter_missing_configs(configs, []))
    assert len(missing) == 2


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
    assert "algorithm.model.trust_remote_code=true" in cmd
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
    config = _data_config("meta-llama/Llama-2-7b-chat-hf", 32, 32, 1)
    cmd = build_command(config, attacks_dir=Path("/tmp/attacks"))
    assert "experiment=finetune-with-strategy" in cmd
    assert "algorithm/strategy=data" in cmd
    assert "algorithm.strategy.r=16" in cmd
    assert "algorithm.strategy.lora_alpha=16" in cmd
    assert "algorithm.dataset_handler.max_examples=32" in cmd
    assert any(c.startswith("algorithm.dataset_handler.train_data_path=") for c in cmd)


def test_build_command_data_strategy_emits_max_epochs_override() -> None:
    config = _data_config("meta-llama/Llama-2-7b-chat-hf", 16, 64, 0)
    cmd = build_command(config, attacks_dir=Path("/tmp/attacks"))
    assert "trainer.max_epochs=64" in cmd


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


def test_build_command_trust_remote_code_always_true(tmp_path: Path) -> None:
    for model in (
        "meta-llama/Llama-2-7b-chat-hf",
        "/network/scratch/b/brownet/information-safety/models/safety-pair-safe",
        "openai/gpt-oss-20b",
    ):
        cmd = build_command(_baseline_config(model=model), attacks_dir=tmp_path)
        assert "algorithm.model.trust_remote_code=true" in cmd


def test_build_command_disables_checkpointing_for_pool_jobs(tmp_path: Path) -> None:
    """Pool worker shares 2 TiB SCRATCH across 4 GPUs; even at 1 ckpt/job * 162 jobs * 17 GiB the
    disk fills at job ~55.

    Pool jobs are scored from generations written separately, so checkpoints are useless. Disable
    them everywhere.
    """
    attacks_dir = tmp_path / "attacks"
    attacks_dir.mkdir()
    (attacks_dir / "gcg-meta-llama_Llama-2-7b-chat-hf.jsonl").write_text("")
    configs = [
        _baseline_config(),
        _data_config("meta-llama/Llama-2-7b-chat-hf", 32, 32, 1),
        _data_config("meta-llama/Llama-2-7b-chat-hf", 16, 64, 0),
        _gcg_config("meta-llama/Llama-2-7b-chat-hf"),
        _wmdp_config("meta-llama/Llama-2-7b-chat-hf"),
    ]
    roleplay = _baseline_config()
    roleplay["experiment_name"] = "RoleplayStrategy"
    configs.append(roleplay)
    for cfg in configs:
        cmd = build_command(cfg, attacks_dir=attacks_dir)
        assert "trainer.enable_checkpointing=false" in cmd, cfg
        assert "trainer/callbacks=no_checkpoints" in cmd, cfg


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


def test_pending_payload_has_attempts_zero(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    configs = [_baseline_config()]
    write_pending_jobs(
        configs, queue_root=queue_root, attacks_dir=tmp_path / "attacks"
    )
    pending_files = list((queue_root / "pending").iterdir())
    assert len(pending_files) == 1
    payload = json.loads(pending_files[0].read_text())
    assert payload["attempts"] == 0


def test_build_command_wmdp_data_strategy_emits_corpus_overrides() -> None:
    config = _wmdp_data_config(
        "Qwen/Qwen3-4B",
        128,
        8,
        0,
        corpus_fraction=0.5,
        corpus_subset="bio",
    )
    cmd = build_command(config, attacks_dir=Path("/tmp/attacks"))
    assert "experiment=finetune-with-strategy" in cmd
    assert "algorithm/strategy=data" in cmd
    assert "algorithm/dataset_handler=wmdp" in cmd
    assert "algorithm.dataset_handler.corpus_fraction=0.5" in cmd
    assert "algorithm.dataset_handler.corpus_subset=bio" in cmd
    assert not any(c.startswith("algorithm.dataset_handler.train_data_path=") for c in cmd)
    assert "algorithm.dataset_handler.max_examples=128" in cmd
    assert "trainer.max_epochs=8" in cmd


def test_build_command_wmdp_data_strategy_mix_zero_emits_null_subset() -> None:
    config = _wmdp_data_config(
        "Qwen/Qwen3-4B",
        128,
        8,
        0,
        corpus_fraction=0.0,
        corpus_subset=None,
    )
    cmd = build_command(config, attacks_dir=Path("/tmp/attacks"))
    assert "algorithm.dataset_handler.corpus_fraction=0.0" in cmd
    assert "algorithm.dataset_handler.corpus_subset=null" in cmd


def test_build_command_wmdp_roleplay_uses_wmdp_handler() -> None:
    config = _wmdp_roleplay_config("Qwen/Qwen3-4B")
    cmd = build_command(config, attacks_dir=Path("/tmp/attacks"))
    assert "experiment=prompt-attack" in cmd
    assert "algorithm/strategy=roleplay" in cmd
    assert "algorithm/dataset_handler=wmdp" in cmd


def test_iter_missing_dedups_wmdp_data_strategy_distinguishes_subset() -> None:
    bio_config = _wmdp_data_config(
        "Qwen/Qwen3-4B", 128, 8, 0, corpus_fraction=0.5, corpus_subset="bio"
    )
    cyber_config = _wmdp_data_config(
        "Qwen/Qwen3-4B", 128, 8, 0, corpus_fraction=0.5, corpus_subset="cyber"
    )
    missing = list(iter_missing_configs([bio_config, cyber_config], []))
    assert len(missing) == 2


def test_iter_missing_dedups_wmdp_data_strategy_distinguishes_fraction() -> None:
    mix_zero = _wmdp_data_config(
        "Qwen/Qwen3-4B", 128, 8, 0, corpus_fraction=0.0, corpus_subset=None
    )
    mix_half = _wmdp_data_config(
        "Qwen/Qwen3-4B", 128, 8, 0, corpus_fraction=0.5, corpus_subset="bio"
    )
    missing = list(iter_missing_configs([mix_zero, mix_half], []))
    assert len(missing) == 2


def test_iter_missing_collapses_wmdp_data_strategy_epoch_rows() -> None:
    epoch_rows = [
        _wmdp_data_config(
            "Qwen/Qwen3-4B", 128, 8, ep, corpus_fraction=0.5, corpus_subset="bio"
        )
        for ep in range(8)
    ]
    missing = list(iter_missing_configs(epoch_rows, []))
    assert len(missing) == 1


def _evilmath_data_config(
    model: str,
    max_ex: int,
    max_epochs: int,
    epoch: int,
) -> dict[str, object]:
    return {
        "experiment_name": "DataStrategy",
        "dataset_name": "evilmath",
        "model_name": model,
        "max_examples": max_ex,
        "max_epochs": max_epochs,
        "epoch": epoch,
    }


def test_build_command_evilmath_baseline_uses_evilmath_handler() -> None:
    config = {
        "experiment_name": "BaselineStrategy",
        "dataset_name": "evilmath",
        "model_name": "Qwen/Qwen3-4B",
        "max_examples": None,
        "epoch": 0,
    }
    cmd = build_command(config, attacks_dir=Path("/tmp/attacks"))
    assert "experiment=prompt-attack" in cmd
    assert "algorithm/strategy=baseline" in cmd
    assert "algorithm/dataset_handler=evilmath" in cmd


def test_build_command_evilmath_roleplay_uses_evilmath_handler() -> None:
    config = {
        "experiment_name": "RoleplayStrategy",
        "dataset_name": "evilmath",
        "model_name": "Qwen/Qwen3-4B",
        "max_examples": None,
        "epoch": 0,
    }
    cmd = build_command(config, attacks_dir=Path("/tmp/attacks"))
    assert "experiment=prompt-attack" in cmd
    assert "algorithm/strategy=roleplay" in cmd
    assert "algorithm/dataset_handler=evilmath" in cmd


def test_build_command_evilmath_data_strategy_no_corpus_or_train_data_overrides() -> None:
    config = _evilmath_data_config("Qwen/Qwen3-4B", 128, 8, 0)
    cmd = build_command(config, attacks_dir=Path("/tmp/attacks"))
    assert "experiment=finetune-with-strategy" in cmd
    assert "algorithm/strategy=data" in cmd
    assert "algorithm/dataset_handler=evilmath" in cmd
    assert "algorithm.dataset_handler.max_examples=128" in cmd
    assert "trainer.max_epochs=8" in cmd
    assert not any(c.startswith("algorithm.dataset_handler.train_data_path=") for c in cmd)
    assert not any(c.startswith("algorithm.dataset_handler.corpus_fraction") for c in cmd)
    assert not any(c.startswith("algorithm.dataset_handler.corpus_subset") for c in cmd)


def test_iter_missing_collapses_evilmath_data_strategy_epoch_rows() -> None:
    epoch_rows = [
        _evilmath_data_config("Qwen/Qwen3-4B", 128, 8, ep)
        for ep in range(8)
    ]
    missing = list(iter_missing_configs(epoch_rows, []))
    assert len(missing) == 1


def _strongreject_data_config(
    model: str,
    max_ex: int,
    max_epochs: int,
    epoch: int,
) -> dict[str, object]:
    return {
        "experiment_name": "DataStrategy",
        "dataset_name": "strongreject",
        "model_name": model,
        "max_examples": max_ex,
        "max_epochs": max_epochs,
        "epoch": epoch,
    }


def test_build_command_strongreject_baseline_uses_strongreject_handler() -> None:
    config = {
        "experiment_name": "BaselineStrategy",
        "dataset_name": "strongreject",
        "model_name": "Qwen/Qwen3-4B",
        "max_examples": None,
        "epoch": 0,
    }
    cmd = build_command(config, attacks_dir=Path("/tmp/attacks"))
    assert "experiment=prompt-attack" in cmd
    assert "algorithm/strategy=baseline" in cmd
    assert "algorithm/dataset_handler=strongreject" in cmd


def test_build_command_strongreject_data_strategy_no_corpus_overrides() -> None:
    config = _strongreject_data_config("Qwen/Qwen3-4B", 128, 8, 0)
    cmd = build_command(config, attacks_dir=Path("/tmp/attacks"))
    assert "experiment=finetune-with-strategy" in cmd
    assert "algorithm/strategy=data" in cmd
    assert "algorithm/dataset_handler=strongreject" in cmd
    assert "algorithm.dataset_handler.max_examples=128" in cmd
    assert "trainer.max_epochs=8" in cmd
    assert any(c.startswith("algorithm.dataset_handler.train_data_path=") for c in cmd)
    assert not any(c.startswith("algorithm.dataset_handler.corpus_fraction") for c in cmd)
    assert not any(c.startswith("algorithm.dataset_handler.corpus_subset") for c in cmd)


def test_iter_missing_collapses_strongreject_data_strategy_epoch_rows() -> None:
    epoch_rows = [
        _strongreject_data_config("Qwen/Qwen3-4B", 128, 8, ep) for ep in range(8)
    ]
    missing = list(iter_missing_configs(epoch_rows, []))
    assert len(missing) == 1


def test_default_paths_track_scratch_env_var() -> None:
    import scripts.build_job_queue as bjq

    assert str(bjq.default_results_file()).startswith("/scratch/t/tvergara/")
    assert str(bjq.default_attacks_dir()).startswith("/scratch/t/tvergara/")
    assert bjq.default_train_data().startswith("/scratch/t/tvergara/")


def test_build_command_train_data_path_uses_scratch_default() -> None:
    config = _data_config("meta-llama/Llama-2-7b-chat-hf", 32, 32, 0)
    cmd = build_command(config, attacks_dir=Path("/tmp/attacks"))
    train_arg = next(
        c for c in cmd if c.startswith("algorithm.dataset_handler.train_data_path=")
    )
    assert "/scratch/t/tvergara/" in train_arg


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
