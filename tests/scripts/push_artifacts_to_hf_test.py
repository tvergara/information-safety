"""Tests for the HF artifact push script."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from scripts.push_artifacts_to_hf import (
    discover_defense_dirs,
    push_attacks_data,
    push_defenses,
)


def _make_defense(root: Path, name: str, files: list[str]) -> Path:
    d = root / name
    d.mkdir()
    for f in files:
        (d / f).write_bytes(b"x" * 1024)
    return d


def test_discover_defense_dirs_skips_empty(tmp_path: Path) -> None:
    files = ["config.json", "model.safetensors"]
    _make_defense(tmp_path, "sft-wmdp-Llama-3.1-8B-Instruct-x", files)
    _make_defense(tmp_path, "cb-wmdp-Llama-3.1-8B-Instruct-y", files)
    (tmp_path / "base-Llama-3.1-8B-Instruct").mkdir()
    found = discover_defense_dirs(tmp_path)
    names = sorted(d.name for d in found)
    assert names == [
        "cb-wmdp-Llama-3.1-8B-Instruct-y",
        "sft-wmdp-Llama-3.1-8B-Instruct-x",
    ]


def test_discover_defense_dirs_only_defense_prefixes(tmp_path: Path) -> None:
    _make_defense(tmp_path, "sft-something", ["a.bin"])
    _make_defense(tmp_path, "cb-something", ["a.bin"])
    _make_defense(tmp_path, "tar-something", ["a.bin"])
    _make_defense(tmp_path, "random-folder", ["a.bin"])
    found = discover_defense_dirs(tmp_path)
    names = sorted(d.name for d in found)
    assert names == ["cb-something", "sft-something", "tar-something"]


def test_push_defenses_creates_public_repos_and_uploads(tmp_path: Path) -> None:
    d1 = _make_defense(tmp_path, "sft-wmdp-x", ["config.json"])
    d2 = _make_defense(tmp_path, "cb-wmdp-y", ["config.json"])
    api = MagicMock()
    push_defenses(defenses=[d1, d2], namespace="tvergara", api=api, dry_run=False)
    create_calls = api.create_repo.call_args_list
    upload_calls = api.upload_folder.call_args_list
    assert len(create_calls) == 2
    for call in create_calls:
        assert call.kwargs["private"] is False
        assert call.kwargs["repo_type"] == "model"
        assert call.kwargs["exist_ok"] is True
    repo_ids = {c.kwargs["repo_id"] for c in create_calls}
    assert repo_ids == {"tvergara/sft-wmdp-x", "tvergara/cb-wmdp-y"}
    assert len(upload_calls) == 2
    for call in upload_calls:
        assert call.kwargs["repo_type"] == "model"


def test_push_defenses_dry_run_skips_api_calls(tmp_path: Path) -> None:
    d = _make_defense(tmp_path, "sft-x", ["a.bin"])
    api = MagicMock()
    push_defenses(defenses=[d], namespace="tvergara", api=api, dry_run=True)
    api.create_repo.assert_not_called()
    api.upload_folder.assert_not_called()


def test_push_attacks_data_creates_private_dataset(tmp_path: Path) -> None:
    attacks = tmp_path / "attacks"
    data = tmp_path / "data"
    attacks.mkdir()
    data.mkdir()
    (attacks / "gcg-x.jsonl").write_text("{}")
    (data / "train.jsonl").write_text("{}")
    api = MagicMock()
    push_attacks_data(
        attacks_dir=attacks,
        data_dir=data,
        repo_id="tvergara/info-safety-private",
        api=api,
        dry_run=False,
    )
    api.create_repo.assert_called_once()
    create_call = api.create_repo.call_args
    assert create_call.kwargs["repo_id"] == "tvergara/info-safety-private"
    assert create_call.kwargs["repo_type"] == "dataset"
    assert create_call.kwargs["private"] is True
    assert create_call.kwargs["exist_ok"] is True
    upload_calls = api.upload_folder.call_args_list
    assert len(upload_calls) == 2
    paths_in_repo = {c.kwargs["path_in_repo"] for c in upload_calls}
    assert paths_in_repo == {"attacks", "data"}
    for c in upload_calls:
        assert c.kwargs["repo_type"] == "dataset"


def test_push_attacks_data_dry_run(tmp_path: Path) -> None:
    attacks = tmp_path / "attacks"
    data = tmp_path / "data"
    attacks.mkdir()
    data.mkdir()
    api = MagicMock()
    push_attacks_data(
        attacks_dir=attacks,
        data_dir=data,
        repo_id="tvergara/x",
        api=api,
        dry_run=True,
    )
    api.create_repo.assert_not_called()
    api.upload_folder.assert_not_called()
