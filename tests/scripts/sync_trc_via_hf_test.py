"""Tests for the HF-bucket trc results sync."""

from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import scripts.sync_trc_via_hf as sync


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _make_work_root(root: Path, *, with_main: bool = True) -> Path:
    _write(root / "results" / "final-results.jsonl", '{"eval_run_id": "adv-1"}\n')
    _write(root / "generations" / "adv-1" / "responses.jsonl", '{"response": "x"}\n')
    if with_main:
        _write(root / "main" / "final-results.jsonl", '{"eval_run_id": "pool-1"}\n')
        _write(
            root / "main" / "generations" / "pool-1" / "responses.jsonl",
            '{"response": "y"}\n',
        )
    return root


def test_stage_for_upload_collects_jsonls_and_tarballs(tmp_path: Path) -> None:
    work_root = _make_work_root(tmp_path / "work")
    staging = tmp_path / "staging"
    sources = [
        sync.SyncSource(
            "adversariallm",
            work_root / "results" / "final-results.jsonl",
            work_root / "generations",
        ),
        sync.SyncSource(
            "main",
            work_root / "main" / "final-results.jsonl",
            work_root / "main" / "generations",
        ),
    ]
    staged = sync.stage_for_upload(sources, staging)
    names = sorted(p.name for p in staged)
    assert names == [
        "adversariallm-final-results.jsonl",
        "adversariallm-generations.tar.gz",
        "main-final-results.jsonl",
        "main-generations.tar.gz",
    ]
    for tarball in staging.glob("*.tar.gz"):
        assert tarfile.is_tarfile(tarball)


def test_stage_for_upload_skips_missing_main(tmp_path: Path) -> None:
    work_root = _make_work_root(tmp_path / "work", with_main=False)
    staging = tmp_path / "staging"
    sources = [
        sync.SyncSource(
            "adversariallm",
            work_root / "results" / "final-results.jsonl",
            work_root / "generations",
        ),
        sync.SyncSource(
            "main",
            work_root / "main" / "final-results.jsonl",
            work_root / "main" / "generations",
        ),
    ]
    staged = sync.stage_for_upload(sources, staging)
    names = sorted(p.name for p in staged)
    assert names == [
        "adversariallm-final-results.jsonl",
        "adversariallm-generations.tar.gz",
    ]


def test_stage_for_upload_clears_stale_files(tmp_path: Path) -> None:
    work_root = _make_work_root(tmp_path / "work", with_main=False)
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "stale.tar.gz").write_text("old")
    sources = [
        sync.SyncSource(
            "adversariallm",
            work_root / "results" / "final-results.jsonl",
            work_root / "generations",
        ),
    ]
    sync.stage_for_upload(sources, staging)
    assert not (staging / "stale.tar.gz").exists()


def test_upload_to_hf_creates_private_dataset_and_uploads(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    api = MagicMock()
    sync.upload_to_hf(staging, repo_id="tvergara/x", api=api)
    create_call = api.create_repo.call_args
    assert create_call.kwargs["repo_id"] == "tvergara/x"
    assert create_call.kwargs["repo_type"] == "dataset"
    assert create_call.kwargs["private"] is True
    assert create_call.kwargs["exist_ok"] is True
    upload_call = api.upload_folder.call_args
    assert upload_call.kwargs["folder_path"] == str(staging)
    assert upload_call.kwargs["path_in_repo"] == sync.HF_SYNC_PREFIX
    assert upload_call.kwargs["repo_id"] == "tvergara/x"
    assert upload_call.kwargs["repo_type"] == "dataset"


def test_unpack_download_extracts_and_concats(tmp_path: Path) -> None:
    trc_sync = tmp_path / "trc-sync"
    trc_sync.mkdir()
    _write(trc_sync / "adversariallm-final-results.jsonl", '{"eval_run_id": "adv-1"}\n')
    _write(trc_sync / "main-final-results.jsonl", '{"eval_run_id": "pool-1"}\n')

    gen_src = tmp_path / "gensrc" / "generations"
    _write(gen_src / "adv-1" / "responses.jsonl", '{"response": "x"}\n')
    subprocess.run(
        ["tar", "czf", str(trc_sync / "adversariallm-generations.tar.gz"),
         "-C", str(gen_src.parent), "generations"],
        check=True,
    )
    gen_src2 = tmp_path / "gensrc2" / "generations"
    _write(gen_src2 / "pool-1" / "responses.jsonl", '{"response": "y"}\n')
    subprocess.run(
        ["tar", "czf", str(trc_sync / "main-generations.tar.gz"),
         "-C", str(gen_src2.parent), "generations"],
        check=True,
    )

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    generations_dir = tmp_path / "gens"
    generations_dir.mkdir()

    sync.unpack_download(
        trc_sync, results_dir=results_dir, generations_dir=generations_dir
    )

    assert (generations_dir / "from-trc" / "adv-1" / "responses.jsonl").exists()
    assert (generations_dir / "from-trc" / "pool-1" / "responses.jsonl").exists()
    synced = results_dir / "final-results.jsonl.trc"
    assert synced.exists()
    assert len([ln for ln in synced.read_text().splitlines() if ln.strip()]) == 2


def test_build_upload_container_cmd_targets_trc_repo() -> None:
    cmd = sync.build_upload_container_cmd(repo_id="tvergara/x")
    assert f"cd {sync.TRC_REPO_DIR}" in cmd
    assert sync.TRC_INFORMATION_SAFETY_VENV in cmd
    assert f"export HF_HOME={sync.TRC_HF_HOME}" in cmd
    assert "python -m scripts.sync_trc_via_hf upload" in cmd
    assert "tvergara/x" in cmd


def test_eval_sweep_inflight_detects_conflicting_jobs() -> None:
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="eval-sweep\neval-shard-3\nis_pool_x\n"
    )
    with patch.object(sync.subprocess, "run", return_value=completed):
        assert sync.eval_sweep_inflight() == ["eval-sweep", "eval-shard-3"]


def test_eval_sweep_inflight_empty_when_no_jobs() -> None:
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="is_pool_x\nis_evalpool_y\n"
    )
    with patch.object(sync.subprocess, "run", return_value=completed):
        assert sync.eval_sweep_inflight() == []


def test_run_sync_orchestrates_in_order(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    generations_dir = tmp_path / "gens"
    download_dest = tmp_path / "dl"
    results_dir.mkdir()
    generations_dir.mkdir()

    calls: list[str] = []
    trc_sync_dir = download_dest / sync.HF_SYNC_PREFIX

    with (
        patch.object(sync, "eval_sweep_inflight", return_value=[]),
        patch.object(sync, "ensure_trc_synced", side_effect=lambda: calls.append("synced")),
        patch.object(sync, "submit_upload_job", side_effect=lambda **_: calls.append("submit") or "uuid-1"),
        patch.object(sync, "_wait_for_eai_job", side_effect=lambda *_a, **_k: calls.append("wait")),
        patch.object(sync, "download_from_hf", side_effect=lambda **_: calls.append("download") or trc_sync_dir),
        patch.object(sync, "unpack_download", side_effect=lambda *_a, **_k: calls.append("unpack")),
        patch.object(sync, "merge_results", side_effect=lambda **_: calls.append("merge") or sync.MergeSummary(1, 0, 1, 0)),
    ):
        sync.run_sync(
            repo_id="tvergara/x",
            results_dir=results_dir,
            generations_dir=generations_dir,
            download_dest=download_dest,
        )
    assert calls == ["synced", "submit", "wait", "download", "unpack", "merge"]


def test_run_sync_refuses_when_eval_sweep_inflight(tmp_path: Path) -> None:
    with patch.object(sync, "eval_sweep_inflight", return_value=["eval-sweep"]):
        try:
            sync.run_sync(
                repo_id="tvergara/x",
                results_dir=tmp_path,
                generations_dir=tmp_path,
                download_dest=tmp_path,
            )
        except SystemExit as exc:
            assert "eval-sweep" in str(exc)
        else:
            raise AssertionError("expected SystemExit")
