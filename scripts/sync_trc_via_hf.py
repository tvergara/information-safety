"""Sync trc results + generations to Mila through a HuggingFace dataset bucket.

trc's ``/work`` volume is only visible inside ``eai job`` containers, so the
results produced there cannot be rsynced off the login node directly. An eai
job uploads the per-source jsonls + generations tarballs to
``<repo>/trc-sync/``; Mila then ``snapshot_download``s them back, extracts the
tarballs, concatenates the jsonls, and runs ``merge_synced_results``.

Usage:
    python -m scripts.sync_trc_via_hf            # Mila side (full sync)
    python -m scripts.sync_trc_via_hf upload     # trc side (inside eai job)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from scripts._trc_common import (
    TRC_BASE_DIR,
    TRC_HF_HOME,
    TRC_INFORMATION_SAFETY_VENV,
    TRC_REPO_DIR,
    _wait_for_eai_job,
    build_eai_submit_remote,
    ensure_trc_synced,
    extract_eai_uuid,
    shell_quote,
)
from scripts.merge_synced_results import MergeSummary, merge_results

HF_RESULTS_REPO = "tvergara/information-safety-attack-artifacts"
HF_SYNC_PREFIX = "trc-sync"

TRC_UPLOAD_STAGING = "/work/sync-staging/trc-sync"

MILA_RESULTS_DIR = Path("/network/scratch/b/brownet/information-safety/results")
MILA_GENERATIONS_DIR = Path("/network/scratch/b/brownet/information-safety/generations")
MILA_DOWNLOAD_DEST = Path("/network/scratch/b/brownet/information-safety/trc-hf-sync")

_CONFLICTING_EVAL_JOB_PREFIXES = ("eval-sweep", "eval-shard-")


@dataclass
class SyncSource:
    name: str
    jsonl: Path
    generations: Path


def trc_sources() -> list[SyncSource]:
    base = Path(TRC_BASE_DIR)
    return [
        SyncSource(
            "adversariallm",
            base / "results" / "final-results.jsonl",
            base / "generations",
        ),
        SyncSource(
            "main",
            base / "main" / "final-results.jsonl",
            base / "main" / "generations",
        ),
    ]


def _tar_dir(directory: Path, tarball: Path) -> None:
    result = subprocess.run(
        ["tar", "czf", str(tarball), "-C", str(directory.parent), directory.name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"tar of {directory} failed ({result.returncode}): {result.stderr}"
        )


def _extract_tar(tarball: Path, dest: Path) -> None:
    result = subprocess.run(
        ["tar", "xzf", str(tarball), "-C", str(dest), "--strip-components=1"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"extracting {tarball} failed: {result.stderr}")


def stage_for_upload(sources: list[SyncSource], staging: Path) -> list[Path]:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    staged: list[Path] = []
    for src in sources:
        if src.jsonl.exists():
            dst = staging / f"{src.name}-final-results.jsonl"
            shutil.copyfile(src.jsonl, dst)
            staged.append(dst)
        if src.generations.is_dir():
            tarball = staging / f"{src.name}-generations.tar.gz"
            _tar_dir(src.generations, tarball)
            staged.append(tarball)
    return staged


def upload_to_hf(staging: Path, *, repo_id: str, api: HfApi) -> None:
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(staging),
        path_in_repo=HF_SYNC_PREFIX,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Sync trc results + generations",
    )


def download_from_hf(*, repo_id: str, dest: Path) -> Path:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{HF_SYNC_PREFIX}/*"],
        local_dir=str(dest),
    )
    return dest / HF_SYNC_PREFIX


def unpack_download(
    trc_sync_dir: Path,
    *,
    results_dir: Path,
    generations_dir: Path,
) -> None:
    from_trc = generations_dir / "from-trc"
    from_trc.mkdir(parents=True, exist_ok=True)
    for tarball in sorted(trc_sync_dir.glob("*-generations.tar.gz")):
        _extract_tar(tarball, from_trc)
    synced = results_dir / "final-results.jsonl.trc"
    with open(synced, "w") as out:
        for jsonl in sorted(trc_sync_dir.glob("*-final-results.jsonl")):
            with open(jsonl) as f:
                shutil.copyfileobj(f, out)


def eval_sweep_inflight() -> list[str]:
    result = subprocess.run(
        ["squeue", "--me", "--noheader", "-o", "%j"],
        capture_output=True,
        text=True,
    )
    return [
        job
        for job in result.stdout.splitlines()
        if job.strip().startswith(_CONFLICTING_EVAL_JOB_PREFIXES)
    ]


def build_upload_container_cmd(*, repo_id: str) -> str:
    return " && ".join([
        f"cd {TRC_REPO_DIR}",
        f"source {TRC_INFORMATION_SAFETY_VENV}/bin/activate",
        f"export HF_HOME={TRC_HF_HOME}",
        f"python -m scripts.sync_trc_via_hf upload --repo-id {shell_quote(repo_id)}",
    ])


def submit_upload_job(*, repo_id: str) -> str:
    remote = build_eai_submit_remote(
        name=f"is_sync_trc_hf_{int(time.time())}",
        container_cmd=build_upload_container_cmd(repo_id=repo_id),
        cpu=4,
        mem=16,
        max_run_time=1800,
        preemptable=False,
    )
    result = subprocess.run(
        ["ssh", "trc", remote], check=True, capture_output=True, text=True
    )
    return extract_eai_uuid(result.stdout)


def run_upload(*, repo_id: str, staging: Path) -> None:
    staged = stage_for_upload(trc_sources(), staging)
    print(f"[trc] staged {len(staged)} file(s) for upload: "
          f"{', '.join(p.name for p in staged)}")
    upload_to_hf(staging, repo_id=repo_id, api=HfApi())
    print(f"[trc] uploaded to {repo_id}:{HF_SYNC_PREFIX}/")


def run_sync(
    *,
    repo_id: str,
    results_dir: Path,
    generations_dir: Path,
    download_dest: Path,
) -> None:
    inflight = eval_sweep_inflight()
    if inflight:
        raise SystemExit(
            "Refusing to sync: conflicting eval job(s) in the queue: "
            f"{inflight}. Wait for them to finish (or scancel) first."
        )

    ensure_trc_synced()

    uuid = submit_upload_job(repo_id=repo_id)
    print(f"[trc] upload job uuid: {uuid}")
    _wait_for_eai_job(uuid)
    print("[trc] upload job SUCCEEDED.")

    trc_sync_dir = download_from_hf(repo_id=repo_id, dest=download_dest)
    unpack_download(
        trc_sync_dir,
        results_dir=results_dir,
        generations_dir=generations_dir,
    )
    print(f"[mila] results -> {results_dir / 'final-results.jsonl.trc'}")
    print(f"[mila] generations -> {generations_dir / 'from-trc'}")

    summary: MergeSummary = merge_results(
        cluster="trc",
        results_dir=results_dir,
        generations_dir=generations_dir,
    )
    print(
        f"[mila] merged: {summary.new_rows} new rows | "
        f"{summary.asr_preserved_skips} ASR-preserved skips | "
        f"{summary.new_symlinks} new symlinks | "
        f"{summary.skipped_malformed} malformed lines skipped"
    )
    print("\nMerge complete. When ready, backfill ASR with:")
    print("  sbatch slurm/eval-sweep-results.sh")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=["sync", "upload"], default="sync")
    parser.add_argument("--repo-id", default=HF_RESULTS_REPO)
    parser.add_argument("--staging", type=Path, default=Path(TRC_UPLOAD_STAGING))
    parser.add_argument("--results-dir", type=Path, default=MILA_RESULTS_DIR)
    parser.add_argument("--generations-dir", type=Path, default=MILA_GENERATIONS_DIR)
    parser.add_argument("--download-dest", type=Path, default=MILA_DOWNLOAD_DEST)
    args = parser.parse_args(argv)

    if args.mode == "upload":
        run_upload(repo_id=args.repo_id, staging=args.staging)
        return
    run_sync(
        repo_id=args.repo_id,
        results_dir=args.results_dir,
        generations_dir=args.generations_dir,
        download_dest=args.download_dest,
    )


if __name__ == "__main__":
    main()
