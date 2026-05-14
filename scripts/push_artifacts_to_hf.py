"""Push defense weights and adversarial artifacts to HuggingFace Hub.

Defenses go to public model repos under ``<namespace>/<defense-id>``.
Attack suffix files and train data go to a single private dataset repo.

Idempotent: ``create_repo(exist_ok=True)`` and ``upload_folder`` will
resume on retry.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

DEFENSE_PREFIXES = ("sft-", "cb-", "tar-")


def discover_defense_dirs(defenses_root: Path) -> list[Path]:
    return sorted(
        d
        for d in defenses_root.iterdir()
        if d.is_dir()
        and d.name.startswith(DEFENSE_PREFIXES)
        and any(d.iterdir())
    )


def push_defenses(
    *,
    defenses: list[Path],
    namespace: str,
    api: HfApi,
    dry_run: bool,
) -> None:
    for defense_dir in defenses:
        repo_id = f"{namespace}/{defense_dir.name}"
        if dry_run:
            print(f"[dry-run] would push {defense_dir} -> {repo_id} (public model)")
            continue
        print(f"creating {repo_id} ...")
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            exist_ok=True,
        )
        print(f"uploading {defense_dir} -> {repo_id} ...")
        api.upload_folder(
            folder_path=str(defense_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {defense_dir.name}",
        )
        print(f"done {repo_id}")


def push_attacks_data(
    *,
    attacks_dir: Path,
    data_dir: Path,
    repo_id: str,
    api: HfApi,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"[dry-run] would create private dataset {repo_id}")
        print(f"[dry-run] would upload {attacks_dir} -> attacks/")
        print(f"[dry-run] would upload {data_dir} -> data/")
        return
    print(f"creating private dataset {repo_id} ...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    print(f"uploading {attacks_dir} -> {repo_id}:attacks/ ...")
    api.upload_folder(
        folder_path=str(attacks_dir),
        path_in_repo="attacks",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload attack suffix files",
    )
    print(f"uploading {data_dir} -> {repo_id}:data/ ...")
    api.upload_folder(
        folder_path=str(data_dir),
        path_in_repo="data",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload train data",
    )
    print("done attacks+data")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defenses-root", type=Path, default=None)
    parser.add_argument("--attacks-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--namespace", default="tvergara")
    parser.add_argument(
        "--dataset-repo",
        default="tvergara/information-safety-attack-artifacts",
    )
    parser.add_argument("--skip-defenses", action="store_true")
    parser.add_argument("--skip-attacks-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    base = Path(f"{os.environ['SCRATCH']}/information-safety")
    defenses_root = args.defenses_root if args.defenses_root else base / "defenses"
    attacks_dir = args.attacks_dir if args.attacks_dir else base / "attacks"
    data_dir = args.data_dir if args.data_dir else base / "data"

    api = HfApi()
    if not args.skip_defenses:
        defenses = discover_defense_dirs(defenses_root)
        print(f"found {len(defenses)} defense dirs")
        push_defenses(
            defenses=defenses,
            namespace=args.namespace,
            api=api,
            dry_run=args.dry_run,
        )
    if not args.skip_attacks_data:
        push_attacks_data(
            attacks_dir=attacks_dir,
            data_dir=data_dir,
            repo_id=args.dataset_repo,
            api=api,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
