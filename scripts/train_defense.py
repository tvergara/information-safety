"""Single committed entry-point for defense training.

Usage:
  python scripts/train_defense.py \
    --method {sft,cb,tar} \
    --target {wmdp,evilmath,both} \
    --base-model <hf-name-or-path> \
    --output-dir $SCRATCH/defenses/<defense_id> \
    [--seed 0] [--dry-run] [--max-steps N]

Behavior:
1. Ensures the defense data exists (calls scripts/build_defense_data.py if
   missing and --build-data-if-missing is true).
2. For cb/tar, ensures the vendored repo is present at the pinned SHA.
3. Dispatches to the matching adapter in information_safety/defenses/.
4. Writes defense_meta.json to output_dir capturing the lineage so eval rows
   can be linked back to the defense run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from information_safety.defenses.circuit_breakers import (
    CBHParams,
    train_circuit_breakers,
)
from information_safety.defenses.refusal_sft import SFTHParams, train_refusal_sft
from information_safety.defenses.tar import TARHParams, train_tar

REPO_ROOT = Path(__file__).resolve().parent.parent

CB_REPO_DEFAULT = REPO_ROOT / "circuit-breakers"
TAR_REPO_DEFAULT = REPO_ROOT / "tar"



def compute_defense_id(method: str, target: str, base_model: str, seed: int) -> str:
    payload = f"{method}|{target}|{base_model}|{seed}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    short_model = base_model.rsplit("/", 1)[-1]
    return f"{method}-{target}-{short_model}-{digest}"


def _ensure_data(
    data_dir: Path,
    target: str,
    seed: int,
    build_if_missing: bool,
    evilmath_rewrite_path: Path | None,
) -> tuple[Path, Path]:
    refusals_path = data_dir / "refusals.jsonl"
    retain_path = data_dir / "retain.jsonl"

    if not refusals_path.exists() or not retain_path.exists():
        if not build_if_missing:
            raise FileNotFoundError(
                f"Defense data missing at {data_dir}; "
                "run scripts/build_defense_data.py or pass "
                "--build-data-if-missing"
            )
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_defense_data.py"),
            "--target", target,
            "--output-dir", str(data_dir),
            "--seed", str(seed),
        ]
        if evilmath_rewrite_path is not None:
            cmd.extend(["--evilmath-rewrite-path", str(evilmath_rewrite_path)])
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"build_defense_data.py failed with returncode {result.returncode}"
            )

    return refusals_path, retain_path


def _ensure_cb_repo(cb_repo: Path) -> str:
    if not cb_repo.exists():
        raise FileNotFoundError(
            f"circuit-breakers repo not found at {cb_repo}. Clone it before running."
        )
    return _git_sha(cb_repo)


def _ensure_tar_repo(tar_repo: Path) -> str:
    if not tar_repo.exists():
        raise FileNotFoundError(
            f"tar/ vendored repo not found at {tar_repo}. "
            "It should be in-tree as part of this repo."
        )
    return _git_sha(REPO_ROOT)


def _git_sha(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_defense_meta(
    output_dir: Path,
    defense_id: str,
    method: str,
    target: str,
    base_model: str,
    seed: int,
    refusals_path: Path,
    retain_path: Path,
    vendor_sha: str | None,
) -> None:
    meta = {
        "defense_id": defense_id,
        "method": method,
        "target": target,
        "base_model": base_model,
        "seed": seed,
        "refusals_path": str(refusals_path),
        "retain_path": str(retain_path),
        "vendor_sha": vendor_sha,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "defense_meta.json").write_text(json.dumps(meta, indent=2))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["sft", "cb", "tar"])
    parser.add_argument("--target", choices=["wmdp", "evilmath", "both"])
    parser.add_argument("--base-model")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--defense-data-dir", type=Path, default=None)
    parser.add_argument("--evilmath-rewrite-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--cb-repo", type=Path, default=CB_REPO_DEFAULT)
    parser.add_argument("--tar-repo", type=Path, default=TAR_REPO_DEFAULT)
    parser.add_argument(
        "--build-data-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--print-defense-id", action="store_true")
    return parser


def _require(args: argparse.Namespace, names: list[str]) -> None:
    missing = [n for n in names if getattr(args, n.replace("-", "_")) is None]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join('--' + n for n in missing)}")


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.print_defense_id:
        _require(args, ["method", "target", "base-model"])
        print(compute_defense_id(args.method, args.target, args.base_model, args.seed))
        return

    _require(args, ["method", "target", "base-model", "output-dir"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    defense_id = compute_defense_id(
        args.method, args.target, args.base_model, args.seed
    )

    if args.defense_data_dir is None:
        data_dir = (
            Path(os.environ["SCRATCH"]) / "information-safety" / "defense-data" / args.target
        )
    else:
        data_dir = args.defense_data_dir

    refusals_path, retain_path = _ensure_data(
        data_dir,
        args.target,
        args.seed,
        build_if_missing=args.build_data_if_missing,
        evilmath_rewrite_path=args.evilmath_rewrite_path,
    )

    vendor_sha: str | None = None
    if args.method == "sft":
        train_refusal_sft(
            base_model=args.base_model,
            refusals_path=refusals_path,
            retain_path=retain_path,
            output_dir=output_dir,
            hparams=SFTHParams(max_steps=args.max_steps)
            if args.max_steps is not None
            else SFTHParams(),
            dry_run=args.dry_run,
        )
    elif args.method == "cb":
        vendor_sha = _ensure_cb_repo(args.cb_repo)
        train_circuit_breakers(
            base_model=args.base_model,
            refusals_path=refusals_path,
            output_dir=output_dir,
            hparams=CBHParams(max_steps=args.max_steps)
            if args.max_steps is not None
            else CBHParams(),
            cb_repo=args.cb_repo,
            dry_run=args.dry_run,
        )
    elif args.method == "tar":
        vendor_sha = _ensure_tar_repo(args.tar_repo)
        train_tar(
            base_model=args.base_model,
            output_dir=output_dir,
            target=args.target,
            hparams=TARHParams(max_steps=args.max_steps)
            if args.max_steps is not None
            else TARHParams(),
            tar_repo=args.tar_repo,
            evilmath_data_path=args.evilmath_rewrite_path,
            dry_run=args.dry_run,
        )
    else:
        raise ValueError(f"Unknown method: {args.method!r}")

    _write_defense_meta(
        output_dir=output_dir,
        defense_id=defense_id,
        method=args.method,
        target=args.target,
        base_model=args.base_model,
        seed=args.seed,
        refusals_path=refusals_path,
        retain_path=retain_path,
        vendor_sha=vendor_sha,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
