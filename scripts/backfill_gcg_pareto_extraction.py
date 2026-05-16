"""Backfill existing AdversariaLLM run.json directories with Pareto-step extraction.

Walks a save-dir root containing one subdirectory per ``<base>`` (AdversariaLLM Hydra
layout: ``<base>/<YYYY-MM-DD>/<HH-MM-SS>/<i>/run.json``). For each base, re-extracts the
attack rows via the canonical extractor (GCG fans out to one row per Pareto-optimal step;
AutoDAN/PAIR remain single-row with ``pareto_step_idx=0``) and writes the result to the
canonical per-shard JSONL at ``<attacks-dir>/<base>.jsonl``, overwriting any pre-Pareto
single-row file that ``scripts/run_one_attack.sh`` originally produced. The downstream
``scripts/rebuild_merged_jsonl.py`` consumes these per-shard files.

Does NOT touch ``final-results.jsonl`` or the ``generations/`` tree -- those backfill steps
are out of scope for this script and are owned by the eval pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from information_safety.attack_bits import compute_attack_bits
from scripts.extract_adversariallm_attacks import (
    _dispatch_attack_name,
    _extract_rows_from_run,
)


def _find_run_files(base: Path) -> list[Path]:
    return sorted(base.rglob("run.json"))


def _extract_rows_for_base(run_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    for run_file in run_files:
        with open(run_file) as f:
            run_json = json.load(f)
        attack_params = run_json["config"]["attack_params"]
        attack_name = _dispatch_attack_name(attack_params)
        if attack_name == "pair":
            attack_bits = compute_attack_bits(
                attack_name,
                attack_params,
                attacked_model_name=run_json["config"]["model_params"]["id"],
            )
        else:
            attack_bits = compute_attack_bits(attack_name, attack_params)
        new_rows = _extract_rows_from_run(
            run_json, str(run_file), attack_name, attack_bits
        )
        if not new_rows:
            continue
        behavior = new_rows[0]["behavior"]
        if behavior in seen:
            raise ValueError(
                f"duplicate behavior across runs: {run_file} and {seen[behavior]}"
            )
        seen[behavior] = run_file
        rows.extend(new_rows)
    return rows


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


def backfill_root(*, adv_save_dir_root: Path, attacks_dir: Path, dry_run: bool) -> int:
    """Re-extract Pareto rows for every base directory under ``adv_save_dir_root``.

    Writes each base's rows to ``attacks_dir/<base>.jsonl``, overwriting any pre-existing
    single-row file. Returns the number of base directories whose JSONL was rewritten (or
    would have been, in dry-run mode).
    """
    touched = 0
    for base in sorted(p for p in adv_save_dir_root.iterdir() if p.is_dir()):
        run_files = _find_run_files(base)
        if not run_files:
            continue
        rows = _extract_rows_for_base(run_files)
        if not dry_run:
            _atomic_write_jsonl(attacks_dir / f"{base.name}.jsonl", rows)
        touched += 1
    return touched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adv-save-dir-root", required=True, type=Path)
    parser.add_argument("--attacks-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    n = backfill_root(
        adv_save_dir_root=args.adv_save_dir_root,
        attacks_dir=args.attacks_dir,
        dry_run=args.dry_run,
    )
    suffix = " (dry run)" if args.dry_run else ""
    print(f"Backfilled {n} base directories into {args.attacks_dir}{suffix}")


if __name__ == "__main__":
    main()
