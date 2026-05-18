"""Audit Tamia pending queue against TRC + Nibi live state + final-results.jsonl.

Classifies each pending spec into one of:
  LIVE_TRC      — spec id is currently queued/running on TRC
  LIVE_NIBI     — spec id is currently queued/running on Nibi
  ALL_COMPLETE  — DS: every expected per-epoch row exists with ASR populated
                  attack-opt: shard output file already exists locally
  NEEDS_EVAL    — DS: rows exist but some lack ASR (fix is eval-sweep-results.sh)
  PARTIAL       — DS: some epochs done, some missing (re-run will fill gaps)
  MISSING       — DS: no rows at all / attack-opt: shard not produced — NEEDED

With ``--quarantine``, moves wasted-category specs into sibling
``quarantine-<reason>-YYYY-MM-DD/`` dirs (recoverable, not deleted).

Run BEFORE launching new pool consumers — otherwise workers waste compute on
specs that TRC/Nibi will also run, or that have already completed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import enum
import json
import os
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

TRC_LIVE_PAT = re.compile(r"\bis_(?:pool_)?([0-9a-f]{16})(?:[0-9_]|$)")
NIBI_LIVE_PAT = re.compile(r"nibi-pool-([0-9a-f]{16})")

TAMIA_ROBOT = "robot.tamia.ecpia.ca"
TAMIA_QUEUE_ROOT_BASE = "/scratch/t/tvergara/information-safety/job-pool"
TAMIA_ATTACKS_PREFIX = "/scratch/t/tvergara/information-safety/attacks/"
MILA_ATTACKS_DEFAULT = Path("/network/scratch/b/brownet/information-safety/attacks")

NIBI_ROBOT = "robot.nibi.alliancecan.ca"
TRC_HOST = "trc"


class Category(enum.Enum):
    LIVE_TRC = "LIVE_TRC"
    LIVE_NIBI = "LIVE_NIBI"
    ALL_COMPLETE = "ALL_COMPLETE"
    NEEDS_EVAL = "NEEDS_EVAL"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"

    @property
    def is_wasted(self) -> bool:
        return self in {Category.LIVE_TRC, Category.LIVE_NIBI,
                        Category.ALL_COMPLETE, Category.NEEDS_EVAL}


def parse_live_ids(text: str, pattern: re.Pattern[str]) -> set[str]:
    out: set[str] = set()
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            out.add(m.group(1))
    return out


def _row_matches(row: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Match a result row against an expected-row cfg.

    ``model_name`` uses substring matching (cfg holds the bare HF id, e.g.
    ``"Llama-3-8B"``; row holds the org-prefixed id, e.g.
    ``"meta-llama/Llama-3-8B-Instruct"``). All other keys must match exactly.
    """
    for k, v in cfg.items():
        if k == "model_name":
            row_model = row.get(k)
            if not isinstance(row_model, str) or v not in row_model:
                return False
        elif row.get(k) != v:
            return False
    return True


def expected_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "experiment_name": cfg["experiment_name"],
        "dataset_name": cfg["dataset_name"],
        "model_name": cfg["model_name"],
        "max_examples": cfg["max_examples"],
    }
    if cfg["experiment_name"] == "DataStrategy":
        out = []
        for epoch in range(cfg["max_epochs"]):
            row_cfg = dict(base, epoch=epoch)
            if cfg["dataset_name"] == "wmdp":
                row_cfg["corpus_fraction"] = cfg["corpus_fraction"]
                row_cfg["corpus_subset"] = cfg["corpus_subset"]
            out.append(row_cfg)
        return out
    return [dict(base, epoch=0)]


def classify_ds(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> Category:
    expected = expected_rows(cfg)
    hits_per_epoch = [[r for r in rows if _row_matches(r, ec)] for ec in expected]
    n_expected = len(expected)
    n_with_row = sum(1 for h in hits_per_epoch if h)
    if n_with_row == 0:
        return Category.MISSING
    if n_with_row < n_expected:
        return Category.PARTIAL
    n_with_asr = sum(
        1 for h in hits_per_epoch if all(r.get("asr") is not None for r in h)
    )
    return Category.ALL_COMPLETE if n_with_asr == n_expected else Category.NEEDS_EVAL


def attack_opt_output_path(spec: dict[str, Any], *, mila_attacks_dir: Path) -> Path | None:
    cmd = spec["command"]
    if "--output-jsonl" not in cmd:
        return None
    out = cmd[cmd.index("--output-jsonl") + 1]
    if out.startswith(TAMIA_ATTACKS_PREFIX):
        return mila_attacks_dir / out[len(TAMIA_ATTACKS_PREFIX):]
    return Path(out)


def classify_attack(spec: dict[str, Any], *, mila_attacks_dir: Path) -> Category:
    out = attack_opt_output_path(spec, mila_attacks_dir=mila_attacks_dir)
    if out is None:
        return Category.MISSING
    return Category.ALL_COMPLETE if out.exists() else Category.MISSING


def classify_spec(
    spec: dict[str, Any],
    *,
    results_rows: list[dict[str, Any]],
    mila_attacks_dir: Path,
    trc_live_ids: set[str],
    nibi_live_ids: set[str],
) -> Category:
    spec_id = spec["id"]
    if spec_id in trc_live_ids:
        return Category.LIVE_TRC
    if spec_id in nibi_live_ids:
        return Category.LIVE_NIBI
    if "config" in spec:
        return classify_ds(spec["config"], results_rows)
    return classify_attack(spec, mila_attacks_dir=mila_attacks_dir)


def load_results(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("invalidated_reason") is None:
            rows.append(r)
    return rows


def audit_pending(
    *,
    pending_dir: Path,
    results_file: Path,
    mila_attacks_dir: Path,
    trc_live_ids: set[str],
    nibi_live_ids: set[str],
) -> dict[Category, list[str]]:
    rows = load_results(results_file)
    out: dict[Category, list[str]] = defaultdict(list)
    for p in sorted(pending_dir.glob("*.json")):
        spec = json.loads(p.read_text())
        cat = classify_spec(
            spec,
            results_rows=rows,
            mila_attacks_dir=mila_attacks_dir,
            trc_live_ids=trc_live_ids,
            nibi_live_ids=nibi_live_ids,
        )
        out[cat].append(spec["id"])
    return out


def _rsync_pending(queue_root: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    remote = f"{TAMIA_ROBOT}:{TAMIA_QUEUE_ROOT_BASE}/{queue_root}/pending/"
    subprocess.run(
        ["rsync", "-a", "--delete", "--include=*.json", "--exclude=*",
         remote, f"{cache_dir}/"],
        check=True, capture_output=True, text=True,
    )
    return cache_dir


def _fetch_trc_live() -> set[str]:
    result = subprocess.run(
        ["ssh", "-n", TRC_HOST, "eai job ls --me --state queued,running --limit 1000 --no-header"],
        check=True, capture_output=True, text=True,
    )
    return parse_live_ids(result.stdout, TRC_LIVE_PAT)


def _fetch_nibi_live() -> set[str]:
    result = subprocess.run(
        ["ssh", "-n", NIBI_ROBOT, "squeue --me --noheader --format=%j"],
        check=True, capture_output=True, text=True,
    )
    return parse_live_ids(result.stdout, NIBI_LIVE_PAT)


def _ssh_robot(cmd: str) -> None:
    subprocess.run(
        ["ssh", "-n", TAMIA_ROBOT, cmd],
        check=True, capture_output=True, text=True,
    )


def _print_summary(categories: dict[Category, list[str]]) -> None:
    total = sum(len(v) for v in categories.values())
    print(f"{'Category':<14} {'Count':>6}  {'Pct':>6}")
    for cat in Category:
        n = len(categories.get(cat, []))
        pct = 100 * n / total if total else 0
        print(f"{cat.name:<14} {n:>6}  {pct:>5.1f}%")
    wasted = sum(len(categories.get(c, [])) for c in Category if c.is_wasted)
    needed = total - wasted
    print(f"{'TOTAL':<14} {total:>6}")
    print(f"WASTED (quarantine candidates): {wasted}")
    print(f"NEEDED (keep in pending):       {needed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", required=True,
                        help="Tamia queue root name (e.g., run-wmdp-rerun)")
    parser.add_argument("--results-file",
                        default=Path(os.environ["SCRATCH"]) / "information-safety/results/final-results.jsonl",
                        type=Path)
    parser.add_argument("--mila-attacks-dir", default=MILA_ATTACKS_DEFAULT, type=Path)
    parser.add_argument("--quarantine", action="store_true",
                        help="Move wasted specs into quarantine subdirs on Tamia")
    parser.add_argument("--cache-dir",
                        default=Path.home() / ".cache" / "information-safety" / "audit-queue-dedup",
                        type=Path)
    args = parser.parse_args()

    if not args.results_file.exists():
        print(f"results file not found: {args.results_file}", file=sys.stderr)
        return 2

    print(f"Fetching Tamia pending for queue {args.queue_root!r}...", file=sys.stderr)
    pending_dir = _rsync_pending(args.queue_root, args.cache_dir / args.queue_root)
    print("Fetching TRC + Nibi live state...", file=sys.stderr)
    trc_live = _fetch_trc_live()
    nibi_live = _fetch_nibi_live()
    print(f"TRC live: {len(trc_live)}  Nibi live: {len(nibi_live)}", file=sys.stderr)

    categories = audit_pending(
        pending_dir=pending_dir,
        results_file=args.results_file,
        mila_attacks_dir=args.mila_attacks_dir,
        trc_live_ids=trc_live,
        nibi_live_ids=nibi_live,
    )
    _print_summary(categories)

    if not args.quarantine:
        print("\n(dry run — pass --quarantine to actually move wasted specs)")
        return 0

    today = _dt.date.today().isoformat()
    moved = 0
    for cat in Category:
        if not cat.is_wasted:
            continue
        ids = categories.get(cat, [])
        if not ids:
            continue
        subdir = f"quarantine-{cat.name.lower().replace('_', '-')}-{today}"
        subdir_abs = f"{TAMIA_QUEUE_ROOT_BASE}/{args.queue_root}/{subdir}"
        _ssh_robot(f"mkdir -p {shlex.quote(subdir_abs)}")
        print(f"\nQuarantining {len(ids)} {cat.name} specs → {subdir}/", file=sys.stderr)
        for sid in ids:
            src = f"{TAMIA_QUEUE_ROOT_BASE}/{args.queue_root}/pending/{sid}.json"
            dst = f"{subdir_abs}/{sid}.json"
            _ssh_robot(f"mv {shlex.quote(src)} {shlex.quote(dst)}")
            moved += 1
    print(f"\nMoved {moved} specs out of pending.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
