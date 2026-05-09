"""Self-heal the attack pipeline between pool launches.

Three idempotent operations:

1. ``reextract_failed_attack_opt`` — for each attack-opt job in ``failed/``
   whose AdversariaLLM output dir still has ``run.json`` files, re-run the
   extraction step (CPU only) and move the spec to ``done/`` on success.

2. ``merge_completed_attacks`` — for each ``(attack, model, dataset)`` triple
   whose every shard has landed in ``done/``, assemble the per-shard JSONLs
   into the canonical ``{attack}-{slug}-{dataset}-merged.jsonl`` that the
   precomputed-strategy producer keys on. No-op if the merged file already
   exists or any shard is still pending/claimed/failed.

3. ``repend_recoverable_failures`` — for each precomputed-strategy job in
   ``failed/`` whose required merged JSONL now exists, atomically move it
   back to ``pending/`` with the rebuilt command.

Run by ``slurm/run-job-pool.sh`` at pool startup, after the stale-claim
reaper. Together they make the queue self-healing across pool launches.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from scripts.build_job_queue import (
    PRECOMPUTED_TO_ATTACK,
    build_command,
    default_attacks_dir,
)
from scripts.extract_adversariallm_attacks import convert_run_dir
from scripts.rebuild_merged_jsonl import rebuild_merged_jsonl

QUEUE_STATES = ("pending", "claimed", "done", "failed")


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "_")


def _iter_payloads(directory: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.json")):
        yield path, json.loads(path.read_text())


def attack_jobs_by_triple(
    queue_root: Path,
) -> dict[tuple[str, str, str], dict[str, set[int]]]:
    """Group attack-shard payloads by (attack, model, dataset) and queue state."""
    groups: dict[tuple[str, str, str], dict[str, set[int]]] = defaultdict(
        lambda: {state: set() for state in QUEUE_STATES}
    )
    for state in QUEUE_STATES:
        for path, payload in _iter_payloads(queue_root / state):
            spec = payload.get("spec")
            if not isinstance(spec, dict) or "attack" not in spec:
                continue
            key = (spec["attack"], spec["model"], spec["dataset"])
            groups[key][state].add(spec["shard"])
    return dict(groups)


def merge_completed_attacks(
    queue_root: Path, attacks_dir: Path
) -> dict[str, int]:
    """Assemble merged JSONLs for triples whose every shard is in done/.

    The queue is treated as authoritative: a triple's expected shard set is
    derived from the union of shards present across pending/claimed/done/failed,
    not from the producer's ``compute_shards``. Manually-deleted payloads will
    therefore yield a partial merge — by design, since the recover script has
    no other source of truth for what was originally emitted.
    """
    counts = {"merged": 0, "skipped_existing": 0, "skipped_missing_shard_files": 0}
    for (attack, model, dataset), states in attack_jobs_by_triple(queue_root).items():
        all_shards = set().union(*states.values())
        done_shards = states["done"]
        expected = set(range(max(all_shards) + 1)) if all_shards else set()
        if all_shards != expected or done_shards != expected:
            continue

        slug = _model_slug(model)
        merged_path = attacks_dir / f"{attack}-{slug}-{dataset}-merged.jsonl"
        if merged_path.exists():
            counts["skipped_existing"] += 1
            continue

        shard_paths = [
            attacks_dir / f"{attack}-{slug}-{dataset}-shard{i}.jsonl"
            for i in sorted(expected)
        ]
        missing = [p for p in shard_paths if not p.exists()]
        if missing:
            print(
                f"WARNING: {attack}-{slug}-{dataset}: "
                f"{len(missing)} shard JSONL(s) missing on disk despite done/ payload — skipping"
            )
            counts["skipped_missing_shard_files"] += 1
            continue

        rebuild_merged_jsonl(
            attack=attack,
            shard_paths=shard_paths,
            output_path=merged_path,
            pair_attacked_model=model if attack == "pair" else None,
        )
        counts["merged"] += 1
    return counts


def _atomic_write(dest: Path, payload: dict[str, Any]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=str))
    os.replace(tmp, dest)


def repend_recoverable_failures(
    queue_root: Path, attacks_dir: Path
) -> dict[str, int]:
    """Move failed precomputed-strategy jobs back to pending when recoverable.

    Recoverability is decided by ``build_command``: if it can resolve the
    suffix file (via ``resolve_suffix_file``), the job is re-pendable.
    Sharing this check with the producer avoids drift between the two paths.

    ``attempts`` is preserved (not reset) so a job that keeps failing after the
    merged JSONL is in place — e.g., malformed merge content — eventually hits
    the reaper's ``max_attempts`` ceiling instead of looping forever.
    """
    failed_dir = queue_root / "failed"
    pending_dir = queue_root / "pending"
    counts = {"repended": 0, "skipped_no_suffix": 0}

    for path, payload in _iter_payloads(failed_dir):
        config = payload.get("config")
        if not isinstance(config, dict):
            continue
        experiment = config.get("experiment_name")
        if experiment not in PRECOMPUTED_TO_ATTACK:
            continue

        try:
            rebuilt = build_command(config, attacks_dir=attacks_dir)
        except FileNotFoundError:
            counts["skipped_no_suffix"] += 1
            continue

        new_payload = {
            "id": payload["id"],
            "command": rebuilt,
            "config": config,
            "attempts": payload["attempts"],
        }
        _atomic_write(pending_dir / f"{payload['id']}.json", new_payload)
        os.remove(path)
        counts["repended"] += 1
    return counts


def _command_arg(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def reextract_failed_attack_opt(queue_root: Path) -> dict[str, int]:
    """Re-run the extraction step for failed attack-opt shards.

    For each ``failed/<id>.json`` whose payload has a ``spec``, parse the
    command for ``--adv-save-dir`` and ``--output-jsonl``. If the
    AdversariaLLM output directory still contains ``run.json`` files,
    invoke ``convert_run_dir`` and on success atomically move the spec to
    ``done/``.
    """
    failed_dir = queue_root / "failed"
    done_dir = queue_root / "done"
    counts = {"reextracted": 0, "skipped_no_run_dir": 0, "extraction_still_fails": 0}

    for path, payload in _iter_payloads(failed_dir):
        if "spec" not in payload:
            continue
        command = payload["command"]
        adv_save_dir = _command_arg(command, "--adv-save-dir")
        output_jsonl = _command_arg(command, "--output-jsonl")

        adv_save_path = Path(adv_save_dir)
        if not adv_save_path.exists() or not any(adv_save_path.glob("**/run.json")):
            counts["skipped_no_run_dir"] += 1
            continue

        try:
            convert_run_dir(adv_save_dir, output_jsonl)
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING: re-extraction still fails for {payload['id']}: "
                f"{type(exc).__name__}: {exc}"
            )
            counts["extraction_still_fails"] += 1
            continue

        os.rename(path, done_dir / f"{payload['id']}.json")
        counts["reextracted"] += 1
    return counts


def recover(queue_root: Path, attacks_dir: Path) -> dict[str, int]:
    """Run all self-heal phases.

    Returns combined stats.
    """
    reextract_stats = reextract_failed_attack_opt(queue_root)
    merge_stats = merge_completed_attacks(queue_root, attacks_dir)
    repend_stats = repend_recoverable_failures(queue_root, attacks_dir)
    return {**reextract_stats, **merge_stats, **repend_stats}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--attacks-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    attacks_dir = args.attacks_dir if args.attacks_dir is not None else default_attacks_dir()

    stats = recover(args.queue_root, attacks_dir)
    print(
        f"reextracted={stats['reextracted']} "
        f"skipped_no_run_dir={stats['skipped_no_run_dir']} "
        f"extraction_still_fails={stats['extraction_still_fails']} "
        f"merged={stats['merged']} "
        f"skipped_existing={stats['skipped_existing']} "
        f"skipped_missing_shard_files={stats['skipped_missing_shard_files']} "
        f"repended={stats['repended']} "
        f"skipped_no_suffix={stats['skipped_no_suffix']}"
    )


if __name__ == "__main__":
    main()
