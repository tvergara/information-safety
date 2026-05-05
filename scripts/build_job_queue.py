"""Producer for the cluster job pool.

Reads the canonical ``check_results.py`` config grid and the centralized
``final-results.jsonl``, then writes one job-spec file per missing config
into ``<queue_root>/pending/``. The companion ``job_pool_worker.py``
consumes these files inside a fat SLURM allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from scripts.check_results import (
    EVILMATH_CONFIGS,
    HARMBENCH_CONFIGS,
    STRONGREJECT_CONFIGS,
    WMDP_CONFIGS,
)

DEFAULT_RESULTS_FILE = Path(
    "/network/scratch/b/brownet/information-safety/results/final-results.jsonl"
)
DEFAULT_ATTACKS_DIR = Path("/network/scratch/b/brownet/information-safety/attacks")
DEFAULT_TRAIN_DATA = (
    "/network/scratch/b/brownet/information-safety/data/advbench-completions-smollm3.jsonl"
)

PRECOMPUTED_TO_ATTACK = {
    "PrecomputedGCGStrategy": "gcg",
    "PrecomputedAutoDANStrategy": "autodan",
    "PrecomputedPAIRStrategy": "pair",
}

PRECOMPUTED_TO_STRATEGY_OVERRIDE = {
    "PrecomputedGCGStrategy": "precomputed_gcg",
    "PrecomputedAutoDANStrategy": "precomputed_autodan",
    "PrecomputedPAIRStrategy": "precomputed_pair",
}


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "_")


def _short_model(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1]


def _last_epoch_for(config: dict[str, Any]) -> int:
    if config["experiment_name"] == "DataStrategy":
        return config["max_epochs"] - 1
    return 0


def _matches(row: dict[str, Any], config: dict[str, Any]) -> bool:
    return all(row.get(k) == v for k, v in config.items())


def is_present(config: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    """Return True if ``config`` is already represented in ``rows``.

    For ``DataStrategy``, presence is determined by the *last* epoch row
    (``epoch = max_epochs - 1``). For everything else, presence is by
    exact match on the config keys.
    """
    last_epoch = _last_epoch_for(config)
    probe = {**config, "epoch": last_epoch}
    return any(_matches(row, probe) for row in rows)


def iter_missing_configs(
    configs: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return configs not yet present, deduped per experiment family."""
    seen: set[tuple[str, str, str, str, str, str, str]] = set()
    missing: list[dict[str, Any]] = []
    for config in configs:
        if is_present(config, rows):
            continue
        is_data_strategy = config["experiment_name"] == "DataStrategy"
        max_epochs = config["max_epochs"] if is_data_strategy else None
        dataset_name = config["dataset_name"]
        has_corpus_fraction = (
            is_data_strategy and dataset_name in {"wmdp", "evilmath"}
        )
        corpus_fraction = config["corpus_fraction"] if has_corpus_fraction else None
        is_wmdp_data = is_data_strategy and dataset_name == "wmdp"
        corpus_subset = config["corpus_subset"] if is_wmdp_data else None
        key = (
            config["experiment_name"],
            dataset_name,
            config["model_name"],
            str(config["max_examples"]),
            str(max_epochs),
            str(corpus_fraction),
            str(corpus_subset),
        )
        if key in seen:
            continue
        seen.add(key)
        missing.append(config)
    return missing


def resolve_suffix_file(
    *, attack: str, model_name: str, attacks_dir: Path
) -> Path | None:
    """Resolve a precomputed-attack suffix file path.

    Tries the canonical name first, then falls back to a glob and picks
    the newest by mtime. Returns ``None`` if nothing matches.
    """
    slug = _model_slug(model_name)
    canonical = attacks_dir / f"{attack}-{slug}.jsonl"
    if canonical.exists():
        return canonical
    candidates = sorted(
        attacks_dir.glob(f"{attack}-{slug}-*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return None


def _hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_command(config: dict[str, Any], *, attacks_dir: Path) -> list[str]:
    """Map a config dict to a Hydra-override CLI command.

    Raises ``FileNotFoundError`` if a precomputed-attack suffix file
    cannot be resolved — callers that want to skip-with-warning should
    catch that and emit a warning.
    """
    experiment_name = config["experiment_name"]
    model_name = config["model_name"]
    dataset_name = config["dataset_name"]
    dataset_handler = dataset_name
    name = f"pool-{experiment_name}-{_short_model(model_name)}"
    if config["max_examples"] is not None:
        name = f"{name}-mex{config['max_examples']}"
    if experiment_name == "DataStrategy":
        name = f"{name}-ep{config['max_epochs']}"
        if dataset_name == "wmdp":
            name = f"{name}-frac{config['corpus_fraction']}"
            if config["corpus_subset"] is not None:
                name = f"{name}-{config['corpus_subset']}"
        elif dataset_name == "evilmath":
            name = f"{name}-frac{config['corpus_fraction']}"

    cmd = [
        "python",
        "information_safety/main.py",
    ]

    if experiment_name == "DataStrategy":
        cmd += [
            "experiment=finetune-with-strategy",
            "algorithm/strategy=data",
            "algorithm.strategy.r=16",
            "algorithm.strategy.lora_alpha=16",
            f"algorithm.dataset_handler.train_data_path={DEFAULT_TRAIN_DATA}",
            f"algorithm.dataset_handler.max_examples={_hydra_value(config['max_examples'])}",
            f"trainer.max_epochs={config['max_epochs']}",
        ]
        if dataset_name == "wmdp":
            cmd += [
                f"algorithm.dataset_handler.corpus_fraction={config['corpus_fraction']}",
                f"algorithm.dataset_handler.corpus_subset={_hydra_value(config['corpus_subset'])}",
            ]
        elif dataset_name == "evilmath":
            cmd += [
                f"algorithm.dataset_handler.corpus_fraction={config['corpus_fraction']}",
            ]
    elif experiment_name == "BaselineStrategy":
        cmd += ["experiment=prompt-attack", "algorithm/strategy=baseline"]
    elif experiment_name == "RoleplayStrategy":
        cmd += ["experiment=prompt-attack", "algorithm/strategy=roleplay"]
    elif experiment_name in PRECOMPUTED_TO_ATTACK:
        attack = PRECOMPUTED_TO_ATTACK[experiment_name]
        strategy_override = PRECOMPUTED_TO_STRATEGY_OVERRIDE[experiment_name]
        suffix = resolve_suffix_file(
            attack=attack, model_name=model_name, attacks_dir=attacks_dir
        )
        if suffix is None:
            raise FileNotFoundError(
                f"No suffix file for attack={attack} model={model_name} "
                f"under {attacks_dir}"
            )
        cmd += [
            "experiment=prompt-attack",
            f"algorithm/strategy={strategy_override}",
            f"algorithm.strategy.suffix_file={suffix}",
        ]
    else:
        raise ValueError(f"Unknown experiment_name: {experiment_name}")

    cmd += [
        f"algorithm/dataset_handler={dataset_handler}",
        f"algorithm.model.pretrained_model_name_or_path={model_name}",
        "algorithm.model.trust_remote_code=true",
        "trainer.precision=bf16-mixed",
        f"name={name}",
    ]
    return cmd


def deterministic_id(config: dict[str, Any]) -> str:
    """Stable hash of a config dict — same input always produces same id."""
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def write_pending_jobs(
    configs: list[dict[str, Any]],
    *,
    queue_root: Path,
    attacks_dir: Path,
) -> list[Path]:
    """Write one ``<id>.json`` per config into ``<queue_root>/pending/``.

    Returns the list of files written. Re-running with the same configs is a no-op (deterministic
    ids → same filenames). Configs whose suffix file cannot be resolved are skipped with a printed
    warning.
    """
    pending_dir = queue_root / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("claimed", "done", "failed", "logs"):
        (queue_root / sub).mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for config in configs:
        try:
            command = build_command(config, attacks_dir=attacks_dir)
        except FileNotFoundError as exc:
            print(f"WARNING: skipping config — {exc}")
            continue

        job_id = deterministic_id(config)
        path = pending_dir / f"{job_id}.json"
        already_known = path.exists() or any(
            any((queue_root / sub).glob(f"{job_id}*.json"))
            for sub in ("claimed", "done", "failed")
        )
        if already_known:
            continue

        payload = {
            "id": job_id,
            "command": command,
            "config": config,
            "attempts": 0,
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, default=str))
        os.replace(tmp, path)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build cluster job-pool queue")
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--attacks-dir", type=Path, default=DEFAULT_ATTACKS_DIR)
    args = parser.parse_args(argv)

    with open(args.results_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    configs = HARMBENCH_CONFIGS + WMDP_CONFIGS + EVILMATH_CONFIGS + STRONGREJECT_CONFIGS
    missing = iter_missing_configs(configs, rows)
    written = write_pending_jobs(
        missing, queue_root=args.queue_root, attacks_dir=args.attacks_dir
    )
    print(f"Total configs: {len(configs)}")
    print(f"Missing (deduped): {len(missing)}")
    print(f"Pending jobs written: {len(written)}")
    print(f"Queue root: {args.queue_root}")


if __name__ == "__main__":
    main()
