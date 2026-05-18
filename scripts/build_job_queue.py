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
    EVILMATH_DEFENSES,
    HARMBENCH_CONFIGS,
    STRONGREJECT_CONFIGS,
    WMDP_CONFIGS,
    WMDP_DEFENSES,
    _matches,
)

DEFENSE_IDS = set(WMDP_DEFENSES) | set(EVILMATH_DEFENSES)


def default_base_dir() -> Path:
    return Path(f"{os.environ['SCRATCH']}/information-safety")


def _defense_dir(base_dir: Path | None = None) -> Path:
    base = base_dir if base_dir is not None else default_base_dir()
    return base / "defenses"


def _resolve_model_path(
    model_name: str,
    base_dir: Path | None = None,
    defense_hf_namespace: str | None = None,
) -> str:
    if model_name in DEFENSE_IDS:
        if defense_hf_namespace is not None:
            return f"{defense_hf_namespace}/{model_name}"
        return str(_defense_dir(base_dir) / model_name)
    return model_name


def default_results_file(base_dir: Path | None = None) -> Path:
    base = base_dir if base_dir is not None else default_base_dir()
    return base / "results" / "final-results.jsonl"


def default_attacks_dir(base_dir: Path | None = None) -> Path:
    base = base_dir if base_dir is not None else default_base_dir()
    return base / "attacks"


def default_train_data(base_dir: Path | None = None) -> str:
    base = base_dir if base_dir is not None else default_base_dir()
    return str(base / "data" / "advbench-completions-smollm3.jsonl")

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

EVAL_TO_ATTACK_OPT_DATASET = {
    "advbench_harmbench": "harmbench",
    "strongreject": "strongreject",
}


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "_")


def _short_model(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1]


def _last_epoch_for(config: dict[str, Any]) -> int:
    if config["experiment_name"] == "DataStrategy":
        return config["max_epochs"] - 1
    return 0


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
        is_wmdp_data = is_data_strategy and dataset_name == "wmdp"
        corpus_fraction = config["corpus_fraction"] if is_wmdp_data else None
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
    *,
    attack: str,
    model_name: str,
    attacks_dir: Path,
    dataset: str,
    existence_check_dir: Path | None = None,
) -> Path | None:
    """Resolve a precomputed-attack suffix file path.

    Order of preference:
      1. Canonical ``{attack}-{slug}.jsonl`` (dataset-agnostic legacy form).
      2. Dataset-specific ``{attack}-{slug}-{attack_opt_dataset}-merged.jsonl``,
         where ``attack_opt_dataset`` is mapped from the eval ``dataset``
         via ``EVAL_TO_ATTACK_OPT_DATASET`` (identity if unmapped).
      3. Pre-dataset-suffix legacy ``{attack}-{slug}-merged.jsonl``, but only
         when ``attack_opt_dataset == "harmbench"`` — those legacy files were
         optimized against harmbench behaviors. Strongreject must never fall
         back to this path (it triggers silent 313-missing-behavior failures).

    ``existence_check_dir`` validates presence in a different directory than the
    one written into the returned path (trc: check Mila SCRATCH, emit /work path).
    """
    slug = _model_slug(model_name)
    check_dir = existence_check_dir if existence_check_dir is not None else attacks_dir
    canonical_name = f"{attack}-{slug}.jsonl"
    if (check_dir / canonical_name).exists():
        return attacks_dir / canonical_name
    attack_opt_dataset = EVAL_TO_ATTACK_OPT_DATASET.get(dataset, dataset)
    merged_name = f"{attack}-{slug}-{attack_opt_dataset}-merged.jsonl"
    if (check_dir / merged_name).exists():
        return attacks_dir / merged_name
    if attack_opt_dataset == "harmbench":
        legacy_name = f"{attack}-{slug}-merged.jsonl"
        if (check_dir / legacy_name).exists():
            return attacks_dir / legacy_name
    return None


def _hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_command(
    config: dict[str, Any],
    *,
    attacks_dir: Path,
    base_dir: Path | None = None,
    existence_check_attacks_dir: Path | None = None,
    defense_hf_namespace: str | None = None,
    defer_eval: bool = False,
) -> list[str]:
    """Map a config dict to a Hydra-override CLI command.

    ``base_dir`` is the cluster-specific results root ($SCRATCH on Mila/Tamia,
    /work/information-safety-results on trc); it scopes train_data and defense
    weight paths. ``existence_check_attacks_dir`` lets the Mila-side trc
    submitter validate suffix files locally while emitting /work paths.

    ``defense_hf_namespace`` routes defense weights through the HuggingFace Hub
    instead of a local path — when set, defense ``model_name``s in
    ``DEFENSE_IDS`` are emitted as ``f"{namespace}/{model_name}"``. Used by the
    trc submitter so defense weights are pulled from HF rather than from
    ``/work`` (which isn't pre-populated).

    Raises ``FileNotFoundError`` if a precomputed-attack suffix file cannot
    be resolved.
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

    cmd = [
        "python",
        "information_safety/main.py",
    ]

    if experiment_name == "DataStrategy":
        strategy_override = "data-deferred" if defer_eval else "data"
        cmd += [
            "experiment=finetune-with-strategy",
            f"algorithm/strategy={strategy_override}",
            "algorithm.strategy.r=16",
            "algorithm.strategy.lora_alpha=16",
            f"algorithm.dataset_handler.max_examples={_hydra_value(config['max_examples'])}",
            f"trainer.max_epochs={config['max_epochs']}",
        ]
        if model_name == "openai/gpt-oss-20b":
            cmd.append("algorithm.strategy.gradient_checkpointing=true")
        if dataset_name == "wmdp":
            cmd += [
                f"algorithm.dataset_handler.corpus_fraction={config['corpus_fraction']}",
                f"algorithm.dataset_handler.corpus_subset={_hydra_value(config['corpus_subset'])}",
            ]
        elif dataset_name in {"advbench_harmbench", "strongreject"}:
            cmd += [
                f"algorithm.dataset_handler.train_data_path={default_train_data(base_dir)}",
            ]
    elif experiment_name == "BaselineStrategy":
        cmd += ["experiment=prompt-attack", "algorithm/strategy=baseline"]
    elif experiment_name == "RoleplayStrategy":
        cmd += ["experiment=prompt-attack", "algorithm/strategy=roleplay"]
    elif experiment_name in PRECOMPUTED_TO_ATTACK:
        attack = PRECOMPUTED_TO_ATTACK[experiment_name]
        strategy_override = PRECOMPUTED_TO_STRATEGY_OVERRIDE[experiment_name]
        suffix = resolve_suffix_file(
            attack=attack,
            model_name=model_name,
            attacks_dir=attacks_dir,
            dataset=dataset_name,
            existence_check_dir=existence_check_attacks_dir,
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

    model_path = _resolve_model_path(model_name, base_dir, defense_hf_namespace)
    cmd += [
        f"algorithm/dataset_handler={dataset_handler}",
        f"algorithm.model.pretrained_model_name_or_path={model_path}",
        "algorithm.model.trust_remote_code=true",
        "trainer.precision=bf16-mixed",
        # Pool jobs are scored from saved generations, not from checkpoints.
        # 162 jobs * ~17 GiB last.ckpt would overflow the 2 TiB SCRATCH share at job ~55.
        "trainer.enable_checkpointing=false",
        "trainer/callbacks=no_checkpoints",
        f"name={name}",
    ]
    return cmd


def deterministic_id(config: dict[str, Any]) -> str:
    """Stable hash of a config dict — same input always produces same id."""
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def ensure_queue_dirs(queue_root: Path) -> None:
    for sub in ("pending", "claimed", "done", "failed", "logs"):
        (queue_root / sub).mkdir(parents=True, exist_ok=True)


def is_job_known(queue_root: Path, job_id: str) -> bool:
    if (queue_root / "pending" / f"{job_id}.json").exists():
        return True
    return any(
        any((queue_root / sub).glob(f"{job_id}*.json"))
        for sub in ("claimed", "done", "failed")
    )


def write_pending_payload(
    pending_dir: Path, job_id: str, payload: dict[str, Any]
) -> Path:
    path = pending_dir / f"{job_id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, default=str))
    os.replace(tmp, path)
    return path


def write_pending_jobs(
    configs: list[dict[str, Any]],
    *,
    queue_root: Path,
    attacks_dir: Path,
    base_dir: Path | None = None,
    defer_eval: bool = False,
) -> list[Path]:
    """Write one ``<id>.json`` per config into ``<queue_root>/pending/``.

    Returns the list of files written. Re-running with the same configs is a no-op (deterministic
    ids → same filenames). Configs whose suffix file cannot be resolved are skipped with a printed
    warning.

    When ``defer_eval`` is True, DataStrategy configs are emitted under
    ``algorithm/strategy=data-deferred`` and their spec ID hash includes the
    ``defer_eval: True`` field so IDs never collide with non-deferred specs.
    The training command is also prefixed with ``SPEC_ID=<id>`` so the strategy
    can stamp its adapter directory.
    """
    ensure_queue_dirs(queue_root)
    pending_dir = queue_root / "pending"

    written: list[Path] = []
    for config in configs:
        try:
            command = build_command(
                config, attacks_dir=attacks_dir, base_dir=base_dir,
                defer_eval=defer_eval,
            )
        except FileNotFoundError as exc:
            print(f"WARNING: skipping config — {exc}")
            continue

        id_payload = dict(config)
        if defer_eval:
            id_payload["defer_eval"] = True
        job_id = deterministic_id(id_payload)
        if is_job_known(queue_root, job_id):
            continue

        if defer_eval:
            command = ["env", f"SPEC_ID={job_id}", *command]

        payload = {
            "id": job_id,
            "command": command,
            "config": config,
            "attempts": 0,
        }
        if defer_eval:
            payload["defer_eval"] = True
        written.append(write_pending_payload(pending_dir, job_id, payload))
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build cluster job-pool queue")
    parser.add_argument("--results-file", type=Path, default=default_results_file())
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--attacks-dir", type=Path, default=default_attacks_dir())
    parser.add_argument(
        "--defer-eval", action="store_true",
        help="Emit DataStrategy specs that defer eval to scripts/run_eval_pool.py "
             "(uses algorithm/strategy=data-deferred). Spec IDs are namespaced "
             "so they never collide with non-deferred specs.",
    )
    args = parser.parse_args(argv)

    with open(args.results_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = [r for r in rows if r.get("invalidated_reason") is None]

    configs = HARMBENCH_CONFIGS + WMDP_CONFIGS + EVILMATH_CONFIGS + STRONGREJECT_CONFIGS
    missing = iter_missing_configs(configs, rows)
    written = write_pending_jobs(
        missing, queue_root=args.queue_root, attacks_dir=args.attacks_dir,
        defer_eval=args.defer_eval,
    )
    print(f"Total configs: {len(configs)}")
    print(f"Missing (deduped): {len(missing)}")
    print(f"Pending jobs written: {len(written)}")
    print(f"Queue root: {args.queue_root}")


if __name__ == "__main__":
    main()
