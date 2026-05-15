"""Rebuild a merged precomputed-attack JSONL from per-shard files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from information_safety.attack_bits import compute_attack_bits


def rebuild_merged_jsonl(
    *,
    attack: str,
    shard_paths: list[Path],
    output_path: Path,
    pair_attacked_model: str | None = None,
) -> int:
    if attack == "pair":
        if pair_attacked_model is None:
            raise ValueError(
                "pair_attacked_model is required for attack='pair' "
                "(aux=0 same-model assumption)"
            )
        config = {"attack_model": {"id": pair_attacked_model, "num_params": 0}}
        canonical_bits = compute_attack_bits(
            "pair", config, attacked_model_name=pair_attacked_model
        )
    else:
        canonical_bits = compute_attack_bits(attack, {})

    rows_by_key: dict[tuple[str, int], dict] = {}
    for path in shard_paths:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                row["attack_bits"] = canonical_bits
                pareto_step_idx = row.get("pareto_step_idx", 0)
                row["pareto_step_idx"] = pareto_step_idx
                rows_by_key[(row["behavior"], pareto_step_idx)] = row

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        for row in rows_by_key.values():
            f.write(json.dumps(row) + "\n")
    os.replace(tmp_path, output_path)
    return len(rows_by_key)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack", required=True, choices=["gcg", "autodan", "pair"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pair-attacked-model", default=None)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args(argv)

    n = rebuild_merged_jsonl(
        attack=args.attack,
        shard_paths=args.shards,
        output_path=args.output,
        pair_attacked_model=args.pair_attacked_model,
    )
    print(f"Wrote {n} rows to {args.output}")


if __name__ == "__main__":
    main()
