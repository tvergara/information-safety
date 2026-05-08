from __future__ import annotations

import json
from pathlib import Path

import pytest

from information_safety.attack_bits import compute_attack_bits
from scripts.rebuild_merged_jsonl import rebuild_merged_jsonl


def _write_shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _row(behavior: str, *, attack_bits: int, attack_flops: int = 1) -> dict:
    return {
        "behavior": behavior,
        "adversarial_prompt": f"prompt-for-{behavior}",
        "attack_flops": attack_flops,
        "attack_bits": attack_bits,
    }


def test_dedups_overlapping_behaviors_keeps_last(tmp_path: Path) -> None:
    a = tmp_path / "shard_a.jsonl"
    b = tmp_path / "shard_b.jsonl"
    _write_shard(a, [_row("b0", attack_bits=100), _row("b1", attack_bits=100)])
    _write_shard(b, [_row("b1", attack_bits=200, attack_flops=999)])
    out = tmp_path / "out.jsonl"

    n = rebuild_merged_jsonl(attack="gcg", shard_paths=[a, b], output_path=out)

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert n == 2
    assert len(rows) == 2
    by_behavior = {r["behavior"]: r for r in rows}
    assert set(by_behavior) == {"b0", "b1"}
    assert by_behavior["b1"]["attack_flops"] == 999


def test_overwrites_attack_bits_with_canonical(tmp_path: Path) -> None:
    a = tmp_path / "shard_a.jsonl"
    _write_shard(a, [_row(f"b{i}", attack_bits=100 + i) for i in range(3)])
    out = tmp_path / "out.jsonl"

    rebuild_merged_jsonl(attack="gcg", shard_paths=[a], output_path=out)

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    bits = {r["attack_bits"] for r in rows}
    assert len(bits) == 1
    assert next(iter(bits)) > 0


def test_gcg_uses_compute_attack_bits(tmp_path: Path) -> None:
    a = tmp_path / "shard.jsonl"
    _write_shard(a, [_row("b0", attack_bits=999)])
    out = tmp_path / "out.jsonl"

    rebuild_merged_jsonl(attack="gcg", shard_paths=[a], output_path=out)

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["attack_bits"] == compute_attack_bits("gcg", {})


def test_pair_same_model_zero_aux(tmp_path: Path) -> None:
    a = tmp_path / "shard.jsonl"
    _write_shard(a, [_row("b0", attack_bits=999)])
    out = tmp_path / "out.jsonl"

    rebuild_merged_jsonl(
        attack="pair",
        shard_paths=[a],
        output_path=out,
        pair_attacked_model="Qwen/Qwen3-4B",
    )

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    config = {"attack_model": {"id": "Qwen/Qwen3-4B", "num_params": 0}}
    expected = compute_attack_bits(
        "pair", config, attacked_model_name="Qwen/Qwen3-4B"
    )
    assert rows[0]["attack_bits"] == expected


def test_writes_via_tmp_then_renames(tmp_path: Path) -> None:
    a = tmp_path / "shard.jsonl"
    _write_shard(a, [_row("b0", attack_bits=1)])
    out = tmp_path / "out.jsonl"

    rebuild_merged_jsonl(attack="gcg", shard_paths=[a], output_path=out)

    assert out.exists()
    assert not (tmp_path / "out.jsonl.tmp").exists()


def test_pair_requires_attacked_model(tmp_path: Path) -> None:
    a = tmp_path / "shard.jsonl"
    _write_shard(a, [_row("b0", attack_bits=999)])
    with pytest.raises(ValueError, match="pair_attacked_model"):
        rebuild_merged_jsonl(
            attack="pair", shard_paths=[a], output_path=tmp_path / "out.jsonl"
        )
