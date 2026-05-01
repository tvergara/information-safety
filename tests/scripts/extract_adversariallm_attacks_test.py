"""Tests for the AdversariaLLM GCG attack-output converter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.extract_adversariallm_attacks import convert_run_dir

_GCG_CONFIG: dict[str, object] = {
    "num_steps": 10,
    "topk": 8,
    "optim_str_init": "x x x x",
}
_AUTODAN_CONFIG: dict[str, object] = {
    "num_steps": 5,
    "batch_size": 4,
    "mutation": 0.25,
}
_PAIR_CONFIG: dict[str, object] = {
    "num_steps": 3,
    "num_streams": 2,
    "attack_model": {
        "max_new_tokens": 8,
        "max_attempts": 4,
        "id": "lmsys/vicuna-13b",
        "num_params": 13_000_000_000,
    },
}
_ATTACKED_MODEL_ID = "meta-llama/Llama-3-8B"


def _write_run_json(
    path: Path,
    behavior: str,
    steps: list[dict],
    attack_params: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = attack_params if attack_params is not None else dict(_GCG_CONFIG)
    payload = {
        "config": {
            "attack_params": params,
            "model_params": {"id": _ATTACKED_MODEL_ID},
        },
        "runs": [
            {
                "original_prompt": [
                    {"role": "user", "content": behavior},
                    {"role": "assistant", "content": "target_A"},
                ],
                "steps": steps,
                "total_time": 0.0,
            }
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def test_convert_run_dir_emits_one_row_per_behavior(tmp_path: Path) -> None:
    run_dir = tmp_path / "adv_out"
    _write_run_json(
        run_dir / "0" / "run.json",
        behavior="beh_A",
        steps=[
            {
                "step": 0,
                "loss": 2.0,
                "flops": 100,
                "model_input": [
                    {"role": "user", "content": "beh_A suffix_init"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
            {
                "step": 1,
                "loss": 0.5,
                "flops": 200,
                "model_input": [
                    {"role": "user", "content": "beh_A suffix_opt"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
        ],
    )
    _write_run_json(
        run_dir / "1" / "run.json",
        behavior="beh_B",
        steps=[
            {
                "step": 0,
                "loss": 1.5,
                "flops": 300,
                "model_input": [
                    {"role": "user", "content": "beh_B suffix_B"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
        ],
    )

    out_file = tmp_path / "out.jsonl"
    convert_run_dir(str(run_dir), str(out_file))

    with open(out_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 2
    behaviors = {r["behavior"] for r in rows}
    assert behaviors == {"beh_A", "beh_B"}
    for row in rows:
        assert isinstance(row["adversarial_prompt"], str)
        assert "behavior" in row and "adversarial_prompt" in row
        assert "attack_bits" in row and isinstance(row["attack_bits"], int)
        assert row["attack_bits"] > 0


def test_convert_run_dir_selects_lowest_loss_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "adv_out"
    _write_run_json(
        run_dir / "0" / "run.json",
        behavior="beh_A",
        steps=[
            {
                "step": 0,
                "loss": 3.0,
                "flops": 10,
                "model_input": [
                    {"role": "user", "content": "beh_A worst"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
            {
                "step": 1,
                "loss": 0.1,
                "flops": 10,
                "model_input": [
                    {"role": "user", "content": "beh_A best"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
            {
                "step": 2,
                "loss": 0.5,
                "flops": 10,
                "model_input": [
                    {"role": "user", "content": "beh_A middle"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
        ],
    )

    out_file = tmp_path / "out.jsonl"
    convert_run_dir(str(run_dir), str(out_file))

    with open(out_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
    assert rows[0]["adversarial_prompt"] == "beh_A best"


def test_convert_run_dir_raises_on_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "adv_out"
    run_dir.mkdir()

    out_file = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="No run.json"):
        convert_run_dir(str(run_dir), str(out_file))


def test_convert_run_dir_raises_on_duplicate_behavior(tmp_path: Path) -> None:
    run_dir = tmp_path / "adv_out"
    for subdir in ("0", "1"):
        _write_run_json(
            run_dir / subdir / "run.json",
            behavior="beh_A",
            steps=[
                {
                    "step": 0,
                    "loss": 1.0,
                    "flops": 10,
                    "model_input": [
                        {"role": "user", "content": "beh_A suffix"},
                        {"role": "assistant", "content": ""},
                    ],
                    "model_completions": ["resp"],
                },
            ],
        )

    out_file = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="duplicate behavior"):
        convert_run_dir(str(run_dir), str(out_file))


def test_extract_handles_pair_shaped_runs_with_all_none_losses(tmp_path: Path) -> None:
    """PAIR scores via a judge and sets `loss=None` on every step.

    The extractor must pick the last step (most-refined) and still sum flops across all steps.
    """
    run_dir = tmp_path / "adv_out"
    _write_run_json(
        run_dir / "0" / "run.json",
        behavior="beh_A",
        steps=[
            {
                "step": 0,
                "loss": None,
                "flops": 1_000_000_000,
                "model_input": [
                    {"role": "user", "content": "beh_A step0"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
            {
                "step": 1,
                "loss": None,
                "flops": 1_000_000_000,
                "model_input": [
                    {"role": "user", "content": "beh_A step1"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
            {
                "step": 2,
                "loss": None,
                "flops": 1_000_000_000,
                "model_input": [
                    {"role": "user", "content": "beh_A step2"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
        ],
    )

    out_file = tmp_path / "out.jsonl"
    convert_run_dir(str(run_dir), str(out_file))

    with open(out_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
    assert rows[0]["adversarial_prompt"] == "beh_A step2"
    assert rows[0]["attack_flops"] == 3_000_000_000


def test_convert_run_dir_sums_flops_across_steps(tmp_path: Path) -> None:
    run_dir = tmp_path / "adv_out"
    _write_run_json(
        run_dir / "0" / "run.json",
        behavior="beh_A",
        steps=[
            {
                "step": 0,
                "loss": 2.0,
                "flops": 100,
                "model_input": [
                    {"role": "user", "content": "beh_A s0"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
            {
                "step": 1,
                "loss": 0.5,
                "flops": 250,
                "model_input": [
                    {"role": "user", "content": "beh_A s1"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
        ],
    )

    out_file = tmp_path / "out.jsonl"
    convert_run_dir(str(run_dir), str(out_file))

    with open(out_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
    assert rows[0]["attack_flops"] == 350
    assert "gcg_flops" not in rows[0]


def _make_steps(behavior: str) -> list[dict]:
    return [
        {
            "step": 0,
            "loss": 1.0,
            "flops": 10,
            "model_input": [
                {"role": "user", "content": f"{behavior} suffix"},
                {"role": "assistant", "content": ""},
            ],
            "model_completions": ["resp"],
        },
    ]


def test_convert_run_dir_writes_attack_bits_for_gcg(tmp_path: Path) -> None:
    from information_safety.attack_bits import compute_attack_bits

    run_dir = tmp_path / "adv_out"
    _write_run_json(
        run_dir / "0" / "run.json",
        behavior="beh_A",
        steps=_make_steps("beh_A"),
        attack_params=dict(_GCG_CONFIG),
    )

    out_file = tmp_path / "out.jsonl"
    convert_run_dir(str(run_dir), str(out_file))

    with open(out_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    expected_bits = compute_attack_bits("gcg", dict(_GCG_CONFIG))
    assert len(rows) == 1
    assert rows[0]["attack_bits"] == expected_bits


def test_convert_run_dir_writes_attack_bits_for_autodan(tmp_path: Path) -> None:
    from information_safety.attack_bits import compute_attack_bits

    run_dir = tmp_path / "adv_out"
    _write_run_json(
        run_dir / "0" / "run.json",
        behavior="beh_A",
        steps=_make_steps("beh_A"),
        attack_params=dict(_AUTODAN_CONFIG),
    )

    out_file = tmp_path / "out.jsonl"
    convert_run_dir(str(run_dir), str(out_file))

    with open(out_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    expected_bits = compute_attack_bits("autodan", dict(_AUTODAN_CONFIG))
    assert len(rows) == 1
    assert rows[0]["attack_bits"] == expected_bits


def test_convert_run_dir_writes_attack_bits_for_pair(tmp_path: Path) -> None:
    from information_safety.attack_bits import compute_attack_bits

    run_dir = tmp_path / "adv_out"
    _write_run_json(
        run_dir / "0" / "run.json",
        behavior="beh_A",
        steps=_make_steps("beh_A"),
        attack_params=dict(_PAIR_CONFIG),
    )

    out_file = tmp_path / "out.jsonl"
    convert_run_dir(str(run_dir), str(out_file))

    with open(out_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    expected_bits = compute_attack_bits(
        "pair", dict(_PAIR_CONFIG), attacked_model_name=_ATTACKED_MODEL_ID
    )
    assert len(rows) == 1
    assert rows[0]["attack_bits"] == expected_bits


def test_convert_run_dir_raises_on_unknown_attack_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "adv_out"
    _write_run_json(
        run_dir / "0" / "run.json",
        behavior="beh_A",
        steps=_make_steps("beh_A"),
        attack_params={"name": "mystery"},
    )

    out_file = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="Unknown attack"):
        convert_run_dir(str(run_dir), str(out_file))


def test_extract_includes_model_completion(tmp_path: Path) -> None:
    run_dir = tmp_path / "adv_out"
    _write_run_json(
        run_dir / "0" / "run.json",
        behavior="beh_A",
        steps=[
            {
                "step": 0,
                "loss": 0.5,
                "flops": 100,
                "model_input": [
                    {"role": "user", "content": "beh_A best"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["the_stored_response"],
            },
        ],
    )

    out_file = tmp_path / "out.jsonl"
    convert_run_dir(str(run_dir), str(out_file))

    with open(out_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
    assert rows[0]["model_completion"] == "the_stored_response"


def test_extract_omits_model_completion_when_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "adv_out"
    _write_run_json(
        run_dir / "0" / "run.json",
        behavior="beh_A",
        steps=[
            {
                "step": 0,
                "loss": 0.5,
                "flops": 100,
                "model_input": [
                    {"role": "user", "content": "beh_A best"},
                    {"role": "assistant", "content": ""},
                ],
            },
        ],
    )

    out_file = tmp_path / "out.jsonl"
    convert_run_dir(str(run_dir), str(out_file))

    with open(out_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
    assert "model_completion" not in rows[0]


def test_extract_autodan_uses_last_step_flops_not_sum(tmp_path: Path) -> None:
    """AutoDAN writes step.flops cumulatively.

    Extractor should take the last value.
    """
    run_dir = tmp_path / "adv_out"
    _write_run_json(
        run_dir / "0" / "run.json",
        behavior="beh_A",
        steps=[
            {
                "step": 0,
                "loss": 1.0,
                "flops": 10,
                "model_input": [
                    {"role": "user", "content": "beh_A s0"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
            {
                "step": 1,
                "loss": 0.5,
                "flops": 30,
                "model_input": [
                    {"role": "user", "content": "beh_A s1"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
            {
                "step": 2,
                "loss": 0.2,
                "flops": 60,
                "model_input": [
                    {"role": "user", "content": "beh_A s2"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
        ],
        attack_params=dict(_AUTODAN_CONFIG),
    )

    out_file = tmp_path / "out.jsonl"
    convert_run_dir(str(run_dir), str(out_file))

    with open(out_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
    assert rows[0]["attack_flops"] == 60


def test_extract_gcg_keeps_sum(tmp_path: Path) -> None:
    """GCG writes step.flops as deltas.

    Extractor should sum across all steps.
    """
    run_dir = tmp_path / "adv_out"
    _write_run_json(
        run_dir / "0" / "run.json",
        behavior="beh_A",
        steps=[
            {
                "step": 0,
                "loss": 1.0,
                "flops": 10,
                "model_input": [
                    {"role": "user", "content": "beh_A s0"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
            {
                "step": 1,
                "loss": 0.5,
                "flops": 20,
                "model_input": [
                    {"role": "user", "content": "beh_A s1"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
            {
                "step": 2,
                "loss": 0.2,
                "flops": 30,
                "model_input": [
                    {"role": "user", "content": "beh_A s2"},
                    {"role": "assistant", "content": ""},
                ],
                "model_completions": ["resp"],
            },
        ],
        attack_params=dict(_GCG_CONFIG),
    )

    out_file = tmp_path / "out.jsonl"
    convert_run_dir(str(run_dir), str(out_file))

    with open(out_file) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
    assert rows[0]["attack_flops"] == 60
