"""Tests for the AdversariaLLM GCG attack-output converter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import extract_adversariallm_attacks
from scripts.extract_adversariallm_attacks import (
    _extract_gcg_pareto,
    convert_run_dir,
)


@pytest.fixture(autouse=True)
def _relax_min_attack_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extract_adversariallm_attacks, "MIN_ATTACK_STEPS", 1)

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
    behaviors = {r["behavior"] for r in rows}
    assert behaviors == {"beh_A", "beh_B"}
    for row in rows:
        assert isinstance(row["adversarial_prompt"], str)
        assert "behavior" in row and "adversarial_prompt" in row
        assert "attack_bits" in row and isinstance(row["attack_bits"], int)
        assert row["attack_bits"] > 0
        assert "pareto_step_idx" in row


def test_convert_run_dir_raises_on_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "adv_out"
    run_dir.mkdir()

    out_file = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="No run.json"):
        convert_run_dir(str(run_dir), str(out_file))


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
    canonical = next(r for r in rows if r["pareto_step_idx"] == 0)
    assert canonical["attack_flops"] == 350
    for row in rows:
        assert "gcg_flops" not in row


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


def test_extract_autodan_uses_step_flops_not_sum(tmp_path: Path) -> None:
    """AutoDAN writes step.flops cumulatively.

    Extractor should take each Pareto step's own `flops` field, not a sum.
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
    canonical = next(r for r in rows if r["pareto_step_idx"] == 0)
    assert canonical["attack_flops"] == 60


def test_extract_gcg_keeps_sum(tmp_path: Path) -> None:
    """GCG writes step.flops as deltas.

    Extractor should sum across all steps (cumulative through the final pareto step).
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
    canonical = next(r for r in rows if r["pareto_step_idx"] == 0)
    assert canonical["attack_flops"] == 60


def _gcg_step(
    step_idx: int, loss: float | None, flops: int, prompt_text: str
) -> dict:
    return {
        "step": step_idx,
        "loss": loss,
        "flops": flops,
        "model_input": [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": ""},
        ],
        "model_completions": [f"completion_for_{prompt_text}"],
    }


def _make_gcg_run_json(behavior: str, steps: list[dict]) -> dict:
    return {
        "config": {
            "attack_params": dict(_GCG_CONFIG),
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


class TestExtractGCGPareto:
    def test_gcg_emits_one_row_per_pareto_step(self) -> None:
        steps = [
            _gcg_step(0, 3.0, 10, "p_step_0"),
            _gcg_step(1, 2.5, 10, "p_step_1"),
            _gcg_step(2, 2.5, 10, "p_step_2"),
            _gcg_step(3, 1.8, 10, "p_step_3"),
            _gcg_step(4, 2.0, 10, "p_step_4"),
            _gcg_step(5, 1.5, 10, "p_step_5"),
        ]
        run_json = _make_gcg_run_json("beh_A", steps)

        rows = _extract_gcg_pareto(run_json, "fake.json")

        assert len(rows) == 4
        adversarial_prompts = [r["adversarial_prompt"] for r in rows]
        assert adversarial_prompts == [
            "p_step_5",
            "p_step_3",
            "p_step_1",
            "p_step_0",
        ]
        pareto_idxs = [r["pareto_step_idx"] for r in rows]
        assert pareto_idxs == [0, 1, 2, 3]
        attack_flops = [r["attack_flops"] for r in rows]
        assert attack_flops == [60, 40, 20, 10]
        for row in rows:
            assert row["behavior"] == "beh_A"

    def test_gcg_pareto_step_0_is_argmin_loss(self, tmp_path: Path) -> None:
        steps = [
            _gcg_step(0, 3.0, 100, "worst"),
            _gcg_step(1, 0.1, 100, "best"),
            _gcg_step(2, 0.5, 100, "middle"),
        ]
        run_json = _make_gcg_run_json("beh_A", steps)

        rows = _extract_gcg_pareto(run_json, "fake.json")

        zero = next(r for r in rows if r["pareto_step_idx"] == 0)
        assert zero["adversarial_prompt"] == "best"
        assert zero["attack_flops"] == 200
        assert zero["model_completion"] == "completion_for_best"

        run_dir = tmp_path / "adv_out"
        _write_run_json(
            run_dir / "0" / "run.json",
            behavior="beh_A",
            steps=steps,
        )
        out_file = tmp_path / "out.jsonl"
        convert_run_dir(str(run_dir), str(out_file))

        with open(out_file) as f:
            converted_rows = [json.loads(line) for line in f if line.strip()]
        zero_converted = next(
            r for r in converted_rows if r.get("pareto_step_idx", 0) == 0
        )
        assert zero_converted["adversarial_prompt"] == zero["adversarial_prompt"]
        assert zero_converted["attack_flops"] == zero["attack_flops"]
        assert zero_converted["model_completion"] == zero["model_completion"]


class TestAutodanTrajectory:
    def test_autodan_emits_one_row_per_pareto_step(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "adv_out"
        _write_run_json(
            run_dir / "0" / "run.json",
            behavior="beh_A",
            steps=[
                _gcg_step(0, 3.0, 10, "p_step_0"),
                _gcg_step(1, 2.5, 20, "p_step_1"),
                _gcg_step(2, 2.5, 30, "p_step_2"),
                _gcg_step(3, 1.8, 40, "p_step_3"),
                _gcg_step(4, 2.0, 50, "p_step_4"),
                _gcg_step(5, 1.5, 60, "p_step_5"),
            ],
            attack_params=dict(_AUTODAN_CONFIG),
        )
        out_file = tmp_path / "out.jsonl"
        convert_run_dir(str(run_dir), str(out_file))

        with open(out_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]

        assert len(rows) == 4
        adversarial_prompts = [r["adversarial_prompt"] for r in rows]
        assert adversarial_prompts == [
            "p_step_5",
            "p_step_3",
            "p_step_1",
            "p_step_0",
        ]
        pareto_idxs = [r["pareto_step_idx"] for r in rows]
        assert pareto_idxs == [0, 1, 2, 3]
        attack_flops = [r["attack_flops"] for r in rows]
        assert attack_flops == [60, 40, 20, 10]
        for row in rows:
            assert row["behavior"] == "beh_A"
            assert row["model_completion"] is not None

    def test_autodan_pareto_step_0_is_argmin_loss(self, tmp_path: Path) -> None:
        steps = [
            _gcg_step(0, 3.0, 100, "worst"),
            _gcg_step(1, 0.1, 200, "best"),
            _gcg_step(2, 0.5, 300, "middle"),
        ]
        run_dir = tmp_path / "adv_out"
        _write_run_json(
            run_dir / "0" / "run.json",
            behavior="beh_A",
            steps=steps,
            attack_params=dict(_AUTODAN_CONFIG),
        )
        out_file = tmp_path / "out.jsonl"
        convert_run_dir(str(run_dir), str(out_file))

        with open(out_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        zero = next(r for r in rows if r["pareto_step_idx"] == 0)
        assert zero["adversarial_prompt"] == "best"
        assert zero["attack_flops"] == 200
        assert zero["model_completion"] == "completion_for_best"

    def test_autodan_respects_min_attack_steps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(extract_adversariallm_attacks, "MIN_ATTACK_STEPS", 20)
        run_dir = tmp_path / "adv_out"
        _write_run_json(
            run_dir / "0" / "run.json",
            behavior="beh_A",
            steps=[_gcg_step(i, 1.0 - 0.01 * i, 10 + i, f"s{i}") for i in range(5)],
            attack_params=dict(_AUTODAN_CONFIG),
        )

        out_file = tmp_path / "out.jsonl"
        convert_run_dir(str(run_dir), str(out_file))

        with open(out_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert rows == []

    def test_autodan_all_rows_have_model_completion(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "adv_out"
        _write_run_json(
            run_dir / "0" / "run.json",
            behavior="beh_A",
            steps=[
                _gcg_step(0, 3.0, 10, "p0"),
                _gcg_step(1, 1.8, 30, "p1"),
                _gcg_step(2, 1.0, 60, "p2"),
            ],
            attack_params=dict(_AUTODAN_CONFIG),
        )
        out_file = tmp_path / "out.jsonl"
        convert_run_dir(str(run_dir), str(out_file))

        with open(out_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == 3
        for row in rows:
            assert row["model_completion"] is not None
            assert isinstance(row["model_completion"], str)


class TestPairTrajectory:
    def test_pair_emits_strided_rows_plus_final(self, tmp_path: Path) -> None:
        n = 25
        run_dir = tmp_path / "adv_out"
        _write_run_json(
            run_dir / "0" / "run.json",
            behavior="beh_A",
            steps=[_gcg_step(i, None, 1, f"p{i}") for i in range(n)],
            attack_params=dict(_PAIR_CONFIG),
        )
        out_file = tmp_path / "out.jsonl"
        convert_run_dir(str(run_dir), str(out_file))

        with open(out_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == 3
        adversarial_prompts = [r["adversarial_prompt"] for r in rows]
        assert adversarial_prompts == ["p24", "p14", "p4"]
        pareto_idxs = [r["pareto_step_idx"] for r in rows]
        assert pareto_idxs == [0, 1, 2]
        attack_flops = [r["attack_flops"] for r in rows]
        assert attack_flops == [25, 15, 5]
        for row in rows:
            assert row["behavior"] == "beh_A"
            assert row["model_completion"] is not None

    def test_pair_short_run_emits_single_final_row(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "adv_out"
        _write_run_json(
            run_dir / "0" / "run.json",
            behavior="beh_A",
            steps=[_gcg_step(i, None, 7, f"p{i}") for i in range(3)],
            attack_params=dict(_PAIR_CONFIG),
        )
        out_file = tmp_path / "out.jsonl"
        convert_run_dir(str(run_dir), str(out_file))

        with open(out_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == 1
        assert rows[0]["pareto_step_idx"] == 0
        assert rows[0]["adversarial_prompt"] == "p2"
        assert rows[0]["attack_flops"] == 21

    def test_pair_exempt_from_min_attack_steps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(extract_adversariallm_attacks, "MIN_ATTACK_STEPS", 100)
        run_dir = tmp_path / "adv_out"
        _write_run_json(
            run_dir / "0" / "run.json",
            behavior="beh_A",
            steps=[_gcg_step(i, None, 1, f"p{i}") for i in range(3)],
            attack_params=dict(_PAIR_CONFIG),
        )
        out_file = tmp_path / "out.jsonl"
        convert_run_dir(str(run_dir), str(out_file))

        with open(out_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == 1

    def test_pair_all_rows_have_model_completion(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "adv_out"
        _write_run_json(
            run_dir / "0" / "run.json",
            behavior="beh_A",
            steps=[_gcg_step(i, None, 1, f"p{i}") for i in range(25)],
            attack_params=dict(_PAIR_CONFIG),
        )
        out_file = tmp_path / "out.jsonl"
        convert_run_dir(str(run_dir), str(out_file))

        with open(out_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for row in rows:
            assert row["model_completion"] is not None
            assert isinstance(row["model_completion"], str)

    def test_pair_final_step_cumulative_flops(self, tmp_path: Path) -> None:
        """N steps with per-step flops sum to total cumulative flops at final step."""
        n = 12
        run_dir = tmp_path / "adv_out"
        _write_run_json(
            run_dir / "0" / "run.json",
            behavior="beh_A",
            steps=[_gcg_step(i, None, 5 + i, f"p{i}") for i in range(n)],
            attack_params=dict(_PAIR_CONFIG),
        )
        out_file = tmp_path / "out.jsonl"
        convert_run_dir(str(run_dir), str(out_file))

        with open(out_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        zero = next(r for r in rows if r["pareto_step_idx"] == 0)
        expected_total = sum(5 + i for i in range(n))
        assert zero["attack_flops"] == expected_total
        assert zero["model_completion"] == f"completion_for_p{n - 1}"


class TestMinAttackStepsFilter:
    def test_below_threshold_drops_behavior(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(extract_adversariallm_attacks, "MIN_ATTACK_STEPS", 5)
        run_dir = tmp_path / "adv_out"
        _write_run_json(
            run_dir / "0" / "run.json",
            behavior="beh_A",
            steps=[_gcg_step(i, 1.0 - 0.1 * i, 10, f"s{i}") for i in range(4)],
            attack_params=dict(_GCG_CONFIG),
        )

        out_file = tmp_path / "out.jsonl"
        convert_run_dir(str(run_dir), str(out_file))

        with open(out_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert rows == []

    def test_at_threshold_emits_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(extract_adversariallm_attacks, "MIN_ATTACK_STEPS", 5)
        run_dir = tmp_path / "adv_out"
        _write_run_json(
            run_dir / "0" / "run.json",
            behavior="beh_A",
            steps=[_gcg_step(i, 1.0 - 0.1 * i, 10, f"s{i}") for i in range(5)],
            attack_params=dict(_GCG_CONFIG),
        )

        out_file = tmp_path / "out.jsonl"
        convert_run_dir(str(run_dir), str(out_file))

        with open(out_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == 5

    def test_convert_run_dir_skips_filtered_behaviors_but_keeps_others(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(extract_adversariallm_attacks, "MIN_ATTACK_STEPS", 5)
        run_dir = tmp_path / "adv_out"
        _write_run_json(
            run_dir / "0" / "run.json",
            behavior="beh_short",
            steps=[_gcg_step(i, 1.0 - 0.1 * i, 10, f"x{i}") for i in range(3)],
            attack_params=dict(_GCG_CONFIG),
        )
        _write_run_json(
            run_dir / "1" / "run.json",
            behavior="beh_long",
            steps=[_gcg_step(i, 1.0 - 0.05 * i, 10, f"y{i}") for i in range(5)],
            attack_params=dict(_GCG_CONFIG),
        )

        out_file = tmp_path / "out.jsonl"
        convert_run_dir(str(run_dir), str(out_file))

        with open(out_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        behaviors = {r["behavior"] for r in rows}
        assert behaviors == {"beh_long"}

    @pytest.mark.parametrize(
        ("attack_params", "expected_count"),
        [
            (dict(_PAIR_CONFIG), 1),
            (dict(_AUTODAN_CONFIG), 0),
        ],
    )
    def test_pair_is_exempt_from_min_steps_filter(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        attack_params: dict[str, object],
        expected_count: int,
    ) -> None:
        monkeypatch.setattr(extract_adversariallm_attacks, "MIN_ATTACK_STEPS", 100)
        run_dir = tmp_path / "adv_out"
        _write_run_json(
            run_dir / "0" / "run.json",
            behavior="beh_A",
            steps=[_gcg_step(i, 1.0 - 0.1 * i, 10, f"s{i}") for i in range(3)],
            attack_params=attack_params,
        )

        out_file = tmp_path / "out.jsonl"
        convert_run_dir(str(run_dir), str(out_file))

        with open(out_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == expected_count


class TestRetryTimestamps:
    @pytest.mark.parametrize(
        "timestamps",
        [
            ["2026-05-15/14-48-32", "2026-05-19/06-26-56"],
            ["2026-05-15/14-48-32", "2026-05-15/22-45-01", "2026-05-19/06-26-56"],
        ],
    )
    def test_latest_timestamp_dir_wins(
        self, tmp_path: Path, timestamps: list[str]
    ) -> None:
        run_dir = tmp_path / "adv_out"
        for ts in timestamps:
            day, time = ts.split("/")
            _write_run_json(
                run_dir / day / time / "0" / "run.json",
                behavior="beh_A",
                steps=[_gcg_step(0, 0.5, 10, ts)],
            )

        out_file = tmp_path / "out.jsonl"
        convert_run_dir(str(run_dir), str(out_file))

        with open(out_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == 1
        assert rows[0]["adversarial_prompt"] == timestamps[-1]

    def test_disjoint_behaviors_across_timestamps_are_unioned(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "adv_out"
        _write_run_json(
            run_dir / "2026-05-15" / "14-48-32" / "0" / "run.json",
            behavior="beh_A",
            steps=[_gcg_step(0, 1.0, 10, "A_old")],
        )
        _write_run_json(
            run_dir / "2026-05-19" / "06-26-56" / "0" / "run.json",
            behavior="beh_A",
            steps=[_gcg_step(0, 0.1, 10, "A_new")],
        )
        _write_run_json(
            run_dir / "2026-05-19" / "06-26-56" / "1" / "run.json",
            behavior="beh_B",
            steps=[_gcg_step(0, 0.1, 10, "B_only")],
        )

        out_file = tmp_path / "out.jsonl"
        convert_run_dir(str(run_dir), str(out_file))

        with open(out_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        by_behavior = {r["behavior"]: r["adversarial_prompt"] for r in rows}
        assert by_behavior == {"beh_A": "A_new", "beh_B": "B_only"}
