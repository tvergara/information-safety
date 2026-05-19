"""Regression guard for run_attacks.py per-behavior failure isolation.

When PAIR's attacker-LM exhausts its parse-retries for a single behavior, it
raises ``ValueError``. Without isolation, that exception propagates and kills
the entire shard. The patch in ``third_party/adversariallm/run_attacks.py``
wraps ``attack.run(model, tokenizer, [conversation])`` in a try/except so a
single bad behavior doesn't waste the other behaviors in the shard.

This test inspects the source of run_attacks.py to ensure the isolation
pattern is present, and also exercises the pattern in isolation to confirm
it has the intended semantics.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

RUN_ATTACKS = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "adversariallm"
    / "run_attacks.py"
)


def _find_run_attacks_function() -> ast.FunctionDef:
    tree = ast.parse(RUN_ATTACKS.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_attacks":
            return node
    raise AssertionError("run_attacks function not found in run_attacks.py")


def test_run_attacks_wraps_attack_run_in_try_except() -> None:
    fn = _find_run_attacks_function()
    has_try_calling_attack_run = False
    for node in ast.walk(fn):
        if not isinstance(node, ast.Try):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "run"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "attack"
            ):
                has_try_calling_attack_run = True
                break
        if has_try_calling_attack_run:
            break
    assert has_try_calling_attack_run, (
        "Expected attack.run(...) to be wrapped in try/except inside "
        "run_attacks() so per-behavior failures don't kill the shard."
    )


def test_per_behavior_loop_continues_after_value_error() -> None:
    """Functional smoke test: emulate the patched loop and verify isolation."""
    call_log: list[str] = []

    class FakeAttack:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, model, tokenizer, dataset):
            self.calls += 1
            conversation = dataset[0]
            if conversation == "bad":
                raise ValueError(
                    "Failed to generate output after 10 attempts, "
                    "run again with more attempts or fewer steps."
                )
            return {"runs": [{"conversation": conversation}]}

    def log_attack_mock(idx: int, result) -> None:
        call_log.append(f"logged idx={idx}")

    attack = FakeAttack()
    dataset = ["good_1", "bad", "good_2"]
    full_idx_list = [101, 102, 103]
    skipped: list[int] = []

    for behavior_pos, conversation in enumerate(dataset):
        try:
            single_result = attack.run(None, None, [conversation])
        except Exception as e:
            skipped.append(full_idx_list[behavior_pos])
            assert isinstance(e, ValueError)
            continue
        log_attack_mock(full_idx_list[behavior_pos], single_result)

    assert attack.calls == 3, "attack.run should be invoked once per behavior"
    assert skipped == [102], "Only the bad behavior should be skipped"
    assert call_log == ["logged idx=101", "logged idx=103"], (
        "Good behaviors should still be logged after a bad behavior fails"
    )


def test_run_attacks_file_imports_copy() -> None:
    """The per-behavior patch needs copy.deepcopy for partial_config."""
    tree = ast.parse(RUN_ATTACKS.read_text())
    imports = {n.name for node in ast.walk(tree)
               if isinstance(node, ast.Import)
               for n in node.names}
    assert "copy" in imports, "run_attacks.py must import copy for per-behavior partial_config"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
