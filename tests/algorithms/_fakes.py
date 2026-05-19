"""Shared test fakes for algorithm tests."""

from __future__ import annotations

from typing import Any


class FakeStrategy:
    """Concrete stand-in mirroring BaseStrategy's default FLOPs hooks."""

    def __init__(
        self,
        attack_flops_by_label: dict[str, int],
        label_key: str = "behavior",
    ) -> None:
        self._attack_flops_by_label = attack_flops_by_label
        self._label_key = label_key

    def attack_flops_for(self, meta: dict[str, Any]) -> int:
        return self._attack_flops_by_label.get(meta[self._label_key], 0)

    def eval_forward_flops(self, num_params: int, n_input: int, n_output: int) -> int:
        return 2 * num_params * (n_input + n_output)
