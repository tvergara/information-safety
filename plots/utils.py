"""Shared utilities for plot scripts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def pareto_frontier(
    bits: np.ndarray, asr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Pareto frontier maximizing ASR for a given bits budget.

    Returns sorted (bits, asr) arrays of Pareto-optimal points.
    """
    order = np.argsort(bits)
    sorted_bits = bits[order]
    sorted_asr = asr[order]

    pareto_bits: list[float] = []
    pareto_asr: list[float] = []
    running_max = -1.0

    for b, a in zip(sorted_bits, sorted_asr):
        if a > running_max:
            running_max = a
            pareto_bits.append(b)
            pareto_asr.append(a)

    return np.array(pareto_bits), np.array(pareto_asr)
