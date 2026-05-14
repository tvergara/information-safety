"""Shared constants + shell helpers for the trc job submitters/bootstrappers.

The trc cluster mounts our private workspace at ``/work`` (read-write inside
the container, opaque outside). Every script that submits an ``eai job
submit`` references the same image, mount, and on-volume paths — collected
here so they stay consistent and there is one place to update them.
"""

from __future__ import annotations

import json
from pathlib import Path

TRC_BASE_DIR = "/work/information-safety-results"
TRC_REPO_DIR = "/work/information-safety"
TRC_HF_HOME = "/work/.hf-cache"
TRC_EAI_IMAGE = (
    "registry.toolkit-sp.yul201.service-now.com/snow.research.mmteb/mteb-lite:v1"
)
TRC_DATA_MOUNT = "snow.research.mmteb.safety:/work:rw"


def shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_")


def compute_shards(*, total: int, num_shards: int) -> list[tuple[int, int]]:
    base = total // num_shards
    remainder = total % num_shards
    shards: list[tuple[int, int]] = []
    cursor = 0
    for i in range(num_shards):
        size = base + (1 if i < remainder else 0)
        if size == 0:
            continue
        shards.append((cursor, cursor + size))
        cursor += size
    return shards


def load_submitted_state(state_file: Path) -> set[str]:
    if not state_file.exists():
        return set()
    return {
        json.loads(line)["job_id"]
        for line in state_file.read_text().splitlines()
        if line.strip()
    }


def trc_behaviors_csv(dataset: str) -> str:
    if dataset == "harmbench":
        return f"{TRC_BASE_DIR}/attacks/harmbench_behaviors_standard.csv"
    if dataset == "strongreject":
        return f"{TRC_BASE_DIR}/attacks/strongreject_behaviors.csv"
    raise ValueError(f"Unknown dataset: {dataset}")


def trc_targets_json(dataset: str) -> str:
    if dataset == "harmbench":
        return (
            f"{TRC_REPO_DIR}/third_party/adversariallm/data/"
            "optimizer_targets/harmbench_targets_text.json"
        )
    if dataset == "strongreject":
        return f"{TRC_BASE_DIR}/attacks/strongreject_targets_text.json"
    raise ValueError(f"Unknown dataset: {dataset}")
