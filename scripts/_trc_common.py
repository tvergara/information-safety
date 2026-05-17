"""Shared constants + shell helpers for the trc job submitters/bootstrappers.

The trc cluster mounts our private workspace at ``/work`` (read-write inside
the container, opaque outside). Every script that submits an ``eai job
submit`` references the same image, mount, and on-volume paths — collected
here so they stay consistent and there is one place to update them.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

_EAI_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

TRC_BASE_DIR = "/work/information-safety-results"
TRC_REPO_DIR = "/work/information-safety"
TRC_HF_HOME = "/work/.hf-cache"
TRC_EAI_IMAGE = (
    "registry.toolkit-sp.yul201.service-now.com/snow.research.mmteb/mteb-lite:v1"
)
TRC_DATA_MOUNT = "snow.research.mmteb.safety:/work:rw"


def shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def extract_eai_uuid(stdout: str) -> str:
    m = _EAI_UUID_RE.search(stdout)
    if m is None:
        raise ValueError(f"no UUID found in eai job submit output: {stdout!r}")
    return m.group()


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


def current_git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_sha_pushed(sha: str) -> None:
    result = subprocess.run(
        ["git", "branch", "-r", "--contains", sha],
        check=True,
        capture_output=True,
        text=True,
    )
    if not result.stdout.strip():
        raise RuntimeError(
            f"local sha {sha[:8]} is not on any remote branch; "
            "push before submitting to TRC"
        )


def default_state_file(name: str) -> Path:
    return (
        Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
        / "information-safety"
        / name
    )


def append_state_row(state_file: Path, row: dict[str, Any]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def build_eai_submit_remote(
    *,
    name: str,
    container_cmd: str,
    cpu: int,
    mem: int,
    max_run_time: int,
    preemptable: bool,
    gpu: int = 0,
) -> str:
    gpu_flag = f" --gpu {gpu}" if gpu else ""
    preempt_flag = "--preemptable" if preemptable else "--non-preemptable"
    return (
        f"eai job submit"
        f" --name {name}"
        f" --image {TRC_EAI_IMAGE}"
        f" --data {TRC_DATA_MOUNT}"
        f"{gpu_flag} --cpu {cpu} --mem {mem} --max-run-time {max_run_time}"
        f" {preempt_flag}"
        f" --enforce-name"
        f" -- bash -lc {shell_quote(container_cmd)}"
    )


def trc_behaviors_csv(dataset: str) -> str:
    if dataset == "harmbench":
        return f"{TRC_BASE_DIR}/attacks/harmbench_behaviors_standard.csv"
    if dataset == "strongreject":
        return f"{TRC_BASE_DIR}/attacks/strongreject_behaviors.csv"
    if dataset == "evilmath":
        return f"{TRC_BASE_DIR}/attacks/evilmath_behaviors.csv"
    raise ValueError(f"Unknown dataset: {dataset}")


def trc_targets_json(dataset: str) -> str:
    if dataset == "harmbench":
        return (
            f"{TRC_REPO_DIR}/third_party/adversariallm/data/"
            "optimizer_targets/harmbench_targets_text.json"
        )
    if dataset == "strongreject":
        return f"{TRC_BASE_DIR}/attacks/strongreject_targets_text.json"
    if dataset == "evilmath":
        return f"{TRC_BASE_DIR}/attacks/evilmath_targets_text.json"
    raise ValueError(f"Unknown dataset: {dataset}")
