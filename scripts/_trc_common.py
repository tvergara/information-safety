"""Shared constants + shell helpers for the trc job submitters/bootstrappers.

The trc cluster mounts our private workspace at ``/work`` (read-write inside
the container, opaque outside). Every script that submits an ``eai job
submit`` references the same image, mount, and on-volume paths — collected
here so they stay consistent and there is one place to update them.
"""

from __future__ import annotations

TRC_BASE_DIR = "/work/information-safety-results"
TRC_REPO_DIR = "/work/information-safety"
TRC_HF_HOME = "/work/.hf-cache"
TRC_EAI_IMAGE = (
    "registry.toolkit-sp.yul201.service-now.com/snow.research.mmteb/mteb-lite:v1"
)
TRC_DATA_MOUNT = "snow.research.mmteb.safety:/work:rw"


def shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"
