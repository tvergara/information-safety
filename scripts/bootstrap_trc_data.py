"""Bootstrap the trc /work volume with attack/data artifacts pulled from HF.

Submits a one-shot ``eai job submit`` container that:

1. Writes the user's HF token to ``$HF_HOME/token`` so subsequent jobs can
   read from private dataset repos without further configuration.
2. Calls ``huggingface_hub.snapshot_download`` to materialize the private
   dataset (attacks/ and data/) into ``/work/information-safety-results/``.

Run once per trc workspace; subsequent job submissions can then assume
the artifacts are present on /work.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from huggingface_hub import get_token

from scripts._trc_common import (
    TRC_BASE_DIR,
    TRC_EAI_IMAGE,
    TRC_HF_HOME,
    TRC_REPO_DIR,
    build_eai_submit_remote,
    shell_quote,
)

__all__ = [
    "TRC_BASE_DIR",
    "TRC_EAI_IMAGE",
    "TRC_HF_HOME",
    "build_eai_submit_argv",
    "main",
]


def build_eai_submit_argv(*, hf_token: str, dataset_repo: str) -> list[str]:
    snapshot_py = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={dataset_repo!r}, repo_type='dataset', "
        f"local_dir={TRC_BASE_DIR!r})"
    )
    container_cmd = " && ".join([
        f"mkdir -p {TRC_HF_HOME}",
        f"printf %s {shell_quote(hf_token)} > {TRC_HF_HOME}/token",
        f"chmod 600 {TRC_HF_HOME}/token",
        f"mkdir -p {TRC_BASE_DIR}",
        f"cd {TRC_REPO_DIR}",
        "source .venv/bin/activate",
        f"export HF_HOME={TRC_HF_HOME}",
        f"export HUGGING_FACE_HUB_TOKEN={shell_quote(hf_token)}",
        f"python -c {shell_quote(snapshot_py)}",
        f"ls -la {TRC_BASE_DIR}",
    ])
    remote = build_eai_submit_remote(
        name="is_bootstrap_data",
        container_cmd=container_cmd,
        cpu=4,
        mem=16,
        max_run_time=3600,
        preemptable=False,
    )
    return ["ssh", "trc", remote]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-repo",
        default="tvergara/information-safety-attack-artifacts",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    token = get_token()
    if token is None:
        print(
            "ERROR: no HF token found. Run `huggingface-cli login` first.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    cli_argv = build_eai_submit_argv(hf_token=token, dataset_repo=args.dataset_repo)
    if args.dry_run:
        redacted = [s.replace(token, "<redacted>") for s in cli_argv]
        print(f"{redacted[0]} {redacted[1]} {shell_quote(redacted[2])}")
        return
    result = subprocess.run(cli_argv, check=True, capture_output=True, text=True)
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
