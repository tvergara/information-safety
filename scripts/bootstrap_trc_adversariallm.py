"""Bootstrap the AdversariaLLM venv + behaviors files on the trc /work volume.

Run once per trc workspace; subsequent attack-shard submissions assume
``/work/envs/adversariallm/.venv`` and the behaviors CSVs/targets JSONs are
present.
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
    current_git_sha,
    shell_quote,
)

__all__ = [
    "TRC_ADVERSARIALLM_VENV",
    "TRC_BASE_DIR",
    "TRC_EAI_IMAGE",
    "TRC_HF_HOME",
    "TRC_REPO_DIR",
    "build_eai_submit_argv",
    "main",
]

TRC_ADVERSARIALLM_VENV = "/work/envs/adversariallm/.venv"

_MONGOMOCK_SHIM_PY = (
    "import os\n"
    "if os.environ.get('ADVERSARIALLM_USE_MONGOMOCK') == '1':\n"
    "    import mongomock, pymongo\n"
    "    pymongo.MongoClient = mongomock.MongoClient\n"
)


def build_eai_submit_argv(*, hf_token: str, git_sha: str) -> list[str]:
    site_py = "import site; print(site.getsitepackages()[0])"
    install_adv = (
        f"(source {TRC_ADVERSARIALLM_VENV}/bin/activate"
        " && uv pip install --index-strategy unsafe-best-match"
        " --extra-index-url https://download.pytorch.org/whl/cu128"
        f" -e {TRC_REPO_DIR}/third_party/adversariallm"
        " && uv pip install mongomock"
        f" && SITE=$(python -c {shell_quote(site_py)})"
        f" && printf %s {shell_quote(_MONGOMOCK_SHIM_PY)} > $SITE/_mongomock_shim.py"
        " && printf 'import _mongomock_shim\\n' > $SITE/_mongomock_shim.pth)"
    )
    container_cmd = " && ".join([
        f"mkdir -p {TRC_HF_HOME}",
        f"printf %s {shell_quote(hf_token)} > {TRC_HF_HOME}/token",
        f"chmod 600 {TRC_HF_HOME}/token",
        "unset TRANSFORMERS_CACHE",
        f"export HF_HOME={TRC_HF_HOME}",
        f"export HF_HUB_CACHE={TRC_HF_HOME}/hub",
        f"export HUGGING_FACE_HUB_TOKEN={shell_quote(hf_token)}",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        'export PATH="$HOME/.local/bin:$PATH"',
        f"cd {TRC_REPO_DIR}",
        "git fetch --all",
        f"git checkout {git_sha}",
        "git submodule update --init --recursive",
        "source .venv/bin/activate",
        f"uv venv {TRC_ADVERSARIALLM_VENV} --python 3.10 --clear",
        install_adv,
        (
            f"python {TRC_REPO_DIR}/scripts/build_adversariallm_behaviors.py"
            f" --behaviors-csv-dir {TRC_BASE_DIR}/attacks"
            " --with-strongreject"
        ),
    ])
    remote = build_eai_submit_remote(
        name="is_bootstrap_adversariallm",
        container_cmd=container_cmd,
        cpu=4,
        mem=16,
        max_run_time=7200,
        preemptable=False,
    )
    return ["ssh", "trc", remote]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    token = get_token()
    if token is None:
        print(
            "ERROR: no HF token found. Run `huggingface-cli login` first.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    cli_argv = build_eai_submit_argv(hf_token=token, git_sha=current_git_sha())
    if args.dry_run:
        redacted = [s.replace(token, "<redacted>") for s in cli_argv]
        print(f"{redacted[0]} {redacted[1]} {shell_quote(redacted[2])}")
        return
    result = subprocess.run(cli_argv, check=True, capture_output=True, text=True)
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
