"""Bootstrap the information-safety venv + HF cache on the trc /work volume.

Run once per trc workspace; subsequent ``migrate_pool_to_trc.py`` submissions
assume ``/work/envs/information-safety/.venv`` exists, the HF cache under
``/work/.hf-cache/hub`` is primed with the four DataStrategy models, and these
HuggingFace datasets are materialized so workers can run with
``HF_DATASETS_OFFLINE=1``:

  * ``cais/wmdp`` (subsets ``wmdp-bio``, ``wmdp-cyber``, ``wmdp-chem``) — eval.
  * ``TIGER-Lab/MMLU-Pro`` — DataStrategy training corpus.
  * ``cais/wmdp-corpora`` (``cyber-forget-corpus`` config) — corpus_subset=cyber.
  * ``cais/wmdp-bio-forget-corpus`` — corpus_subset=bio.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from huggingface_hub import get_token

from scripts._trc_common import (
    TRC_EAI_IMAGE,
    TRC_HF_HOME,
    TRC_REPO_DIR,
    build_eai_submit_remote,
    current_git_sha,
    shell_quote,
)

__all__ = [
    "HF_MODELS_TO_PREFETCH",
    "TRC_EAI_IMAGE",
    "TRC_HF_HOME",
    "TRC_INFORMATION_SAFETY_VENV",
    "TRC_REPO_DIR",
    "WMDP_DATASETS_TO_PREFETCH",
    "build_eai_submit_argv",
    "current_git_sha",
    "main",
]

TRC_INFORMATION_SAFETY_VENV = "/work/envs/information-safety/.venv"

HF_MODELS_TO_PREFETCH: list[str] = [
    "Qwen/Qwen3-4B",
    "allenai/Olmo-3-7B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "openai/gpt-oss-20b",
]

WMDP_DATASETS_TO_PREFETCH: list[str] = [
    "wmdp-bio",
    "wmdp-cyber",
    "wmdp-chem",
]

_WMDP_LOAD_LINES = "\n".join(
    f"load_dataset('cais/wmdp', '{subset}', split='test')"
    for subset in WMDP_DATASETS_TO_PREFETCH
)
_DATASET_PREFETCH_PY = (
    "from datasets import load_dataset\n"
    f"{_WMDP_LOAD_LINES}\n"
    "load_dataset('TIGER-Lab/MMLU-Pro', split='test')\n"
    "load_dataset('cais/wmdp-corpora', 'cyber-forget-corpus', split='train')\n"
    "load_dataset('cais/wmdp-bio-forget-corpus', split='train')\n"
)

_SMOKE_HYDRA_OVERRIDES = " ".join([
    "experiment=finetune-with-strategy",
    "algorithm/strategy=data",
    "algorithm.model.pretrained_model_name_or_path=Qwen/Qwen3-4B",
    "algorithm/dataset_handler=wmdp",
    "algorithm.dataset_handler.max_examples=2",
    "trainer.max_epochs=1",
    "debug=true",
])


def build_eai_submit_argv(*, hf_token: str, git_sha: str) -> list[str]:
    install_pkg = (
        f"(source {TRC_INFORMATION_SAFETY_VENV}/bin/activate"
        " && uv pip install --index-strategy unsafe-best-match"
        " --extra-index-url https://download.pytorch.org/whl/cu128"
        f" -e {TRC_REPO_DIR})"
    )
    prefetch_models = " && ".join(
        f"huggingface-cli download {model_id}" for model_id in HF_MODELS_TO_PREFETCH
    )
    prefetch_datasets = (
        f"(source {TRC_INFORMATION_SAFETY_VENV}/bin/activate"
        f" && python -c {shell_quote(_DATASET_PREFETCH_PY)})"
    )
    smoke_test = (
        f"(source {TRC_INFORMATION_SAFETY_VENV}/bin/activate"
        f" && cd {TRC_REPO_DIR}"
        f" && python information_safety/main.py {_SMOKE_HYDRA_OVERRIDES})"
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
        f"uv venv {TRC_INFORMATION_SAFETY_VENV} --python 3.10 --clear",
        install_pkg,
        prefetch_models,
        prefetch_datasets,
        smoke_test,
    ])
    remote = build_eai_submit_remote(
        name="is_bootstrap_information_safety",
        container_cmd=container_cmd,
        gpu=1,
        cpu=8,
        mem=64,
        max_run_time=14400,
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
