"""TAR (Tamper-Resistant Safeguards) defense adapter.

Drives the vendored ``tar/`` repo via subprocess. The inner-attacker mix is the
exact data our ``DataStrategy`` attack consumes for each target, so the defense
is hardened against the attack matrix end-to-end.

- ``--target wmdp``: WMDP forget-corpus + MMLU-Pro 50/50 (matches
  ``WMDPHandler(corpus_fraction=0.5)``).
- ``--target evilmath``: GSM8K (matches ``EvilMathHandler.get_train_dataset``).
- ``--target both``: union of the two, equally sized.

We do **not** rewrite TAR's training loop — the adapter only translates data and
shells out to TAR's training entry point.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, cast

import datasets
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from information_safety.algorithms.dataset_handlers.wmdp import (
    CORPUS_REPOS,
    MULTIPLE_CHOICE_TEMPLATE_10,
)

logger = getLogger(__name__)


@dataclass
class TARHParams:
    inner_steps: int = 4
    inner_lr: float = 2e-5
    outer_lr: float = 2e-5
    outer_steps: int = 200
    masking_layers: str = "all"
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    inner_attacker_examples: int = 1024
    extra_args: list[str] = field(default_factory=list)


def _build_tar_format(
    refusals_path: Path, retain_path: Path, out_path: Path
) -> int:
    """Translate refusals + retain JSONL into TAR's expected JSONL schema.

    Each row has ``prompt``, ``response`` and ``split`` (``refusal`` or
    ``retain``). TAR's training entry point reads this directly.
    """
    rows: list[dict[str, Any]] = []
    with refusals_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            rows.append({
                "prompt": entry["prompt"],
                "response": entry["refusal"],
                "split": "refusal",
                "source": entry["source"],
            })

    with retain_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            rows.append({
                "prompt": entry["prompt"],
                "response": entry["response"],
                "split": "retain",
                "source": entry["source"],
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


def _format_mmlu_prompt(example: dict[str, Any]) -> str:
    options: list[str] = example["options"]
    return MULTIPLE_CHOICE_TEMPLATE_10.format(
        question=example["question"],
        choice_0=options[0],
        choice_1=options[1],
        choice_2=options[2],
        choice_3=options[3],
        choice_4=options[4],
        choice_5=options[5],
        choice_6=options[6],
        choice_7=options[7],
        choice_8=options[8],
        choice_9=options[9],
    )


def _wmdp_inner_examples(num_examples: int) -> list[dict[str, str]]:
    """50/50 mix of MMLU-Pro CoT + WMDP forget corpus (bio + cyber)."""
    half = num_examples // 2
    quarter = half // 2

    mmlu = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    mmlu = mmlu.filter(lambda ex: len(ex["options"]) == 10)  # type: ignore[union-attr]
    mmlu = mmlu.select(range(min(half, len(mmlu))))  # type: ignore[union-attr,arg-type]

    examples: list[dict[str, str]] = []
    for ex in mmlu:
        ex_dict: dict[str, Any] = dict(ex)  # type: ignore[arg-type]
        examples.append({
            "prompt": _format_mmlu_prompt(ex_dict),
            "response": f"{ex_dict['cot_content']}\n\nfinal {ex_dict['answer']}",
        })

    for subject in ("bio", "cyber"):
        repo, config = CORPUS_REPOS[subject]
        if config is None:
            corpus = datasets.load_dataset(repo, split="train")
        else:
            corpus = datasets.load_dataset(repo, config, split="train")
        corpus = corpus.select(range(min(quarter, len(corpus))))  # type: ignore[union-attr,arg-type]
        for ex in corpus:
            ex_dict = dict(ex)  # type: ignore[arg-type]
            examples.append({"prompt": "", "response": ex_dict["text"]})

    return examples


def _evilmath_inner_examples(num_examples: int) -> list[dict[str, str]]:
    gsm = datasets.load_dataset("openai/gsm8k", "main", split="train")
    gsm = gsm.select(range(min(num_examples, len(gsm))))  # type: ignore[union-attr,arg-type]
    out: list[dict[str, str]] = []
    for ex in gsm:
        ex_dict: dict[str, Any] = dict(ex)  # type: ignore[arg-type]
        out.append({"prompt": ex_dict["question"], "response": ex_dict["answer"]})
    return out


def _build_inner_attacker_data(
    target: str, out_path: Path, num_examples: int
) -> int:
    if target == "wmdp":
        examples = _wmdp_inner_examples(num_examples)
    elif target == "evilmath":
        examples = _evilmath_inner_examples(num_examples)
    elif target == "both":
        half = num_examples // 2
        examples = _wmdp_inner_examples(half) + _evilmath_inner_examples(half)
    else:
        raise ValueError(f"Unknown TAR target: {target!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return len(examples)


def _build_subprocess_cmd(
    tar_repo: Path,
    base_model: str,
    tar_data_path: Path,
    attacker_data_path: Path,
    output_dir: Path,
    hparams: TARHParams,
) -> list[str]:
    cmd: list[str] = [
        "python",
        str(tar_repo / "train.py"),
        "--model_name_or_path",
        base_model,
        "--train_set_path",
        str(tar_data_path),
        "--attacker_data",
        str(attacker_data_path),
        "--output_dir",
        str(output_dir),
        "--inner_steps",
        str(hparams.inner_steps),
        "--inner_lr",
        str(hparams.inner_lr),
        "--outer_lr",
        str(hparams.outer_lr),
        "--max_steps",
        str(hparams.outer_steps),
        "--masking_layers",
        hparams.masking_layers,
        "--per_device_train_batch_size",
        str(hparams.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(hparams.gradient_accumulation_steps),
        "--bf16",
        "True",
        "--report_to",
        "none",
    ]
    cmd.extend(hparams.extra_args)
    return cmd


def _merge_lora_into_base(base_model: str, adapter_dir: Path, output_dir: Path) -> None:
    base = AutoModelForCausalLM.from_pretrained(base_model)
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = cast(PreTrainedModel, peft_model.merge_and_unload())  # type: ignore[operator]
    merged.save_pretrained(str(output_dir))
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(str(output_dir))


def _has_full_model_weights(directory: Path) -> bool:
    return (directory / "pytorch_model.bin").exists() or (
        directory / "model.safetensors"
    ).exists()


def train_tar(
    base_model: str,
    refusals_path: Path,
    retain_path: Path,
    output_dir: Path,
    target: str,
    hparams: TARHParams,
    tar_repo: Path,
    dry_run: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_data_path = Path(refusals_path).parent / "tar_format.jsonl"
    n_rows = _build_tar_format(refusals_path, retain_path, tar_data_path)
    logger.info(f"Wrote {n_rows} TAR-formatted rows to {tar_data_path}")

    attacker_data_path = Path(refusals_path).parent / "tar_inner_attacker.jsonl"
    n_inner = _build_inner_attacker_data(
        target, attacker_data_path, hparams.inner_attacker_examples
    )
    logger.info(f"Wrote {n_inner} inner-attacker rows to {attacker_data_path}")

    cmd = _build_subprocess_cmd(
        tar_repo, base_model, tar_data_path, attacker_data_path, output_dir, hparams
    )

    if dry_run:
        logger.info(f"[dry-run] TAR command: {' '.join(cmd)}")
        return

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    result = subprocess.run(cmd, cwd=str(tar_repo), env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"TAR training failed with returncode {result.returncode}")

    if not _has_full_model_weights(output_dir):
        _merge_lora_into_base(base_model, output_dir, output_dir)

    if not _has_full_model_weights(output_dir):
        raise RuntimeError(
            f"TAR output_dir {output_dir} missing merged model weights "
            "(pytorch_model.bin or model.safetensors)"
        )
