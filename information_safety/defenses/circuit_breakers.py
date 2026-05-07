"""Circuit Breakers defense adapter: drives the vendored `circuit-breakers/` repo.

We do **not** rewrite CB's training loop. The adapter:
1. Translates our refusals + retain JSONL into CB's expected schema (a JSON list
   with ``prompt``, ``output``, ``llama3_output`` fields, matching the format
   ``cb_train_dataset.py`` reads from ``data/circuit_breakers_train.json``).
2. Shells out to ``circuit-breakers/src/lorra_circuit_breaker.py`` via
   ``accelerate launch``.
3. Loads the resulting LoRA adapter, merges into the base model, and saves a
   HF-loadable checkpoint at ``output_dir``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, cast

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = getLogger(__name__)

CB_TRAIN_DATA_RELPATH = Path("data") / "circuit_breakers_train.json"


@dataclass
class CBHParams:
    target_layers: str = "10,20"
    transform_layers: str = "-1"
    lorra_alpha: float = 10.0
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_steps: int = 150
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    model_max_length: int = 8192
    use_refusal_retain: bool = True
    extra_args: list[str] = field(default_factory=list)


def _build_cb_format(refusals_path: Path, out_path: Path) -> int:
    """Translate our refusals JSONL into the CB native format and write a JSON list.

    CB's loader iterates EVERY row of ``data/circuit_breakers_train.json`` and treats it
    as a target whose representations should be pushed away from base. Retain rows must
    NOT be written here — CB loads its retain set from ultrachat_200k + xstest internally.
    Writing retain rows collapses MMLU because the model learns to circuit-break on
    benign content (we hit acc=0.17 vs the published ~0.63).
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
                "output": entry["prompt"],
                "llama3_output": entry["refusal"],
                "source": entry["source"],
                "id": entry["id"],
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows))
    return len(rows)


def _install_cb_train_data(cb_repo: Path, cb_data_path: Path) -> None:
    target = cb_repo / CB_TRAIN_DATA_RELPATH
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_suffix(target.suffix + ".bak")
    if target.exists() and not target.is_symlink() and not backup.exists():
        target.rename(backup)
    elif target.exists() or target.is_symlink():
        target.unlink()
    shutil.copyfile(cb_data_path, target)


def _restore_cb_train_data(cb_repo: Path) -> None:
    target = cb_repo / CB_TRAIN_DATA_RELPATH
    backup = target.with_suffix(target.suffix + ".bak")
    if not backup.exists():
        return
    if target.exists() or target.is_symlink():
        target.unlink()
    backup.rename(target)


def _build_subprocess_cmd(
    cb_repo: Path,
    base_model: str,
    output_dir: Path,
    hparams: CBHParams,
    dry_run: bool = False,
) -> list[str]:
    if dry_run and not torch.cuda.is_available():
        num_processes = 2
    else:
        assert torch.cuda.is_available(), "CB training requires GPU"
        num_processes = torch.cuda.device_count()
    cmd: list[str] = [
        "accelerate",
        "launch",
        "--config_file",
        str(cb_repo / "configs" / "accelerate_zero1.yaml"),
        "--num_processes",
        str(num_processes),
        str(cb_repo / "src" / "lorra_circuit_breaker.py"),
        "--model_name_or_path",
        base_model,
        "--output_dir",
        str(output_dir),
        "--target_layers",
        hparams.target_layers,
        "--transform_layers",
        hparams.transform_layers,
        "--lorra_alpha",
        str(hparams.lorra_alpha),
        "--lora_r",
        str(hparams.lora_r),
        "--lora_alpha",
        str(hparams.lora_alpha),
        "--lora_dropout",
        str(hparams.lora_dropout),
        "--max_steps",
        str(hparams.max_steps),
        "--learning_rate",
        str(hparams.learning_rate),
        "--per_device_train_batch_size",
        str(hparams.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(hparams.gradient_accumulation_steps),
        "--model_max_length",
        str(hparams.model_max_length),
        "--bf16",
        "True",
        "--gradient_checkpointing",
        "True",
        "--report_to",
        "none",
        "--save_total_limit",
        "0",
        "--logging_strategy",
        "steps",
        "--logging_steps",
        "10",
        "--lr_scheduler_type",
        "constant",
    ]
    if hparams.use_refusal_retain:
        cmd.append("--use_refusal_retain")
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


def train_circuit_breakers(
    base_model: str,
    refusals_path: Path,
    output_dir: Path,
    hparams: CBHParams,
    cb_repo: Path,
    dry_run: bool = False,
) -> None:
    cb_data_path = Path(refusals_path).parent / "cb_format.json"
    n_rows = _build_cb_format(refusals_path, cb_data_path)
    logger.info(f"Wrote {n_rows} CB-formatted rows to {cb_data_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = _build_subprocess_cmd(cb_repo, base_model, output_dir, hparams, dry_run=dry_run)

    if dry_run:
        logger.info(f"[dry-run] CB command: {' '.join(cmd)}")
        return

    _install_cb_train_data(cb_repo, cb_data_path)

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    try:
        result = subprocess.run(cmd, cwd=str(cb_repo), env=env, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Circuit Breakers training failed with returncode {result.returncode}"
            )
    finally:
        _restore_cb_train_data(cb_repo)

    if not _has_full_model_weights(output_dir):
        _merge_lora_into_base(base_model, output_dir, output_dir)

    if not _has_full_model_weights(output_dir):
        raise RuntimeError(
            f"CB output_dir {output_dir} missing merged model weights "
            "(pytorch_model.bin or model.safetensors)"
        )
