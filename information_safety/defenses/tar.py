"""TAR (Tamper-Resistant Safeguards) defense adapter.

Drives the vendored ``tar/`` repo by shelling out to ``tar/tar.py`` via
``accelerate launch``. We do not rewrite TAR's training loop.

For ``target == "wmdp"`` we piggyback on TAR's existing ``--subject bio``
loader (pile-bio + camel-bio is the WMDP-bio forget corpus).

For ``target == "evilmath"`` we use ``--subject evilmath``, which our patched
``tar/tar.py`` routes to a new dataloader that tokenizes the harmful narratives
at the caller-supplied ``evilmath_data_path`` as the forget corpus.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

import torch
from transformers import AutoTokenizer

logger = getLogger(__name__)


@dataclass
class TARHParams:
    max_steps: int = 200
    tar_inner_loop_steps: int = 16
    inner_optimizer_warmup_steps: int = 20
    warmup_steps: int = 32
    lr: float = 2e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    tar_adversary_batch_size: int = 4
    tar_inner_loop_subsample: int = 4
    tar_tamper_resistance_grad_scale: float = 4.0
    tar_retain_scale: float = 1.0
    schedule_lambda: float = 0.0625
    adversary_lr_samples: str = "2e-6,2e-5,4e-5"
    adversary_lr_schedulers: str = "constant:1.0"
    extra_args: list[str] = field(default_factory=list)


def _adversary_dist_types_for(target: str) -> str:
    if target == "wmdp":
        return "pile-bio:0.33,camel-bio:0.33,retain_forget_switch:0.33"
    return "forget_train:0.5,retain_forget_switch:0.5"


def _build_subprocess_cmd(
    tar_repo: Path,
    base_model: str,
    output_dir: Path,
    target: str,
    hparams: TARHParams,
    evilmath_data_path: Path | None = None,
    dry_run: bool = False,
) -> list[str]:
    if target == "wmdp":
        subject = "bio"
    elif target == "evilmath":
        subject = "evilmath"
        if evilmath_data_path is None:
            raise ValueError("TAR evilmath requires evilmath_data_path")
    else:
        raise ValueError(f"Unknown TAR target: {target!r}")

    if dry_run and not torch.cuda.is_available():
        num_processes = 2
    else:
        assert torch.cuda.is_available(), "TAR training requires GPU"
        num_processes = torch.cuda.device_count()
    accel_config = tar_repo / "configs" / f"accel_config_{num_processes}_gpu.yaml"
    if not accel_config.exists():
        accel_config = tar_repo / "configs" / "accel_config_2_gpu.yaml"

    cmd: list[str] = [
        "accelerate",
        "launch",
        "--config_file",
        str(accel_config),
        "--num_processes",
        str(num_processes),
        str(tar_repo / "tar.py"),
        "--subject",
        subject,
        "--base_model_name",
        base_model,
        "--retain_model_name",
        base_model,
        "--tokenizer_name",
        base_model,
        "--output_dir",
        str(output_dir),
        "--trainer_type",
        "tar_trainer",
        "--max_steps",
        str(hparams.max_steps),
        "--tar_inner_loop_steps",
        str(hparams.tar_inner_loop_steps),
        "--tar_inner_loop_subsample",
        str(hparams.tar_inner_loop_subsample),
        "--tar_adversary_batch_size",
        str(hparams.tar_adversary_batch_size),
        "--tar_tamper_resistance_grad_scale",
        str(hparams.tar_tamper_resistance_grad_scale),
        "--tar_retain_scale",
        str(hparams.tar_retain_scale),
        "--schedule_lambda",
        str(hparams.schedule_lambda),
        "--inner_optimizer_warmup_steps",
        str(hparams.inner_optimizer_warmup_steps),
        "--warmup_steps",
        str(hparams.warmup_steps),
        "--lr",
        str(hparams.lr),
        "--batch_size",
        str(hparams.batch_size),
        "--gradient_accumulation_steps",
        str(hparams.gradient_accumulation_steps),
        "--adversary_lr_samples",
        hparams.adversary_lr_samples,
        "--adversary_lr_schedulers",
        hparams.adversary_lr_schedulers,
        "--adversary_dist_types",
        _adversary_dist_types_for(target),
        "--retain_representations",
        "--unbounded",
        "--use_weighting_schedule",
    ]

    if target == "evilmath":
        assert evilmath_data_path is not None
        cmd.extend(["--evilmath_data_path", str(evilmath_data_path)])

    cmd.extend(hparams.extra_args)
    return cmd


def _has_full_model_weights(directory: Path) -> bool:
    return (
        (directory / "pytorch_model.bin").exists()
        or (directory / "model.safetensors").exists()
        or (directory / "model.safetensors.index.json").exists()
    )


def train_tar(
    base_model: str,
    output_dir: Path,
    target: str,
    hparams: TARHParams,
    tar_repo: Path,
    evilmath_data_path: Path | None = None,
    dry_run: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if target == "evilmath":
        if evilmath_data_path is None:
            raise ValueError(
                "TAR evilmath requires evilmath_data_path "
                "(harmful-narrative rewrites from build_evilmath_refusal_source.py)"
            )
        if not evilmath_data_path.exists():
            raise FileNotFoundError(
                f"TAR evilmath data not found at {evilmath_data_path}; "
                "run scripts/build_evilmath_refusal_source.py first."
            )

    cmd = _build_subprocess_cmd(
        tar_repo,
        base_model,
        output_dir,
        target,
        hparams,
        evilmath_data_path=evilmath_data_path,
        dry_run=dry_run,
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
        raise RuntimeError(
            f"TAR output_dir {output_dir} missing merged model weights "
            "(pytorch_model.bin, model.safetensors, or sharded equivalent)"
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(str(output_dir))
