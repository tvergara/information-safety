"""Export a Lightning checkpoint with PEFT/LoRA to a merged HuggingFace model.

Loads the checkpoint, reconstructs the PEFT model, merges LoRA adapters into base weights, and
saves as a standard HuggingFace model directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def export(
    checkpoint_path: str,
    base_model_name: str,
    output_dir: str,
) -> None:
    """Merge LoRA weights from a Lightning checkpoint and save as HF model."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    lora_a_key = next(k for k in state_dict if "lora_A" in k)
    lora_rank = state_dict[lora_a_key].shape[0]
    print(f"Detected LoRA rank: {lora_rank}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank,
        lora_dropout=0.0,
        target_modules="all-linear",
    )
    peft_model = get_peft_model(base_model, lora_config)

    stripped_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            stripped_state_dict[key[len("model."):]] = value

    peft_model.load_state_dict(stripped_state_dict, strict=True)
    merged_model = peft_model.merge_and_unload()  # pyright: ignore[reportCallIssue]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path)  # pyright: ignore[reportCallIssue]
    tokenizer.save_pretrained(output_path)
    print(f"Saved merged model to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Lightning+LoRA checkpoint to HF model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--base-model", type=str, default="HuggingFaceTB/SmolLM3-3B",
        help="HF base model name",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    export(args.checkpoint, args.base_model, args.output_dir)


if __name__ == "__main__":
    main()
