"""LoRRA circuit-breaker training for SmolLM3-3B.

Adapted from lorra_circuit_breaker.py — DeepSpeed removed (not available),
SmolLM3 chat template added, layer indices adjusted for 36-layer architecture.
"""

import csv
import json
import random

import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.nn.functional import cosine_similarity
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer

random.seed(0)
np.random.seed(0)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CircuitBreakerDataset(Dataset):
    """Paired retain / circuit-breaker dataset for SmolLM3."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, num_examples: int = 10000):
        super().__init__()
        self.max_length = 1024

        user_tag = "<|im_start|>user\n"
        assistant_tag = "<|im_end|>\n<|im_start|>assistant\n"
        one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"

        # ── Retain: UltraChat ────────────────────────────────────────────────
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        orig_s = []
        for example in ds:
            messages = example["messages"]
            if len(messages) < 2:
                continue
            switch = np.random.choice([0, 1])
            if switch == 0:
                text = tokenizer.apply_chat_template(messages, tokenize=False).replace(tokenizer.bos_token or "", "")
            else:
                text = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction="", response=messages[1]["content"])
            orig_s.append(text)
            if len(orig_s) >= num_examples:
                break
        random.shuffle(orig_s)
        print(f"UltraChat retain: {len(orig_s)}")

        # ── Retain: borderline (XSTest) ──────────────────────────────────────
        with open("data/xstest_v2_completions_gpt4_gpteval.csv", newline="") as f:
            xstest = [row for row in csv.DictReader(f) if row["final_label"] == "1_full_compliance"]
        borderline = []
        for d in xstest * 50:
            switch = np.random.choice([0, 1])
            if switch == 0:
                text = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction=d["prompt"], response=d["completion"])
            else:
                text = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction="", response=d["completion"])
            borderline.append(text)
        orig_s += borderline
        random.shuffle(orig_s)
        print(f"After borderline retain: {len(orig_s)}")

        # ── Retain: refusal outputs on harmful prompts ───────────────────────
        with open("data/circuit_breakers_train.json") as f:
            cb_data = json.load(f)
        random.shuffle(cb_data)
        refusal_retain = []
        for d in (cb_data[:2000]) * 2:
            switch = np.random.choice([0, 1])
            if switch == 0:
                text = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction=d["prompt"], response=d["llama3_output"])
            else:
                text = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction="", response=d["llama3_output"])
            refusal_retain.append(text)
        orig_s += refusal_retain
        random.shuffle(orig_s)
        print(f"After refusal retain: {len(orig_s)}")

        self.orig_s_retain = orig_s

        # ── Circuit Breaker: harmful completions ─────────────────────────────
        circuit_breaker_orig = []
        for d in tqdm(cb_data, desc="CB examples"):
            switch = np.random.choice([0, 1])
            if switch == 0:
                text = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction=d["prompt"], response=d["output"])
            else:
                text = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction="", response=d["output"])
            circuit_breaker_orig.append(text)
        random.shuffle(circuit_breaker_orig)
        self.circuit_breaker_orig = circuit_breaker_orig
        print(f"Circuit breaker examples: {len(self.circuit_breaker_orig)}")

        # ── Val ──────────────────────────────────────────────────────────────
        with open("data/circuit_breakers_val.json") as f:
            val_data = json.load(f)
        self.val_orig = [
            one_shot_template.format(
                user_tag=user_tag, assistant_tag=assistant_tag,
                instruction=d["prompt"], response=d["output"])
            for d in val_data
        ]

        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return min(len(self.orig_s_retain), len(self.circuit_breaker_orig))

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        cb_tokenized_kwargs = dict(max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        tokenize_kwargs = dict(max_length=1024, padding="max_length", truncation=True, return_tensors="pt")

        # Circuit breaker: split at <SEPARATOR> into [request, response]
        cb_request, cb_response = self.circuit_breaker_orig[i].split("<SEPARATOR>")
        self.tokenizer.padding_side = "left"
        tok_cb_req = self.tokenizer(cb_request, **cb_tokenized_kwargs)
        self.tokenizer.padding_side = "right"
        tok_cb_resp = self.tokenizer(cb_response, add_special_tokens=False, **cb_tokenized_kwargs)
        self.tokenizer.padding_side = "left"

        input_ids_cb = torch.cat([tok_cb_req["input_ids"], tok_cb_resp["input_ids"]], dim=1)
        attn_mask_cb = torch.cat([tok_cb_req["attention_mask"], tok_cb_resp["attention_mask"]], dim=1)

        # Retain
        tok_retain = self.tokenizer(
            self.orig_s_retain[i].replace("<SEPARATOR>", ""), **tokenize_kwargs)

        # Val
        val_text = self.val_orig[i % len(self.val_orig)].replace("<SEPARATOR>", "")
        tok_val = self.tokenizer(val_text, **tokenize_kwargs)

        return dict(
            input_ids_circuit_breaker=input_ids_cb,
            attention_mask_circuit_breaker=attn_mask_cb,
            input_ids=tok_retain["input_ids"],
            attention_mask=tok_retain["attention_mask"],
            input_ids_val=tok_val["input_ids"],
            attention_mask_val=tok_val["attention_mask"],
        )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(self, model, inputs, target_layers, alpha, return_outputs=False, **kwargs):
    self.current_training_step += 1
    log_now = self.current_training_step % 10 == 0

    retain_input_ids = inputs["input_ids"]
    retain_attention_mask = inputs["attention_mask"]
    cb_input_ids = inputs["input_ids_circuit_breaker"]
    cb_attention_mask = inputs["attention_mask_circuit_breaker"]
    val_input_ids = inputs["input_ids_val"]
    val_attention_mask = inputs["attention_mask_val"]

    retain_inputs = dict(input_ids=retain_input_ids, attention_mask=retain_attention_mask, output_hidden_states=True)
    cb_inputs = dict(input_ids=cb_input_ids, attention_mask=cb_attention_mask, output_hidden_states=True)
    val_inputs = dict(input_ids=val_input_ids, attention_mask=val_attention_mask, output_hidden_states=True)

    progress = self.get_training_progress()
    retain_coeff = alpha * progress
    cb_coeff = alpha * (1 - progress)
    if log_now:
        print(f"\nSTEP {self.current_training_step} | retain_coeff={retain_coeff:.3f} cb_coeff={cb_coeff:.3f}")

    layers_cb_mask = cb_attention_mask.repeat(len(target_layers), 1, 1).unsqueeze(-1)

    with model.disable_adapter():
        model.eval()
        with torch.no_grad():
            if retain_coeff > 0:
                orig_retain_hidden = torch.stack(model(**retain_inputs)["hidden_states"]).detach()
                layers_retain_mask = retain_attention_mask.repeat(orig_retain_hidden.shape[0], 1, 1).unsqueeze(-1)
                orig_retain_hidden *= layers_retain_mask

            if cb_coeff > 0:
                cb_out = model(**cb_inputs)["hidden_states"]
                cb_hidden = torch.stack([cb_out[l].detach() for l in target_layers])

            if log_now:
                val_out = model(**val_inputs)["hidden_states"]
                val_hidden = torch.stack([val_out[l] for l in target_layers])

    model.train()

    retain_loss = torch.tensor(0.0, device=retain_input_ids.device)
    cb_loss = torch.tensor(0.0, device=retain_input_ids.device)

    if retain_coeff > 0:
        lora_retain_hidden = torch.stack(model(**retain_inputs)["hidden_states"]) * layers_retain_mask
        retain_loss = torch.norm(lora_retain_hidden - orig_retain_hidden, dim=-1, p=2, dtype=torch.float).nanmean()
        if log_now:
            cos = cosine_similarity(lora_retain_hidden, orig_retain_hidden, dim=-1) * layers_retain_mask.squeeze(-1)
            print(f"retain_cos_sim: {(cos.sum() / layers_retain_mask.sum()).item():.4f}  retain_loss: {retain_loss.item():.4f}")

    if cb_coeff > 0:
        lora_cb_hidden = torch.stack([model(**cb_inputs)["hidden_states"][l] for l in target_layers])
        norm_lora = lora_cb_hidden / lora_cb_hidden.norm(dim=-1, keepdim=True, dtype=torch.float).clamp(min=1e-8)
        norm_orig = cb_hidden / cb_hidden.norm(dim=-1, keepdim=True, dtype=torch.float).clamp(min=1e-8)
        inner = (norm_lora * norm_orig) * layers_cb_mask
        cb_loss = torch.relu(inner.sum(dim=-1)).sum() / layers_cb_mask.sum()
        if log_now:
            cos = cosine_similarity(cb_hidden, lora_cb_hidden, dim=-1) * layers_cb_mask.squeeze(-1)
            print(f"cb_cos_sim: {(cos.sum() / layers_cb_mask.sum()).item():.4f}  cb_loss: {cb_loss.item():.4f}")

    if log_now:
        with torch.no_grad():
            lora_val_hidden = torch.stack([model(**val_inputs)["hidden_states"][l] for l in target_layers])
            layers_val_mask = val_attention_mask.repeat(len(target_layers), 1, 1).unsqueeze(-1)
            cos = cosine_similarity(val_hidden, lora_val_hidden, dim=-1) * layers_val_mask.squeeze(-1)
            print(f"val_cos_sim: {(cos.sum() / layers_val_mask.sum()).item():.4f}")

    loss = retain_coeff * retain_loss + cb_coeff * cb_loss
    return (loss,) if return_outputs else loss


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

def data_collator(batch_list: list[dict]) -> dict[str, torch.Tensor]:
    out: dict[str, list] = {}
    for item in batch_list:
        for k, v in item.items():
            out.setdefault(k, []).append(v)
    return {k: torch.cat(vs, dim=0) for k, vs in out.items()}


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_model(model, tokenizer, model_name_or_path: str, drop_layers_after: int | None, output_dir: str) -> None:
    import os
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving merged model to {output_dir}")
    merged = model.merge_and_unload()
    if drop_layers_after is not None:
        anchor = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=merged.dtype, device_map="auto")
        merged.model.layers = merged.model.layers + anchor.model.layers[drop_layers_after + 1:]
        merged.config = anchor.config
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train() -> None:
    MODEL = "HuggingFaceTB/SmolLM3-3B"
    OUTPUT_DIR = "/network/scratch/b/brownet/information-safety/models/smollm3-circuit-breaker"

    # SmolLM3 has 36 layers; proportional to Llama-3-8B (10,20/32) → 11,22
    TARGET_LAYERS = [11, 22]
    LORRA_ALPHA = 10
    LORA_R = 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    MAX_STEPS = 150
    BATCH_SIZE = 4
    GRAD_ACCUM = 4  # effective batch 16, same as Llama-3 script

    # Drop layers above max target during training (re-attached at save time)
    drop_layers_after = max(TARGET_LAYERS)

    config = AutoConfig.from_pretrained(MODEL)
    config.num_hidden_layers = drop_layers_after + 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_layers_to_transform = list(range(drop_layers_after + 1))
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        layers_to_transform=lora_layers_to_transform,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    train_dataset = CircuitBreakerDataset(tokenizer, num_examples=10000)

    training_args = transformers.TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=1e-4,
        lr_scheduler_type="constant",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_total_limit=0,
        remove_unused_columns=False,
        report_to="none",
    )

    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_training_steps = MAX_STEPS
            self.current_training_step = 0

        def get_training_progress(self) -> float:
            return self.state.global_step / 300

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            return compute_loss(
                self, model, inputs,
                target_layers=TARGET_LAYERS,
                alpha=LORRA_ALPHA,
                return_outputs=return_outputs,
            )

    trainer = CustomTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    model.config.use_cache = False
    trainer.train()
    save_model(model, tokenizer, MODEL, drop_layers_after, OUTPUT_DIR)


if __name__ == "__main__":
    train()
