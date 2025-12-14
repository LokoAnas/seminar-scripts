import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer


# ---------- Config dataclass ----------

@dataclass
class GemmaQLoRAConfig:
    model_name: str = "google/gemma-2-2b"
    dataset_name: str = "databricks/databricks-dolly-15k"
    dataset_split: str = "train"

    # Training / compute
    output_dir: str = "./gemma-2-2b-qlora-tech"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: Optional[float] = None
    max_steps: int = 100  # As requested, 50–100 to demonstrate pipeline
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 50
    save_total_limit: int = 2
    bf16: bool = torch.cuda.is_available()  # prefer bf16 if supported
    max_seq_length: int = 1024

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "v_proj")

    # 4-bit quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"  # or "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


def get_bnb_config(cfg: GemmaQLoRAConfig) -> BitsAndBytesConfig:
    # Quantized loading (QLoRA)
    compute_dtype = torch.bfloat16 if cfg.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.use_4bit,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    return bnb_config


def get_lora_config(cfg: GemmaQLoRAConfig) -> LoraConfig:
    """
    We target the attention projection matrices (`q_proj` and `v_proj`) because:
    - They control how input tokens are mapped into query/key/value spaces.
    - Small low-rank updates here can strongly influence *what* the model attends to
      and *how* it combines information, which is crucial for adapting to a new domain.
    - Updating only these layers keeps parameter count small while still having high impact
      on the model's behavior, per the LoRA paper.
    """
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        task_type="CAUSAL_LM",
        bias="none",
    )


def load_technical_dataset(cfg: GemmaQLoRAConfig):
    """
    Example: load databricks-dolly-15k and filter for 'programming' / 'technology'
    categories to simulate a 'Ship's Technical Manual' dataset.
    You can replace this with your own starship dataset.
    """
    raw = load_dataset(cfg.dataset_name, split=cfg.dataset_split)

    # Dolly has fields: instruction, context, response, category, etc.
    tech_categories = {"programming", "reasoning", "information_extraction", "math", "roleplay"}
    def is_technical(example):
        return example.get("category") in tech_categories

    tech = raw.filter(is_technical)
    return tech


def format_example(example: Dict[str, Any]) -> str:
    """
    Simple prompt formatting: include instruction and context, expect the response.
    This is what SFTTrainer will try to model as next-token prediction.
    """
    instruction = example.get("instruction", "")
    context = example.get("context", "")
    response = example.get("response", "")

    parts = []
    if instruction:
        parts.append(f"### Instruction:\n{instruction}")
    if context:
        parts.append(f"\n### Context:\n{context}")
    if response:
        parts.append(f"\n### Response:\n{response}")

    return "\n".join(parts)


def main():
    cfg = GemmaQLoRAConfig()

    # Ensure we’ve accepted the Gemma license on Hugging Face before running.
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=True,
    )
    # Gemma expects a specific pad token handling; often we can re-use eos as pad.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---------- Model with 4-bit quantization ----------
    bnb_config = get_bnb_config(cfg)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,  # required for some gated models
    )

    # ---------- LoRA / PEFT config ----------
    peft_config = get_lora_config(cfg)

    # ---------- Dataset ----------
    dataset = load_technical_dataset(cfg)

    # SFTTrainer will call tokenizer on the 'text' field by default, so we map it.
    dataset = dataset.map(
        lambda ex: {"text": format_example(ex)},
        remove_columns=dataset.column_names,
    )

    # ---------- Training arguments ----------
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        fp16=not cfg.bf16,
        optim="paged_adamw_8bit",  # good with QLoRA
        lr_scheduler_type="cosine",
        report_to="none",  # set to "wandb" or "tensorboard" if desired
    )

    # ---------- SFT Trainer ----------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=cfg.max_seq_length,
        args=training_args,
        packing=True,  # pack multiple examples into one sequence for efficiency
    )

    # ---------- Train ----------
    trainer.train()

    # ---------- Save adapter ----------
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print("QLoRA fine-tuning completed. Adapter saved to:", cfg.output_dir)


if __name__ == "__main__":
    main()
