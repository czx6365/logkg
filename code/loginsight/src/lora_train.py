from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

from .prompting import build_prompt
from .utils import load_jsonl, set_seed



def _guess_lora_target_modules(model: torch.nn.Module) -> List[str]:
    candidate_order = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
    ]
    module_suffixes = {name.split(".")[-1] for name, _ in model.named_modules()}
    # 不同底模的注意力/MLP 层命名不一样，这里做一次轻量自动探测。
    targets = [m for m in candidate_order if m in module_suffixes]
    return targets if targets else ["c_attn"]



def _tokenize_records(records: List[Dict[str, Any]], tokenizer: Any, max_len: int) -> Dataset:
    texts = [build_prompt(r["instruction"], r["input"], r["output"]) for r in records]
    ds = Dataset.from_dict({"text": texts})

    def _tok(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        # 监督微调里 labels 直接复制 input_ids，等价于做标准 causal LM 训练。
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        out["labels"] = [x[:] for x in out["input_ids"]]
        return out

    return ds.map(_tok, batched=True, remove_columns=["text"])



def train_lora(model_cfg: Dict[str, Any], base_dir: Path) -> Path:
    """Train a LoRA adapter from instruction JSONL files."""
    seed = int(model_cfg.get("seed", 42))
    set_seed(seed)

    paths_cfg = model_cfg["paths"]
    train_path = (base_dir / paths_cfg["instruction_train"]).resolve()
    val_path = (base_dir / paths_cfg["instruction_val"]).resolve()
    ckpt_dir = (base_dir / paths_cfg["checkpoint_dir"]).resolve()

    model_section = model_cfg["model"]
    train_section = model_cfg["training"]

    base_model_name = str(model_section["base_model_name"])
    max_token_length = int(model_section.get("max_token_length", 4096))

    train_records = load_jsonl(train_path)
    if val_path.exists():
        val_records = load_jsonl(val_path)
    else:
        # 如果用户只提供 train 集，这里自动切一部分做验证。
        val_ratio = float(train_section.get("train_val_split", 0.1))
        labels = [str(r["fault_type"]) for r in train_records]
        train_records, val_records = train_test_split(
            train_records,
            test_size=val_ratio,
            random_state=seed,
            stratify=labels if len(set(labels)) > 1 else None,
        )

    assert train_records, "Empty train records"
    assert val_records, "Empty val records"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # 有 GPU 时让 transformers 自动分配设备；无 GPU 则退回 CPU。
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if bool(train_section.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(train_section.get("lora_r", 8)),
        lora_alpha=int(train_section.get("lora_alpha", 32)),
        lora_dropout=float(train_section.get("lora_dropout", 0.05)),
        target_modules=_guess_lora_target_modules(model),
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    train_ds = _tokenize_records(train_records, tokenizer, max_token_length)
    val_ds = _tokenize_records(val_records, tokenizer, max_token_length)

    fp16_cfg = str(train_section.get("fp16", "auto")).lower()
    bf16_cfg = str(train_section.get("bf16", "auto")).lower()
    fp16 = torch.cuda.is_available() if fp16_cfg == "auto" else (fp16_cfg == "true")
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() if bf16_cfg == "auto" else (bf16_cfg == "true")

    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        learning_rate=float(train_section.get("learning_rate", 1e-4)),
        weight_decay=float(train_section.get("weight_decay", 0.1)),
        per_device_train_batch_size=int(train_section.get("batch_size", 16)),
        per_device_eval_batch_size=int(train_section.get("batch_size", 16)),
        gradient_accumulation_steps=int(train_section.get("gradient_accumulation_steps", 1)),
        num_train_epochs=float(train_section.get("num_train_epochs", 1)),
        logging_steps=int(train_section.get("logging_steps", 10)),
        save_steps=int(train_section.get("save_steps", 100)),
        save_total_limit=int(train_section.get("save_total_limit", 2)),
        evaluation_strategy="steps",
        eval_steps=int(train_section.get("save_steps", 100)),
        fp16=fp16,
        bf16=bf16,
        report_to=[],
    )

    # 这里保存的是 LoRA adapter，而不是完整底模权重。
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.train()

    final_dir = ckpt_dir / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return final_dir
