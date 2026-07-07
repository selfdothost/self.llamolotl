"""
Direct training script using HuggingFace Trainer + DeepSpeed + PEFT.
Replaces axolotl.cli.train with a minimal, direct implementation.

Usage:
    accelerate launch train.py config.yaml
"""

import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ─── Logging ────────────────────────────────────────────────────────────

DEBUG = os.environ.get("DEBUG", "0") == "1"
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG if DEBUG else logging.INFO)
# Only configure root logger at INFO to avoid flooding from httpcore/httpx/filelock
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)


# ─── Config parsing ─────────────────────────────────────────────────────

# Maps config field names (axolotl-compatible) to TrainingArguments field names
_FIELD_MAP = {
    "micro_batch_size": "per_device_train_batch_size",
    "num_epochs": "num_train_epochs",
    "lr_scheduler": "lr_scheduler_type",
    "eval_batch_size": "per_device_eval_batch_size",
}


def load_config(config_path: str) -> dict:
    """Load and normalize a YAML training config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply field name mappings
    for old_key, new_key in _FIELD_MAP.items():
        if old_key in config and new_key not in config:
            config[new_key] = config.pop(old_key)

    return config


# ─── Dataset formatting ─────────────────────────────────────────────────

def _format_chat_template(example, field_messages, tokenizer):
    """Format a chat_template dataset row into a text string."""
    messages = example[field_messages]
    # apply_chat_template returns a string with the model's native format
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def _convert_sharegpt_to_messages(example, field):
    """Convert ShareGPT format to standard messages format."""
    conversations = example[field]
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = []
    for turn in conversations:
        role = role_map.get(turn.get("from", ""), turn.get("from", "user"))
        content = turn.get("value", turn.get("content", ""))
        messages.append({"role": role, "content": content})
    return {"messages": messages}


def _format_alpaca(example, tokenizer):
    """Format an alpaca-style dataset row into a text string."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def _find_text_column(dataset) -> str:
    """Find the first text-like column in a dataset for completion fallback."""
    for col in dataset.column_names:
        if col in ("text", "content", "document", "passage"):
            return col
    # Fall back to first string column
    for col in dataset.column_names:
        if dataset.features[col].dtype == "string":
            return col
    # Last resort: first column
    return dataset.column_names[0]


def load_and_format_datasets(config: dict, tokenizer) -> dict:
    """Load datasets from config and format them for SFTTrainer.

    Returns dict with 'train' and optionally 'eval' splits,
    plus 'text_field' (column name) or 'formatting_func' (callable).
    """
    dataset_configs = config.get("datasets", [])
    if not dataset_configs:
        raise ValueError("No datasets specified in config")

    all_train = []
    all_eval = []
    val_set_size = config.get("val_set_size", 0.0)
    use_formatting_func = False
    text_field = "text"

    for ds_cfg in dataset_configs:
        ds_path = ds_cfg["path"]
        ds_type = ds_cfg.get("type", "completion")

        log.info(f"Loading dataset: {ds_path} (format: {ds_type})")

        # A local file on disk (e.g. curator output uploaded via the API) is
        # loaded as json; anything else is treated as a HuggingFace dataset id.
        try:
            if os.path.exists(ds_path):
                log.info(f"  Loading local dataset file via json loader: {ds_path}")
                ds = load_dataset("json", data_files=ds_path, split="train")
            else:
                ds = load_dataset(ds_path, split="train")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{ds_path}': {e}") from e

        if DEBUG:
            log.debug(f"  Rows: {len(ds)}, Columns: {ds.column_names}")

        # Format based on type
        if ds_type == "chat_template":
            field_messages = ds_cfg.get("field_messages", "messages")
            ds = ds.map(
                lambda ex: _format_chat_template(ex, field_messages, tokenizer),
                remove_columns=ds.column_names,
                desc="Formatting chat_template",
            )
            text_field = "text"

        elif ds_type == "sharegpt":
            field = ds_cfg.get("field", "conversations")
            # Step 1: convert to standard messages
            ds = ds.map(
                lambda ex: _convert_sharegpt_to_messages(ex, field),
                remove_columns=ds.column_names,
                desc="Converting ShareGPT",
            )
            # Step 2: apply chat template
            ds = ds.map(
                lambda ex: _format_chat_template(ex, "messages", tokenizer),
                remove_columns=ds.column_names,
                desc="Formatting ShareGPT via chat_template",
            )
            text_field = "text"

        elif ds_type == "alpaca":
            ds = ds.map(
                lambda ex: _format_alpaca(ex, tokenizer),
                remove_columns=ds.column_names,
                desc="Formatting alpaca",
            )
            text_field = "text"

        elif ds_type == "completion":
            # Use text field directly
            col = ds_cfg.get("field", None)
            if col and col in ds.column_names:
                if col != "text":
                    ds = ds.rename_column(col, "text")
            elif "text" not in ds.column_names:
                col = _find_text_column(ds)
                log.warning(
                    f"Completion format: using column '{col}' as text field"
                )
                ds = ds.rename_column(col, "text")
            text_field = "text"

        else:
            # Unknown format — fall back to completion
            col = _find_text_column(ds)
            log.warning(
                f"Unknown dataset format '{ds_type}', falling back to "
                f"completion mode using column '{col}'"
            )
            if col != "text":
                ds = ds.rename_column(col, "text")
            text_field = "text"

        # Split train/eval
        if val_set_size > 0:
            split = ds.train_test_split(test_size=val_set_size, seed=42)
            all_train.append(split["train"])
            all_eval.append(split["test"])
        else:
            all_train.append(ds)

    # Concatenate all datasets
    train_dataset = concatenate_datasets(all_train) if len(all_train) > 1 else all_train[0]
    eval_dataset = None
    if all_eval:
        eval_dataset = concatenate_datasets(all_eval) if len(all_eval) > 1 else all_eval[0]

    log.info(f"Total training samples: {len(train_dataset)}")
    if eval_dataset:
        log.info(f"Total eval samples: {len(eval_dataset)}")

    if DEBUG:
        # Log token distribution sample
        sample = train_dataset[0]
        sample_tokens = tokenizer(sample["text"], return_length=True)
        log.debug(f"  Sample text length: {len(sample['text'])} chars, {sample_tokens['length'][0]} tokens")

    return {
        "train": train_dataset,
        "eval": eval_dataset,
        "text_field": text_field,
    }


# ─── Model loading ──────────────────────────────────────────────────────

def load_model_and_tokenizer(config: dict):
    """Load the base model and tokenizer with optional quantization."""
    base_model = config["base_model"]
    log.info(f"Loading model: {base_model}")

    # Detect DeepSpeed stage early — needed for quantization and loading decisions
    ds_path = config.get("deepspeed")
    is_zero3 = False
    ds_cfg = None
    if ds_path:
        try:
            with open(ds_path) as f:
                ds_cfg = json.load(f)
            is_zero3 = ds_cfg.get("zero_optimization", {}).get("stage", 0) == 3
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Failed to parse DeepSpeed config %s: %s. Assuming not ZeRO-3.", ds_path, e)

    # Quantization config for QLoRA
    # bnb model quantization works with ZeRO-2 (model stays on GPU) but not
    # ZeRO-3 (which moves parameters between CPU/GPU, breaking bnb's layout).
    bnb_config = None
    if is_zero3 and (config.get("load_in_4bit") or config.get("load_in_8bit")):
        log.warning("Skipping bitsandbytes quantization (load_in_4bit/8bit) — "
                     "incompatible with DeepSpeed ZeRO-3. Using ZeRO-3 "
                     "parameter offloading instead.")
    elif config.get("load_in_4bit"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(
                torch, config.get("bnb_4bit_compute_dtype", "bfloat16")
            ),
            bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
        )
    elif config.get("load_in_8bit"):
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Determine torch dtype
    if config.get("bf16") in (True, "auto"):
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    # Attention implementation
    attn_impl = None
    if config.get("flash_attention"):
        attn_impl = "flash_attention_2"
    elif config.get("attn_implementation"):
        attn_impl = config["attn_implementation"]

    model_kwargs = {
        "dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    if is_zero3:
        # Load on CPU — ZeRO-3 partitioning is handled later by
        # Trainer/accelerate during deepspeed.initialize().
        # low_cpu_mem_usage avoids allocating a second full copy in RAM.
        # We do NOT use deepspeed.zero.Init() or HfDeepSpeedConfig here
        # because meta-device init breaks non-persistent buffers like
        # rotary embedding inv_freq (not in the checkpoint, computed in __init__).
        model_kwargs["low_cpu_mem_usage"] = True
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    elif not ds_path:
        # No DeepSpeed — place directly on GPU
        model_kwargs["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    else:
        # ZeRO-1/2 — place on GPU, DS only manages optimizer/gradients
        model_kwargs["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if DEBUG:
        param_count = sum(p.numel() for p in model.parameters())
        log.debug(f"  Model parameters: {param_count:,}")
        if torch.cuda.is_available():
            log.debug(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    # Handle special tokens from config
    special_tokens = config.get("special_tokens", {})
    if special_tokens.get("pad_token"):
        tokenizer.pad_token = special_tokens["pad_token"]

    return model, tokenizer


# ─── PEFT / LoRA setup ──────────────────────────────────────────────────

def setup_peft(model, config: dict):
    """Wrap model with PEFT LoRA if configured. Returns (model, peft_config)."""
    adapter = config.get("adapter")
    if adapter != "lora":
        return model, None

    log.info("Configuring LoRA adapter")

    # Prepare for quantized training when bnb quantization is actually applied.
    # bnb quantization is used with ZeRO-2 and without DeepSpeed, but not ZeRO-3.
    if config.get("load_in_4bit") or config.get("load_in_8bit"):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.get("gradient_checkpointing", True),
        )

    # Target modules
    target_modules = config.get("lora_target_modules")
    if config.get("lora_target_linear", False) and not target_modules:
        target_modules = "all-linear"

    peft_config = LoraConfig(
        r=config.get("lora_r", 32),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    if DEBUG:
        log.debug(f"  LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}, "
                   f"targets={peft_config.target_modules}")

    return model, peft_config


# ─── Training arguments ─────────────────────────────────────────────────

def _deepspeed_has_optimizer(config: dict) -> bool:
    """Check if the DeepSpeed config defines its own optimizer."""
    ds_path = config.get("deepspeed")
    if not ds_path:
        return False
    try:
        with open(ds_path) as f:
            ds_config = json.load(f)
        return "optimizer" in ds_config
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Failed to parse DeepSpeed config %s for optimizer check: %s", ds_path, e)
        return False


def build_training_args(config: dict, train_size: int) -> SFTConfig:
    """Build SFTConfig (extends TrainingArguments) from the config dict."""

    # Calculate eval/save steps from per-epoch counts
    batch_size = config.get("per_device_train_batch_size", 1)
    grad_accum = config.get("gradient_accumulation_steps", 8)
    effective_batch = batch_size * grad_accum
    steps_per_epoch = max(1, train_size // effective_batch)

    evals_per_epoch = config.get("evals_per_epoch", 4)
    saves_per_epoch = config.get("saves_per_epoch", 1)
    eval_steps = max(1, steps_per_epoch // evals_per_epoch) if evals_per_epoch > 0 else None
    save_steps = max(1, steps_per_epoch // saves_per_epoch) if saves_per_epoch > 0 else None

    log.info(f"Steps per epoch: {steps_per_epoch}, eval every {eval_steps} steps, save every {save_steps} steps")

    args = SFTConfig(
        output_dir=config.get("output_dir", "./outputs/default"),
        # Batch
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        # Epochs / steps
        num_train_epochs=config.get("num_train_epochs", 4),
        max_steps=config.get("max_steps", -1),
        # Optimizer — when DeepSpeed config defines its own optimizer (e.g. for
        # CPU offloading compatibility), defer to it instead of the user's choice
        # which may be incompatible (e.g. adamw_bnb_8bit can't run on CPU).
        optim="adamw_torch" if _deepspeed_has_optimizer(config) else config.get("optimizer", "adamw_torch"),
        learning_rate=config.get("learning_rate", 2e-4),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        warmup_steps=config.get("warmup_steps", 0),
        weight_decay=config.get("weight_decay", 0.0),
        # Precision
        bf16=config.get("bf16", True) if config.get("bf16") != "auto" else True,
        tf32=config.get("tf32", False),
        # Checkpointing
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.get("gradient_checkpointing", True) else None,
        # Eval / save
        eval_strategy="steps" if eval_steps else "no",
        eval_steps=eval_steps,
        save_strategy="steps" if save_steps else "epoch",
        save_steps=save_steps or steps_per_epoch,
        save_total_limit=config.get("save_total_limit", 3),
        # Logging
        logging_steps=config.get("logging_steps", 1),
        report_to="none",
        # SFTTrainer-specific
        max_length=config.get("sequence_len", 4096),
        packing=config.get("sample_packing", False),
        dataset_text_field="text",
        # DeepSpeed
        deepspeed=config.get("deepspeed"),
        # Misc
        auto_find_batch_size=True,
        remove_unused_columns=True,
        dataloader_pin_memory=True,
        seed=config.get("seed", 42),
    )

    return args


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        log.error("Usage: %s <config.yaml>", sys.argv[0])
        sys.exit(1)

    config_path = sys.argv[1]
    log.info(f"Loading config: {config_path}")
    config = load_config(config_path)

    if DEBUG:
        log.debug(f"Config:\n{json.dumps({k: str(v) for k, v in config.items()}, indent=2)}")

    # 1. Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # 2. Setup PEFT/LoRA
    model, peft_config = setup_peft(model, config)

    # 3. Load and format datasets
    data = load_and_format_datasets(config, tokenizer)

    # 4. Build training arguments
    training_args = build_training_args(config, len(data["train"]))

    if DEBUG and torch.cuda.is_available():
        log.debug(f"  GPU memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        log.debug(f"  GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # 5. Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["eval"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 6. Train
    # ZeRO-3 manages nn.Parameter objects but not buffers. Non-persistent
    # buffers (e.g. rotary embedding inv_freq) can end up on CPU after
    # DeepSpeed initialization. Move them to GPU right before training.
    if config.get("deepspeed") and torch.cuda.is_available():
        for name, buf in trainer.model.named_buffers():
            if buf is not None and buf.device.type == "cpu":
                buf.data = buf.data.to(trainer.args.device)
                log.debug(f"  Moved buffer {name} to {trainer.args.device}")

    # Checkpoint resume: auto-detect latest checkpoint or use explicit path
    resume_from = config.get("resume_from_checkpoint")
    if resume_from is True or resume_from == "auto":
        # Auto-detect: find latest checkpoint-* dir in output_dir
        output_dir = Path(training_args.output_dir)
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        if checkpoints:
            resume_from = str(checkpoints[-1])
            log.info(f"Auto-detected checkpoint to resume from: {resume_from}")
        else:
            log.info("No checkpoints found in output_dir, starting from scratch")
            resume_from = None
    elif resume_from and isinstance(resume_from, str) and resume_from not in ("false", "False"):
        log.info(f"Resuming from explicit checkpoint: {resume_from}")
    else:
        resume_from = None

    log.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_from)

    # 7. Save
    log.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    # 8. Log final metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if data["eval"] is not None:
        log.info("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    log.info("Training complete.")

    if DEBUG and torch.cuda.is_available():
        log.debug(f"  Final GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        log.debug(f"  Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
