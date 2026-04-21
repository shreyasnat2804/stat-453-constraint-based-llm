"""
Full fine-tuning script for RECAST constraint-following experiments.

Usage:
    python train_full_finetune.py [options]

Defaults to Llama-3.2-1B-Instruct trained on the local RECAST-30K dataset.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

from datasets import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Default paths ────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATASET = str(SCRIPT_DIR / "datasets" / "RECAST-30K.json")
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


# ── Dataset helpers ──────────────────────────────────────────────────────────


def _infer_difficulty(num_constraints: int) -> str:
    if num_constraints <= 2:
        return "L1"
    elif num_constraints <= 4:
        return "L2"
    elif num_constraints <= 7:
        return "L3"
    return "L4"


def _parse_record(record: dict, idx: int) -> dict:
    prompt = (
        record.get("winner_prompt")
        or record.get("prompt")
        or record.get("instruction")
        or ""
    )
    response = (
        record.get("winner_response")
        or record.get("response")
        or record.get("output")
        or ""
    )
    constraints = record.get("constraints", record.get("constraint_list")) or []
    if isinstance(constraints, str):
        try:
            constraints = json.loads(constraints)
        except json.JSONDecodeError:
            constraints = [{"type": "raw", "description": constraints}]
    if not isinstance(constraints, list):
        constraints = [constraints] if constraints else []

    difficulty = (
        record.get("difficulty_level")
        or record.get("difficulty")
        or record.get("level")
        or _infer_difficulty(len(constraints))
    )
    difficulty = str(difficulty)
    if not difficulty.startswith("L"):
        difficulty = f"L{difficulty}"

    return {
        "id": str(record.get("id", idx)),
        "prompt": prompt,
        "response": response,
        "difficulty_level": difficulty,
    }


def load_recast_dataset(dataset_path: str) -> list[dict]:
    """Load RECAST-30K from a local JSON/JSONL file or a HuggingFace dataset ID."""
    path = Path(dataset_path)

    # Try as a local file first
    if path.exists():
        # Detect Git LFS pointer files (they are tiny and start with "version https://")
        if path.stat().st_size < 1024:
            first_bytes = path.read_bytes()
            if b"git-lfs" in first_bytes:
                raise RuntimeError(
                    f"{path} is a Git LFS pointer. "
                    "Run `git lfs pull` to download the actual dataset file before training."
                )

        logger.info(f"Loading dataset from local file: {path}")
        raw = []
        if path.suffix == ".jsonl":
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw.append(json.loads(line))
        else:
            with open(path) as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                raw = loaded
            elif isinstance(loaded, dict):
                for key in ("data", "instances", "examples", "records"):
                    if key in loaded and isinstance(loaded[key], list):
                        raw = loaded[key]
                        break
                if not raw:
                    raw = [loaded]
    else:
        # Treat as a HuggingFace dataset ID
        logger.info(f"Loading dataset from HuggingFace Hub: {dataset_path}")
        from datasets import load_dataset as hf_load

        hf_ds = hf_load(dataset_path, split="train")
        raw = [dict(row) for row in hf_ds]

    records = [_parse_record(r, i) for i, r in enumerate(raw)]
    logger.info(f"Loaded {len(records)} records")
    return records


# ── Tokenisation ─────────────────────────────────────────────────────────────


def build_tokenised_dataset(
    records: list[dict],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    """
    Format each record as a user/assistant chat turn, tokenise it, and mask
    the prompt tokens in the labels so loss is computed only on the response.
    """
    has_template = (
        hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    )

    def tokenise(record):
        prompt_text = record["prompt"]
        response_text = record["response"]

        if has_template:
            # Build the full conversation and the prompt-only prefix
            full_chat = [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": response_text},
            ]
            prompt_chat = [{"role": "user", "content": prompt_text}]

            full_text = tokenizer.apply_chat_template(
                full_chat, tokenize=False, add_generation_prompt=False
            )
            prompt_text_formatted = tokenizer.apply_chat_template(
                prompt_chat, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text_formatted = f"User: {prompt_text}\nAssistant:"
            full_text = f"{prompt_text_formatted} {response_text}"

        full_enc = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        prompt_enc = tokenizer(
            prompt_text_formatted,
            max_length=max_length,
            truncation=True,
            padding=False,
        )

        input_ids = full_enc["input_ids"]
        labels = list(input_ids)
        prompt_len = len(prompt_enc["input_ids"])
        # Mask prompt tokens with -100 so Trainer ignores them in the loss
        labels[:prompt_len] = [-100] * prompt_len

        # If the entire sequence was masked (prompt >= max_length), signal
        # the caller to drop this example by returning None sentinel values.
        if all(l == -100 for l in labels):
            return {"input_ids": None, "attention_mask": None, "labels": None}

        return {
            "input_ids": input_ids,
            "attention_mask": full_enc["attention_mask"],
            "labels": labels,
        }

    logger.info("Tokenising dataset …")
    hf_ds = Dataset.from_list(records)
    tokenised = hf_ds.map(
        tokenise,
        remove_columns=hf_ds.column_names,
        desc="Tokenising",
        num_proc=1,
    )
    before = len(tokenised)
    tokenised = tokenised.filter(lambda x: x["input_ids"] is not None)
    dropped = before - len(tokenised)
    if dropped:
        logger.warning(
            f"Dropped {dropped} examples where prompt exceeded max_length={max_length}"
        )
    return tokenised


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full fine-tuning on RECAST constraint-following data."
    )
    p.add_argument(
        "--model_name",
        default=DEFAULT_MODEL,
        help="HuggingFace model ID or local path (default: %(default)s)",
    )
    p.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Local JSON/JSONL file or HF dataset ID (default: datasets/RECAST-30K.json)",
    )
    p.add_argument(
        "--output_dir", default="./output/finetuned", help="Where to save the model"
    )
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Steps to accumulate before a weight update (effective batch = device_bs × accum × n_gpus)",
    )
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument(
        "--max_length", type=int, default=512, help="Max sequence length in tokens"
    )
    p.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Fraction of data held out for validation",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace token for gated models",
    )
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing to reduce VRAM usage",
    )
    p.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Use only this many training samples (0 = all)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.hf_token:
        from huggingface_hub import login

        login(token=args.hf_token)
        logger.info("Logged in to HuggingFace Hub.")

    # ── Load & optionally subsample data ────────────────────────────────────
    records = load_recast_dataset(args.dataset)
    if args.num_samples > 0:
        random.shuffle(records)
        records = records[: args.num_samples]
        logger.info(f"Subsampled to {len(records)} records")

    # Train / val split
    random.shuffle(records)
    n_val = max(1, int(len(records) * args.val_split))
    val_records, train_records = records[:n_val], records[n_val:]
    logger.info(f"Train: {len(train_records)}  Val: {len(val_records)}")

    # ── Load tokenizer ───────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Tokenise ─────────────────────────────────────────────────────────────
    train_ds = build_tokenised_dataset(train_records, tokenizer, args.max_length)
    val_ds = build_tokenised_dataset(val_records, tokenizer, args.max_length)

    # ── Load model ───────────────────────────────────────────────────────────
    logger.info(f"Loading model: {args.model_name}")
    # device_map="auto" is incompatible with multi-GPU Trainer (DDP).
    # Use it only for single-GPU; for multi-GPU let Trainer/accelerate handle placement.
    n_gpus = torch.cuda.device_count()
    device_map = "auto" if n_gpus <= 1 else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Required when gradient checkpointing is active
        model.enable_input_require_grads()
        logger.info("Gradient checkpointing enabled.")

    # Full fine-tuning: all parameters are trainable
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)"
    )

    # ── Training arguments ───────────────────────────────────────────────────
    safe_model_name = args.model_name.replace("/", "_")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["labels"],
        run_name=f"full_ft_{safe_model_name}",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    logger.info("Starting full fine-tuning …")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Write a small metadata file
    meta = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "num_train_epochs": args.num_train_epochs,
        "effective_batch_size": (
            args.per_device_train_batch_size * args.gradient_accumulation_steps
        ),
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
    }
    with open(os.path.join(args.output_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Done.")


if __name__ == "__main__":
    main()
