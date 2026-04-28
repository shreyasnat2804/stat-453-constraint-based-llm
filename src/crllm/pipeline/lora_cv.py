"""5-fold LoRA cross-validation for RECAST.

Per fold: train a fresh LoRA adapter on 4 folds, evaluate CSR on the held-out
fold. Final score = mean CSR across folds, in [0, 1].

Validators are factored from src/crllm/cross_validation/cross_validation_kfold.py
(which is a Colab script, not safely importable). Training mirrors the
hyperparameters in src/crllm/training/lora_finetune/lora_base.ipynb.
"""

from __future__ import annotations

import gc
import json
import os
import random
import re
from pathlib import Path
from typing import Callable

PROMPT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

TRAIN_TEMPLATE = PROMPT_TEMPLATE + "{response}<|eot_id|>"


# ── Record loading ───────────────────────────────────────────────────────────


def load_records(path: str | Path) -> list[dict]:
    """Read a RECAST JSONL file. Returns dicts with instruction, response, raw."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instruction = row.get("winner_prompt", row.get("input", ""))
            response = row.get(
                "response_of_winner_prompt",
                row.get("winner_response", row.get("output", "")),
            )
            if instruction and response:
                out.append(
                    {"instruction": instruction, "gold_response": response, "raw": row}
                )
    return out


# ── Validators (factored from cross_validation_kfold.py) ─────────────────────


def _check_length_words(response: str, raw: dict) -> bool:
    try:
        constraints = raw.get("added_constraint", {}).get("Length", [])
        wc = len(response.split())
        for c in constraints:
            nums = re.findall(r"\d+", c)
            if len(nums) >= 2 and "word" in c.lower():
                if not (int(nums[0]) <= wc <= int(nums[1])):
                    return False
        return True
    except Exception:
        return True


def _check_length_sentences(response: str, raw: dict) -> bool:
    try:
        constraints = raw.get("added_constraint", {}).get("Length", [])
        sc = len(re.split(r"[.!?]+", response.strip()))
        for c in constraints:
            nums = re.findall(r"\d+", c)
            if nums and "sentence" in c.lower():
                if sc > int(nums[0]):
                    return False
        return True
    except Exception:
        return True


def _check_keyword(response: str, raw: dict) -> bool:
    try:
        constraints = raw.get("added_constraint", {}).get("Keyword", [])
        rl = response.lower()
        for c in constraints:
            m = re.search(
                r'["“”]([^"]+)["“”].*?(\d+)\s*times',
                c,
                re.IGNORECASE,
            )
            if m and rl.count(m.group(1).lower()) < int(m.group(2)):
                return False
        return True
    except Exception:
        return True


def _check_start_with(response: str, raw: dict) -> bool:
    try:
        constraints = raw.get("added_constraint", {}).get(
            "Strat_With", raw.get("added_constraint", {}).get("Start_With", [])
        )
        for c in constraints:
            m = re.search(r'["“”]([^"]+)["“”]', c)
            if m:
                words = response.strip().lower().split()
                if not words or words[0] != m.group(1).strip().lower():
                    return False
        return True
    except Exception:
        return True


def _check_end_with(response: str, raw: dict) -> bool:
    try:
        constraints = raw.get("added_constraint", {}).get("End_With", [])
        for c in constraints:
            m = re.search(r'["“”]([^"]+)["“”]', c)
            if m:
                words = re.findall(r"\w+", response.lower())
                if not words or words[-1] != m.group(1).strip().lower():
                    return False
        return True
    except Exception:
        return True


def _check_format(response: str, raw: dict) -> bool:
    try:
        constraints = raw.get("added_constraint", {}).get("Format", [])
        for c in constraints:
            if "<<" in c and not re.search(r"<<.+?>>", response):
                return False
        return True
    except Exception:
        return True


def _check_tone(response: str, raw: dict) -> bool:
    informal = ["gonna", "wanna", "gotta", "kinda", "dunno", "lol", "omg"]
    return not any(w in response.lower() for w in informal)


def _passthrough(response: str, raw: dict) -> bool:
    return True


VALIDATORS: dict[str, list[Callable[[str, dict], bool]]] = {
    "Length": [_check_length_words, _check_length_sentences],
    "Keyword": [_check_keyword],
    "Start_With": [_check_start_with],
    "End_With": [_check_end_with],
    "Format": [_check_format],
    "Tone": [_check_tone],
    "Style": [_passthrough],
    "Role_Playing": [_passthrough],
    "Numerical_Constraints": [_passthrough],
}


def score_response(response: str, raw: dict) -> dict:
    results, all_passed = {}, []
    for cat, fns in VALIDATORS.items():
        passed = all(fn(response, raw) for fn in fns)
        results[cat] = passed
        all_passed.append(passed)
    results["csr"] = sum(all_passed) / len(all_passed)
    return results


# ── LoRA train + eval (per fold) ─────────────────────────────────────────────


def _build_sft_dataset(records: list[dict]):
    from datasets import Dataset

    rows = [
        {
            "text": TRAIN_TEMPLATE.format(
                instruction=r["instruction"], response=r["gold_response"]
            )
        }
        for r in records
    ]
    return Dataset.from_list(rows)


def train_lora_on_fold(
    train_records: list[dict],
    output_dir: str | Path,
    base_model: str = "meta-llama/Llama-3.2-1B-Instruct",
    lora_rank: int = 8,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.05,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    learning_rate: float = 1e-4,
    num_epochs: int = 1,
    per_device_batch_size: int = 8,
    grad_accumulation_steps: int = 4,
    max_seq_length: int = 512,
    seed: int = 42,
) -> str:
    """Train a LoRA adapter on the given records. Returns the adapter path."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if lora_alpha is None:
        lora_alpha = lora_rank * 2

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=list(target_modules),
            lora_dropout=lora_dropout,
            bias="none",
            inference_mode=False,
        ),
    )
    model.print_trainable_parameters()

    dataset = _build_sft_dataset(train_records)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,
        optim="adamw_torch",
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    adapter_path = output_dir / "lora_adapter"
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    del trainer, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(adapter_path)


def _generate(model, tokenizer, instruction: str, max_new_tokens: int = 512) -> str:
    import torch

    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def evaluate_adapter_on_fold(
    adapter_path: str,
    eval_records: list[dict],
    base_model: str = "meta-llama/Llama-3.2-1B-Instruct",
    max_new_tokens: int = 512,
    progress_every: int = 25,
) -> list[dict]:
    """Run inference + score on each held-out record. Returns per-record results."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    results = []
    for i, ex in enumerate(eval_records):
        if i % progress_every == 0:
            print(f"    eval {i}/{len(eval_records)}")
        response = _generate(model, tokenizer, ex["instruction"], max_new_tokens)
        scores = score_response(response, ex["raw"])
        results.append({"response": response, **scores})

    del model, base, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


# ── Orchestrator ─────────────────────────────────────────────────────────────


def run_kfold(
    records: list[dict],
    output_root: str | Path,
    k: int = 5,
    eval_per_fold: int = 200,
    seed: int = 42,
    train_kwargs: dict | None = None,
    eval_kwargs: dict | None = None,
) -> dict:
    """Run K-fold LoRA CV.

    For each fold: train on the K-1 other folds, evaluate on a sample of size
    `eval_per_fold` from the held-out fold. Final score is the mean per-record
    CSR across folds, ∈ [0, 1].
    """
    from sklearn.model_selection import KFold

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    train_kwargs = dict(train_kwargs or {})
    eval_kwargs = dict(eval_kwargs or {})

    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    kf = KFold(n_splits=k, shuffle=False)
    fold_scores: list[dict] = []
    all_results: list[dict] = []

    for fold_num, (train_idx, test_idx) in enumerate(kf.split(indices), start=1):
        print(f"\n{'='*60}\n  FOLD {fold_num}/{k}\n{'='*60}")

        train_recs = [records[indices[i]] for i in train_idx]
        eval_pool = [records[indices[i]] for i in test_idx]
        if eval_per_fold and len(eval_pool) > eval_per_fold:
            eval_recs = rng.sample(eval_pool, eval_per_fold)
        else:
            eval_recs = eval_pool

        print(f"  train={len(train_recs)}  eval={len(eval_recs)}")

        fold_dir = output_root / f"fold_{fold_num}"
        adapter_path = train_lora_on_fold(
            train_recs, output_dir=fold_dir, seed=seed, **train_kwargs
        )
        per_record = evaluate_adapter_on_fold(
            adapter_path, eval_recs, **eval_kwargs
        )

        fold_csr = sum(r["csr"] for r in per_record) / max(len(per_record), 1)
        print(f"  fold {fold_num} mean CSR = {fold_csr:.4f}")

        fold_scores.append({"fold": fold_num, "csr": fold_csr, "n": len(per_record)})
        for r in per_record:
            all_results.append({"fold": fold_num, **r})

        with open(fold_dir / "results.json", "w") as f:
            json.dump(per_record, f, indent=2)

    final_score = sum(f["csr"] for f in fold_scores) / max(len(fold_scores), 1)
    summary = {
        "k": k,
        "eval_per_fold": eval_per_fold,
        "fold_scores": fold_scores,
        "final_score": final_score,
    }
    with open(output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_root / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}\n  FINAL K-FOLD CSR = {final_score:.4f}\n{'='*60}")
    return summary
