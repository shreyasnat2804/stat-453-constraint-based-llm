"""
Full fine-tuning script for RECAST constraint-following experiments.

Usage:
    python train_full_finetune.py [options]

Defaults to Llama-3.2-1B-Instruct trained on the local RECAST-30K dataset.
Uses K-Fold cross-validation (default K=5) to report robust eval metrics.
Saves an HTML report locally and optionally to Google Drive.

Constraint-aware loss:
    total_loss = ce_loss + lambda * constraint_loss
where constraint_loss = 1 - (satisfied / total) on the gold responses.
"""

import argparse
import base64
import gc
import io
import json
import logging
import os
import random
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.model_selection import KFold
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SCRIPT_DIR      = Path(__file__).parent
DEFAULT_DATASET = str(SCRIPT_DIR / "datasets" / "RECAST-30K.json")
DEFAULT_MODEL   = "meta-llama/Llama-3.2-1B-Instruct"


# ── Rule-based constraint helpers ────────────────────────────────────────────


def _parse_length_constraint(rule_dict: dict):
    wl = rule_dict.get("word_length")
    if not wl:
        return None
    fi = wl.get("func_input", [])
    if len(fi) < 2:
        return None
    range_ = fi[1]
    if isinstance(range_, list) and len(range_) == 2:
        return (int(range_[0]), int(range_[1]))
    target = fi[2] if len(fi) > 2 and fi[2] is not None else None
    if target is not None:
        return (max(1, int(target * 0.8)), int(target * 1.2))
    return None


def _parse_keyword_constraint(rule_dict: dict):
    kw = rule_dict.get("keyword")
    if not kw:
        return None
    fi = kw.get("func_input", [])
    return fi[1] if len(fi) > 1 and isinstance(fi[1], dict) else None


def _parse_start_with_constraint(rule_dict: dict):
    sw = rule_dict.get("start_with")
    if not sw:
        return None
    fi = sw.get("func_input", [])
    return fi[1] if len(fi) > 1 and isinstance(fi[1], str) else None


def _parse_end_with_constraint(rule_dict: dict):
    ew = rule_dict.get("end_with")
    if not ew:
        return None
    fi = ew.get("func_input", [])
    return fi[1] if len(fi) > 1 and isinstance(fi[1], str) else None


def _constraint_score(response: str, rule_dict: dict) -> float:
    total, passed = 0, 0
    lp = _parse_length_constraint(rule_dict)
    if lp:
        total += 1
        passed += 1 if lp[0] <= len(response.split()) <= lp[1] else 0
    kw = _parse_keyword_constraint(rule_dict)
    if kw:
        total += 1
        rl = response.lower()
        passed += 1 if all(rl.count(k.lower()) >= int(v) for k, v in kw.items()) else 0
    sw = _parse_start_with_constraint(rule_dict)
    if sw:
        total += 1
        passed += 1 if response.strip().lower().startswith(sw.strip().lower()) else 0
    ew = _parse_end_with_constraint(rule_dict)
    if ew:
        total += 1
        passed += 1 if response.strip().lower().endswith(ew.strip().lower()) else 0
    return passed / total if total > 0 else 1.0


def _constraint_score_detailed(response: str, rule_dict: dict) -> dict:
    """Per-constraint-type pass/fail — used for plotting."""
    scores = {}
    lp = _parse_length_constraint(rule_dict)
    if lp:
        scores["Length"] = 1.0 if lp[0] <= len(response.split()) <= lp[1] else 0.0
    kw = _parse_keyword_constraint(rule_dict)
    if kw:
        rl = response.lower()
        scores["Keyword"] = 1.0 if all(rl.count(k.lower()) >= int(v) for k, v in kw.items()) else 0.0
    sw = _parse_start_with_constraint(rule_dict)
    if sw:
        scores["Start_With"] = 1.0 if response.strip().lower().startswith(sw.strip().lower()) else 0.0
    ew = _parse_end_with_constraint(rule_dict)
    if ew:
        scores["End_With"] = 1.0 if response.strip().lower().endswith(ew.strip().lower()) else 0.0
    return scores


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
        record.get("winner_prompt") or record.get("prompt")
        or record.get("instruction") or ""
    )
    response = (
        record.get("winner_response") or record.get("response")
        or record.get("output") or ""
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
        record.get("difficulty_level") or record.get("difficulty")
        or record.get("level") or _infer_difficulty(len(constraints))
    )
    difficulty = str(difficulty)
    if not difficulty.startswith("L"):
        difficulty = f"L{difficulty}"
    return {
        "id":                str(record.get("id", idx)),
        "prompt":            prompt,
        "response":          response,
        "difficulty_level":  difficulty,
        "rule_evaluate_dict": record.get("rule_evaluate_dict", {}),
    }


def _resolve_dataset_path(dataset_arg: str) -> str:
    p = Path(dataset_arg)
    if p.suffix != ".zip" or not p.exists():
        return dataset_arg
    extract_dir = p.parent
    logger.info(f"Extracting {p} → {extract_dir}")
    with zipfile.ZipFile(p, "r") as z:
        z.extractall(extract_dir)
    candidates = sorted(
        f for f in extract_dir.iterdir()
        if f.suffix in (".jsonl", ".json") and f.stem != p.stem
    )
    if not candidates:
        raise RuntimeError(f"No .jsonl/.json file found after extracting {p}")
    return str(candidates[0])


def load_recast_dataset(dataset_path: str) -> list[dict]:
    dataset_path = _resolve_dataset_path(dataset_path)
    path = Path(dataset_path)
    if path.exists():
        if path.stat().st_size < 1024:
            if b"git-lfs" in path.read_bytes():
                raise RuntimeError(
                    f"{path} is a Git LFS pointer. Run `git lfs pull` first."
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
        logger.info(f"Loading dataset from HuggingFace Hub: {dataset_path}")
        from datasets import load_dataset as hf_load
        raw = [dict(row) for row in hf_load(dataset_path, split="train")]
    records = [_parse_record(r, i) for i, r in enumerate(raw)]
    logger.info(f"Loaded {len(records)} records")
    return records


# ── Tokenisation ─────────────────────────────────────────────────────────────


def build_tokenised_dataset(
    records: list[dict],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    has_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None

    def tokenise(record):
        prompt_text   = record["prompt"]
        response_text = record["response"]
        rule_dict     = record.get("rule_evaluate_dict") or {}
        if has_template:
            full_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text},
                 {"role": "assistant", "content": response_text}],
                tokenize=False, add_generation_prompt=False,
            )
            prompt_text_formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt_text_formatted = f"User: {prompt_text}\nAssistant:"
            full_text = f"{prompt_text_formatted} {response_text}"
        full_enc   = tokenizer(full_text,             max_length=max_length, truncation=True, padding=False)
        prompt_enc = tokenizer(prompt_text_formatted, max_length=max_length, truncation=True, padding=False)
        input_ids  = full_enc["input_ids"]
        labels     = list(input_ids)
        labels[:len(prompt_enc["input_ids"])] = [-100] * len(prompt_enc["input_ids"])
        if all(lbl == -100 for lbl in labels):
            return {"input_ids": None, "attention_mask": None, "labels": None, "constraint_score": None}
        return {
            "input_ids":        input_ids,
            "attention_mask":   full_enc["attention_mask"],
            "labels":           labels,
            "constraint_score": _constraint_score(response_text, rule_dict),
        }

    logger.info("Tokenising dataset …")
    hf_ds     = Dataset.from_list(records)
    tokenised = hf_ds.map(tokenise, remove_columns=hf_ds.column_names, desc="Tokenising", num_proc=1)
    before    = len(tokenised)
    tokenised = tokenised.filter(lambda x: x["input_ids"] is not None)
    dropped   = before - len(tokenised)
    if dropped:
        logger.warning(f"Dropped {dropped} examples where prompt exceeded max_length={max_length}")
    return tokenised


# ── Constraint-aware training components ─────────────────────────────────────


class ConstraintDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        scores = [f.pop("constraint_score", 1.0) for f in features]
        batch  = super().__call__(features)
        batch["constraint_score"] = torch.tensor(scores, dtype=torch.float32)
        return batch


class ConstraintAwareTrainer(Trainer):
    def __init__(self, *args, constraint_lambda: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint_lambda = constraint_lambda

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        constraint_scores = inputs.pop("constraint_score", None)
        outputs   = model(**inputs)
        ce_loss   = outputs.loss
        if constraint_scores is not None and self.constraint_lambda > 0:
            constraint_loss = 1.0 - constraint_scores.float().mean()
            loss = ce_loss + self.constraint_lambda * constraint_loss
        else:
            constraint_loss = torch.tensor(0.0)
            loss = ce_loss
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({"ce_loss": ce_loss.detach().item(),
                      "constraint_loss": constraint_loss.detach().item()})
        return (loss, outputs) if return_outputs else loss


# ── Constraint evaluation ────────────────────────────────────────────────────


def evaluate_constraints(
    model, tokenizer, val_records: list[dict], max_length: int, n_samples: int = 50
) -> tuple[dict, float]:
    model.eval()
    sample     = random.sample(val_records, min(n_samples, len(val_records)))
    cat_scores: dict[str, list[float]] = {}
    overall: list[float] = []
    has_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None

    for rec in sample:
        prompt = rec["prompt"]
        rd     = rec.get("rule_evaluate_dict") or {}
        if has_template:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            text = f"User: {prompt}\nAssistant:"
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_length).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        detailed = _constraint_score_detailed(response, rd)
        for cat, score in detailed.items():
            cat_scores.setdefault(cat, []).append(score)
        if detailed:
            overall.append(sum(detailed.values()) / len(detailed))

    cat_means = {cat: float(np.mean(vals)) for cat, vals in cat_scores.items()}
    csr       = float(np.mean(overall)) if overall else 0.0
    return cat_means, csr


# ── Plotting & HTML report ────────────────────────────────────────────────────


def build_and_save_report(
    fold_results: list[dict],
    k_folds: int,
    model_name: str,
    dataset_path: str,
    constraint_lambda: float,
    results_dir: str,
    gdrive_results_dir: str,
) -> None:
    fold_nums    = [r["fold"]       for r in fold_results]
    eval_losses  = [r["eval_loss"]  for r in fold_results]
    train_losses = [r["train_loss"] for r in fold_results]
    csrs         = [r["csr"]        for r in fold_results]

    all_cats = sorted({cat for r in fold_results for cat in r.get("constraint_scores", {})})
    cat_means, cat_stds = [], []
    for cat in all_cats:
        vals = [r["constraint_scores"][cat] for r in fold_results
                if cat in r.get("constraint_scores", {})]
        cat_means.append(np.mean(vals) * 100)
        cat_stds.append(np.std(vals)   * 100)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].plot(fold_nums, train_losses, marker="o", label="Train Loss", color="#FF5722", linewidth=2)
    axes[0].plot(fold_nums, eval_losses,  marker="s", label="Val Loss",   color="#2196F3", linewidth=2)
    axes[0].set_xlabel("Fold"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Train vs Validation Loss per Fold")
    axes[0].set_xticks(fold_nums); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].bar(fold_nums, [c * 100 for c in csrs], color="#4CAF50", alpha=0.85, width=0.5)
    axes[1].axhline(np.mean(csrs) * 100, color="red", linestyle="--",
                    label=f"Mean = {np.mean(csrs)*100:.1f}%")
    axes[1].set_xlabel("Fold"); axes[1].set_ylabel("CSR (%)")
    axes[1].set_title("Overall Constraint Satisfaction Rate per Fold")
    axes[1].set_xticks(fold_nums); axes[1].set_ylim(0, 110)
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

    x    = list(range(len(all_cats)))
    bars = axes[2].bar(x, cat_means, yerr=cat_stds, capsize=6,
                       color="#9C27B0", alpha=0.85, width=0.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([c.replace("_", "\n") for c in all_cats], fontsize=9)
    axes[2].set_ylabel("Satisfaction Rate (%)")
    axes[2].set_title(f"% Constraints Followed per Type\n(mean ± std across {k_folds} folds)")
    axes[2].set_ylim(0, 120); axes[2].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, cat_means):
        axes[2].text(bar.get_x() + bar.get_width() / 2, val + 2,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle(
        f"Full Fine-Tuning — K={k_folds} Fold Cross-Validation\n"
        f"Llama 3.2 1B Instruct | RECAST-30K | STAT 453, Spring 2026",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    fold_rows = ""
    for r in fold_results:
        cat_cells = "".join(
            f"<td>{r['constraint_scores'].get(c, float('nan'))*100:.1f}%</td>"
            for c in all_cats
        )
        fold_rows += (
            f"<tr><td>{r['fold']}</td>"
            f"<td>{r['train_loss']:.4f}</td>"
            f"<td>{r['eval_loss']:.4f}</td>"
            f"<td>{r['csr']*100:.1f}%</td>"
            f"{cat_cells}</tr>\n"
        )

    cat_headers  = "".join(f"<th>{c}</th>" for c in all_cats)
    summary_rows = (
        f"<tr><td>Eval Loss</td><td>{np.mean(eval_losses):.4f}</td><td>{np.std(eval_losses):.4f}</td></tr>\n"
        f"<tr><td>Train Loss</td><td>{np.mean(train_losses):.4f}</td><td>{np.std(train_losses):.4f}</td></tr>\n"
        f"<tr><td>CSR (overall)</td><td>{np.mean(csrs)*100:.1f}%</td><td>{np.std(csrs)*100:.1f}%</td></tr>\n"
    )
    for cat, m, s in zip(all_cats, cat_means, cat_stds):
        summary_rows += f"<tr><td>{cat}</td><td>{m:.1f}%</td><td>{s:.1f}%</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Full Fine-Tuning K-Fold Results</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; color: #333; }}
  h1   {{ color: #1a237e; }}
  h2   {{ color: #283593; border-bottom: 2px solid #9C27B0; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: center; }}
  th   {{ background: #7B1FA2; color: white; }}
  tr:nth-child(even) {{ background: #f3e5f5; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 30px; }}
  img  {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 30px; }}
</style>
</head>
<body>
<h1>Full Fine-Tuning — K-Fold Cross-Validation Results</h1>
<p class="meta">
  Model: <strong>{model_name}</strong> &nbsp;|&nbsp;
  Dataset: <strong>{dataset_path}</strong> &nbsp;|&nbsp;
  K = <strong>{k_folds}</strong> &nbsp;|&nbsp;
  &lambda; = <strong>{constraint_lambda}</strong> &nbsp;|&nbsp;
  Generated: <strong>{datetime.now().strftime("%Y-%m-%d %H:%M")}</strong>
</p>
<h2>Plots</h2>
<img src="data:image/png;base64,{img_b64}" alt="K-Fold Results Plot">
<h2>Per-Fold Metrics</h2>
<table>
  <tr><th>Fold</th><th>Train Loss</th><th>Val Loss</th><th>CSR</th>{cat_headers}</tr>
  {fold_rows}
</table>
<h2>Summary (mean ± std across {k_folds} folds)</h2>
<table>
  <tr><th>Metric</th><th>Mean</th><th>Std</th></tr>
  {summary_rows}
</table>
</body>
</html>"""

    os.makedirs(results_dir, exist_ok=True)
    local_path = os.path.join(results_dir, "kfold_results.html")
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"HTML report saved locally: {local_path}")

    if gdrive_results_dir:
        try:
            os.makedirs(gdrive_results_dir, exist_ok=True)
            gdrive_path = os.path.join(gdrive_results_dir, "kfold_results.html")
            shutil.copy(local_path, gdrive_path)
            logger.info(f"HTML report saved to Google Drive: {gdrive_path}")
        except Exception as e:
            logger.warning(f"Could not save to Google Drive: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full fine-tuning on RECAST constraint-following data with K-Fold CV."
    )
    p.add_argument("--model_name",   default=DEFAULT_MODEL)
    p.add_argument("--dataset",      default=DEFAULT_DATASET)
    p.add_argument("--output_dir",   default="./output/finetuned")
    p.add_argument("--results_dir",  default="./results",
                   help="Local folder for HTML report")
    p.add_argument("--gdrive_results_dir", default="",
                   help="Google Drive path for HTML report (e.g. /content/drive/MyDrive/stat453/results)")
    p.add_argument("--num_train_epochs",             type=int,   default=2)
    p.add_argument("--per_device_train_batch_size",  type=int,   default=4)
    p.add_argument("--per_device_eval_batch_size",   type=int,   default=4)
    p.add_argument("--gradient_accumulation_steps",  type=int,   default=8)
    p.add_argument("--learning_rate",                type=float, default=2e-5)
    p.add_argument("--lr_scheduler_type",            default="cosine")
    p.add_argument("--warmup_ratio",                 type=float, default=0.03)
    p.add_argument("--max_length",                   type=int,   default=512)
    p.add_argument("--k_folds",                      type=int,   default=5,
                   help="Number of K-Fold cross-validation splits")
    p.add_argument("--eval_samples",                 type=int,   default=50,
                   help="Val examples to score for constraint satisfaction per fold")
    p.add_argument("--seed",                         type=int,   default=42)
    p.add_argument("--hf_token",     default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_steps",    type=int, default=500)
    p.add_argument("--eval_steps",    type=int, default=500)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--num_samples",  type=int,   default=0,
                   help="Use only this many records (0 = all)")
    p.add_argument("--constraint_lambda", type=float, default=0.1)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
        logger.info("Logged in to HuggingFace Hub.")

    records = load_recast_dataset(args.dataset)
    if args.num_samples > 0:
        records = records[:args.num_samples]
        logger.info(f"Subsampled to {len(records)} records")
    logger.info(f"Total records: {len(records)} — splitting into {args.k_folds} folds")

    kf           = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold_num, (train_idx, val_idx) in enumerate(kf.split(records), start=1):
        logger.info(f"\n{'='*60}")
        logger.info(f"  FOLD {fold_num} / {args.k_folds}")
        logger.info(f"{'='*60}")

        train_records_fold = [records[i] for i in train_idx]
        val_records_fold   = [records[i] for i in val_idx]
        logger.info(f"Fold {fold_num}: Train={len(train_records_fold)}, Val={len(val_records_fold)}")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        train_ds = build_tokenised_dataset(train_records_fold, tokenizer, args.max_length)
        val_ds   = build_tokenised_dataset(val_records_fold,   tokenizer, args.max_length)

        n_gpus     = torch.cuda.device_count()
        device_map = "auto" if n_gpus <= 1 else None
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map=device_map
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        for param in model.parameters():
            param.requires_grad = True

        fold_output_dir = os.path.join(args.output_dir, f"fold_{fold_num}")
        safe_model_name = args.model_name.replace("/", "_")
        training_args = TrainingArguments(
            output_dir                  = fold_output_dir,
            num_train_epochs            = args.num_train_epochs,
            per_device_train_batch_size = args.per_device_train_batch_size,
            per_device_eval_batch_size  = args.per_device_eval_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            learning_rate               = args.learning_rate,
            lr_scheduler_type           = args.lr_scheduler_type,
            warmup_ratio                = args.warmup_ratio,
            bf16                        = torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16                        = False,
            logging_dir                 = os.path.join(fold_output_dir, "logs"),
            logging_steps               = args.logging_steps,
            eval_strategy               = "steps",
            eval_steps                  = args.eval_steps,
            save_strategy               = "steps",
            save_steps                  = args.save_steps,
            save_total_limit            = 2,
            load_best_model_at_end      = True,
            metric_for_best_model       = "eval_loss",
            greater_is_better           = False,
            report_to                   = "none",
            seed                        = args.seed,
            dataloader_num_workers      = 0,
            remove_unused_columns       = False,
            label_names                 = ["labels"],
            run_name                    = f"full_ft_{safe_model_name}_fold{fold_num}",
        )

        if args.constraint_lambda > 0:
            data_collator = ConstraintDataCollator(
                tokenizer=tokenizer, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100
            )
            trainer = ConstraintAwareTrainer(
                model=model, args=training_args,
                train_dataset=train_ds, eval_dataset=val_ds,
                data_collator=data_collator, processing_class=tokenizer,
                constraint_lambda=args.constraint_lambda,
            )
        else:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100
            )
            trainer = Trainer(
                model=model, args=training_args,
                train_dataset=train_ds, eval_dataset=val_ds,
                data_collator=data_collator, processing_class=tokenizer,
            )

        logger.info(f"Starting fold {fold_num} training …")
        train_result = trainer.train()
        eval_metrics = trainer.evaluate()

        logger.info(f"Evaluating constraint satisfaction on {args.eval_samples} val examples …")
        cat_scores, csr = evaluate_constraints(
            model, tokenizer, val_records_fold, args.max_length, args.eval_samples
        )
        logger.info(f"  CSR={csr:.4f}  per-type: {cat_scores}")

        fold_results.append({
            "fold":              fold_num,
            "eval_loss":         eval_metrics.get("eval_loss"),
            "train_loss":        train_result.training_loss,
            "csr":               csr,
            "constraint_scores": cat_scores,
        })

        trainer.save_model(fold_output_dir)
        tokenizer.save_pretrained(fold_output_dir)
        with open(os.path.join(fold_output_dir, "fold_meta.json"), "w") as f:
            json.dump(fold_results[-1], f, indent=2)

        del model, trainer, train_ds, val_ds, tokenizer, data_collator
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"Fold {fold_num} complete. GPU memory freed.")

    # ── Aggregate ────────────────────────────────────────────────────────────
    eval_losses  = [r["eval_loss"]  for r in fold_results]
    train_losses = [r["train_loss"] for r in fold_results]
    csrs         = [r["csr"]        for r in fold_results]

    logger.info(f"\n{'='*60}")
    logger.info(f"  K-FOLD RESULTS (K={args.k_folds})")
    logger.info(f"{'='*60}")
    for r in fold_results:
        logger.info(
            f"  Fold {r['fold']}: eval_loss={r['eval_loss']:.4f}  "
            f"train_loss={r['train_loss']:.4f}  CSR={r['csr']*100:.1f}%"
        )
    logger.info(f"  Eval Loss  — mean={np.mean(eval_losses):.4f}  std={np.std(eval_losses):.4f}")
    logger.info(f"  Train Loss — mean={np.mean(train_losses):.4f}  std={np.std(train_losses):.4f}")
    logger.info(f"  CSR        — mean={np.mean(csrs)*100:.1f}%  std={np.std(csrs)*100:.1f}%")

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "k_folds":         args.k_folds,
        "model_name":      args.model_name,
        "dataset":         args.dataset,
        "fold_results":    fold_results,
        "eval_loss_mean":  float(np.mean(eval_losses)),
        "eval_loss_std":   float(np.std(eval_losses)),
        "train_loss_mean": float(np.mean(train_losses)),
        "train_loss_std":  float(np.std(train_losses)),
        "csr_mean":        float(np.mean(csrs)),
        "csr_std":         float(np.std(csrs)),
        "constraint_lambda": args.constraint_lambda,
    }
    with open(os.path.join(args.output_dir, "kfold_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    build_and_save_report(
        fold_results       = fold_results,
        k_folds            = args.k_folds,
        model_name         = args.model_name,
        dataset_path       = args.dataset,
        constraint_lambda  = args.constraint_lambda,
        results_dir        = args.results_dir,
        gdrive_results_dir = args.gdrive_results_dir,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
