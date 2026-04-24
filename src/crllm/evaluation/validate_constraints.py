"""
Constraint Validation (CV) script for RECAST-30K constraints.

Evaluates a model on 6 constraint types:

  Rule-based (from rule_evaluate_dict — always checked deterministically):
  - length     : response word count within required [min, max] range
  - keyword    : required keywords present at required frequency
  - end_with   : response ends with required string (case-insensitive)
  - start_with : response starts with required string (case-insensitive)

  LLM-judged only (from added_constraint_from_LLM — require --judge_model):
  - style      : response follows the requested writing style / tone
  - topic      : response stays on the required topic / focus area

By default, all six checks run with the LLM judge when --judge_model is set.
Without --judge_model only the 4 rule-based checks run; style/topic are skipped.
When the judge output is unparseable, the example falls back to rule-based checks
(style/topic count as None for that example).

Scoring per example:
  score = satisfied_constraints / total_applicable_constraints
  loss  = 1 - score          (target: minimize → 0)

Overall loss = 1 - (total_satisfied / total_constraints_across_dataset)

Usage (rule-based only):
    python validate_constraints.py \\
        --model   meta-llama/Llama-3.2-1B-Instruct \\
        --dataset datasets/recast_30k_clean.jsonl

Usage (LLM judge — all 6 constraints):
    python validate_constraints.py \\
        --model       meta-llama/Llama-3.2-1B-Instruct \\
        --dataset     datasets/recast_30k_clean.jsonl \\
        --judge_model Qwen/Qwen2.5-7B-Instruct \\
        --judge_device cuda:1
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
import zipfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Constraint parsing ────────────────────────────────────────────────────────


def parse_length_constraint(rule_dict: dict):
    """Return (min_words, max_words) from word_length entry, or None."""
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
        lo = max(1, int(target * 0.8))
        hi = int(target * 1.2)
        return (lo, hi)
    return None


def parse_keyword_constraint(rule_dict: dict):
    """Return {keyword: required_count} dict, or None."""
    kw = rule_dict.get("keyword")
    if not kw:
        return None
    fi = kw.get("func_input", [])
    if len(fi) < 2 or not isinstance(fi[1], dict):
        return None
    return fi[1]


def parse_start_with_constraint(rule_dict: dict):
    """Return target string, or None."""
    sw = rule_dict.get("start_with")
    if not sw:
        return None
    fi = sw.get("func_input", [])
    return fi[1] if len(fi) > 1 and isinstance(fi[1], str) else None


def parse_end_with_constraint(rule_dict: dict):
    """Return target string, or None."""
    ew = rule_dict.get("end_with")
    if not ew:
        return None
    fi = ew.get("func_input", [])
    return fi[1] if len(fi) > 1 and isinstance(fi[1], str) else None


def parse_style_constraints(llm_constraints: dict) -> list[str] | None:
    """Return list of style descriptions from added_constraint_from_LLM, or None."""
    items = llm_constraints.get("Style", llm_constraints.get("style", []))
    return items if items else None


def parse_topic_constraints(llm_constraints: dict) -> list[str] | None:
    """Return list of topic descriptions from added_constraint_from_LLM, or None."""
    items = llm_constraints.get("Topic", llm_constraints.get("topic", []))
    return items if items else None


# ── Rule-based checkers (used directly or as judge fallback) ──────────────────


def check_length(response: str, min_words: int, max_words: int) -> int:
    return 1 if min_words <= len(response.split()) <= max_words else 0


def check_keyword(response: str, keyword_dict: dict) -> int:
    """1 if every keyword appears at least its required count of times."""
    resp_lower = response.lower()
    for kw, required in keyword_dict.items():
        if resp_lower.count(kw.lower()) < int(required):
            return 0
    return 1


def check_start_with(response: str, target: str) -> int:
    return 1 if response.strip().lower().startswith(target.strip().lower()) else 0


def check_end_with(response: str, target: str) -> int:
    return 1 if response.strip().lower().endswith(target.strip().lower()) else 0


# ── Rule-based scoring ────────────────────────────────────────────────────────


def _verdicts_to_score(verdicts: dict) -> dict:
    """Convert a {key: 0/1/None} verdicts dict into the full scored result."""
    checked = {k: v for k, v in verdicts.items() if v is not None}
    total = len(checked)
    passed = sum(checked.values())
    score = passed / total if total > 0 else 0.0
    return {
        **verdicts,
        "passed": passed,
        "total": total,
        "score": score,
        "loss": round(1.0 - score, 6),
    }


def score_response(response: str, rule_dict: dict) -> dict:
    """
    Deterministic rule-based scoring for a single response.
    Style and topic cannot be checked deterministically and are always None here;
    they are only scored when an LLM judge is provided (see main()).
    """
    length_params = parse_length_constraint(rule_dict)
    kw_params = parse_keyword_constraint(rule_dict)
    sw_param = parse_start_with_constraint(rule_dict)
    ew_param = parse_end_with_constraint(rule_dict)

    verdicts = {
        "length":     check_length(response, *length_params) if length_params else None,
        "keyword":    check_keyword(response, kw_params)     if kw_params     else None,
        "start_with": check_start_with(response, sw_param)   if sw_param      else None,
        "end_with":   check_end_with(response, ew_param)     if ew_param      else None,
        "style":      None,
        "topic":      None,
    }
    return _verdicts_to_score(verdicts)


# ── LLM judge ─────────────────────────────────────────────────────────────────


def _build_judge_prompt(
    response: str,
    rule_dict: dict,
    llm_constraints: dict | None = None,
) -> tuple[str, list[str]]:
    """
    Build a judge prompt describing each applicable constraint in plain English.

    Covers 4 rule-based constraints (from rule_evaluate_dict) plus style and topic
    when llm_constraints (from added_constraint_from_LLM) is provided.

    For style/topic with multiple descriptions, all are listed under a single key;
    the judge must output 1 only if ALL listed descriptions are satisfied.

    Returns:
        (prompt_text, applicable_keys)  where applicable_keys lists the constraint
        names the judge is asked to evaluate (used to parse its output).
    """
    lines = []
    keys = []

    # ── Rule-based constraints ────────────────────────────────────────────────
    lp = parse_length_constraint(rule_dict)
    if lp:
        lines.append(f'- length: The word count must be between {lp[0]} and {lp[1]} words.')
        keys.append("length")

    kw = parse_keyword_constraint(rule_dict)
    if kw:
        parts = ", ".join(f'"{k}" at least {v} time(s)' for k, v in kw.items())
        lines.append(f"- keyword: The response must contain {parts}.")
        keys.append("keyword")

    sw = parse_start_with_constraint(rule_dict)
    if sw:
        lines.append(f'- start_with: The response must begin with the exact text "{sw}".')
        keys.append("start_with")

    ew = parse_end_with_constraint(rule_dict)
    if ew:
        lines.append(f'- end_with: The response must end with the exact text "{ew}".')
        keys.append("end_with")

    # ── LLM constraints (style / topic) ──────────────────────────────────────
    if llm_constraints:
        styles = parse_style_constraints(llm_constraints)
        if styles:
            desc = " AND ".join(f'"{s}"' for s in styles)
            lines.append(
                f"- style: The response must follow this writing style/tone: {desc}. "
                "Output 1 only if ALL style requirements are met."
            )
            keys.append("style")

        topics = parse_topic_constraints(llm_constraints)
        if topics:
            desc = " AND ".join(f'"{t}"' for t in topics)
            lines.append(
                f"- topic: The response must address this topic/focus: {desc}. "
                "Output 1 only if ALL topic requirements are met."
            )
            keys.append("topic")

    if not keys:
        return "", []

    json_template = "{" + ", ".join(f'"{k}": <0 or 1>' for k in keys) + "}"

    prompt = (
        "You are a strict constraint-checking assistant.\n\n"
        "Evaluate whether the following response satisfies each listed constraint.\n"
        "For each constraint output 1 (satisfied) or 0 (not satisfied).\n"
        "Respond ONLY with a valid JSON object — no explanation, no extra text.\n\n"
        f'Response:\n"""\n{response}\n"""\n\n'
        "Constraints:\n"
        + "\n".join(lines)
        + f"\n\nOutput (JSON only): {json_template}"
    )
    return prompt, keys


def _parse_judge_output(text: str, keys: list[str]) -> dict | None:
    """
    Extract {key: 0_or_1} from raw judge output text.
    Returns None if the output cannot be parsed reliably.
    """
    # Try to find the first {...} block
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if not match:
        return None
    try:
        raw = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    verdicts = {}
    for k in keys:
        val = raw.get(k)
        if val is None:
            return None  # incomplete output — reject the whole response
        try:
            verdicts[k] = int(bool(int(val)))
        except (TypeError, ValueError):
            return None
    return verdicts


def _judge_format_prompt(prompt: str, tokenizer) -> str:
    has_template = (
        hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    )
    if has_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {prompt}\nAssistant:"


def batch_judge_scores(
    responses: list[str],
    rule_dicts: list[dict],
    judge_model,
    judge_tokenizer,
    batch_size: int,
    llm_constraints_list: list[dict] | None = None,
) -> list[dict | None]:
    """
    Run all examples through the judge model in batches.

    Evaluates 4 rule-based constraints (length, keyword, start_with, end_with)
    plus style and topic when llm_constraints_list is provided.

    Returns a list of verdict dicts (one per example).  An entry is None when
    the judge output could not be parsed; the caller should fall back to
    rule-based scoring for those examples.
    """
    if llm_constraints_list is None:
        llm_constraints_list = [{}] * len(responses)

    prompts_and_keys = [
        _build_judge_prompt(r, d, lc)
        for r, d, lc in zip(responses, rule_dicts, llm_constraints_list)
    ]
    formatted = [
        _judge_format_prompt(prompt, judge_tokenizer) if prompt else ""
        for prompt, _ in prompts_and_keys
    ]

    all_verdicts: list[dict | None] = [None] * len(responses)
    device = next(judge_model.parameters()).device

    for i in range(0, len(formatted), batch_size):
        batch_texts = formatted[i : i + batch_size]
        batch_keys  = [prompts_and_keys[i + j][1] for j in range(len(batch_texts))]

        # Skip examples with no applicable constraints
        non_empty = [(j, t) for j, t in enumerate(batch_texts) if t]
        if not non_empty:
            continue

        batch_indices, batch_inputs = zip(*non_empty)
        enc = judge_tokenizer(
            list(batch_inputs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            out = judge_model.generate(
                **enc,
                max_new_tokens=64,   # JSON verdict is always short
                pad_token_id=judge_tokenizer.eos_token_id,
                do_sample=False,
            )

        for local_j, global_j in enumerate(batch_indices):
            prompt_len = enc["input_ids"][local_j].shape[0]
            raw_text = judge_tokenizer.decode(
                out[local_j][prompt_len:], skip_special_tokens=True
            )
            keys = batch_keys[global_j]
            verdict = _parse_judge_output(raw_text, keys)
            all_verdicts[i + global_j] = verdict

        done = min(i + batch_size, len(formatted))
        logger.info(f"  Judge scored {done}/{len(formatted)}")

    return all_verdicts


# ── Dataset loading ───────────────────────────────────────────────────────────


def resolve_path(path_arg: str) -> str:
    """Extract zip to its parent dir if needed and return the usable file path."""
    p = Path(path_arg)
    if p.suffix == ".zip" and p.exists():
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
        logger.info(f"Using extracted file: {candidates[0]}")
        return str(candidates[0])
    return path_arg


def load_dataset(dataset_path: str) -> list[dict]:
    path = Path(resolve_path(dataset_path))
    logger.info(f"Loading dataset from {path}")
    records = []
    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with open(path) as f:
            data = json.load(f)
        records = data if isinstance(data, list) else list(data.values())
    logger.info(f"Loaded {len(records)} records")
    return records


# ── Generation ────────────────────────────────────────────────────────────────


def load_model_and_tokenizer(model_name: str, device: str, hf_token: str):
    logger.info(f"Loading tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(
        model_name, token=hf_token or None, padding_side="left"
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    logger.info(f"Loading model: {model_name}")
    kwargs = dict(torch_dtype=torch.float16, token=hf_token or None)
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device != "auto":
        model = model.to(device)
    model.eval()
    return model, tok


def _format_prompt(prompt: str, tokenizer) -> str:
    has_template = (
        hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    )
    if has_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {prompt}\nAssistant:"


def generate_responses(
    prompts: list[str],
    model,
    tokenizer,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    """Batch-generate responses; returns one decoded string per prompt."""
    formatted = [_format_prompt(p, tokenizer) for p in prompts]
    all_responses = []
    device = next(model.parameters()).device

    for i in range(0, len(formatted), batch_size):
        batch = formatted[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        for j, ids in enumerate(out):
            prompt_len = enc["input_ids"][j].shape[0]
            text = tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
            all_responses.append(text.strip())

        done = min(i + batch_size, len(formatted))
        logger.info(f"  Generated {done}/{len(formatted)}")

    return all_responses


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Constraint validation: score model outputs on 4 rule-based constraints."
    )
    p.add_argument("--model",   required=True, help="HuggingFace model ID or local path (model being evaluated)")
    p.add_argument("--dataset", required=True, help="Path to RECAST JSONL (or .zip containing it)")
    p.add_argument("--num_samples",    type=int, default=0,     help="Examples to evaluate; 0 = all")
    p.add_argument("--batch_size",     type=int, default=4,     help="Generation batch size")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--output_dir",     default="./eval_results")
    p.add_argument("--device",         default="auto",          help="Device for the evaluated model")
    p.add_argument("--hf_token",       default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--seed",           type=int, default=42)

    # Judge options
    p.add_argument(
        "--judge_model",
        default=None,
        help=(
            "HuggingFace model ID or local path for the LLM judge. "
            "When set, constraint checks are performed by the judge instead of "
            "deterministic rules. Falls back to rule-based when the judge output "
            "cannot be parsed.  Example: Qwen/Qwen2.5-7B-Instruct"
        ),
    )
    p.add_argument(
        "--judge_device",
        default="auto",
        help="Device for the judge model (default: auto). Use 'cuda:1' to put it on a second GPU.",
    )
    p.add_argument(
        "--judge_batch_size",
        type=int,
        default=4,
        help="Batch size for judge inference (judge inputs are longer; tune independently).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    # ── Load data ────────────────────────────────────────────────────────────
    records = load_dataset(args.dataset)
    if args.num_samples > 0:
        random.shuffle(records)
        records = records[: args.num_samples]
        logger.info(f"Subsampled to {len(records)} records")

    prompts          = [r.get("winner_prompt") or r.get("prompt") or r.get("instruction") or "" for r in records]
    rule_dicts       = [r.get("rule_evaluate_dict", {}) for r in records]
    llm_constraints  = [r.get("added_constraint_from_LLM", {}) for r in records]
    ids              = [str(r.get("id", i)) for i, r in enumerate(records)]

    # ── Generate responses from evaluated model ───────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.hf_token)
    t0 = time.time()
    responses = generate_responses(prompts, model, tokenizer, args.batch_size, args.max_new_tokens)
    gen_elapsed = time.time() - t0
    logger.info(f"Generation complete in {gen_elapsed:.1f}s")

    # ── Score ─────────────────────────────────────────────────────────────────
    using_judge = args.judge_model is not None
    judge_fallback_count = 0

    if using_judge:
        logger.info(f"Loading judge model: {args.judge_model}")
        judge_model, judge_tokenizer = load_model_and_tokenizer(
            args.judge_model, args.judge_device, args.hf_token
        )
        t1 = time.time()
        judge_verdicts = batch_judge_scores(
            responses, rule_dicts, judge_model, judge_tokenizer,
            args.judge_batch_size, llm_constraints_list=llm_constraints,
        )
        judge_elapsed = time.time() - t1
        logger.info(f"Judge scoring complete in {judge_elapsed:.1f}s")
    else:
        judge_verdicts = [None] * len(responses)
        judge_elapsed = 0.0

    scored = []
    for rid, response, rule_dict, lc, jv in zip(ids, responses, rule_dicts, llm_constraints, judge_verdicts):
        if using_judge and jv is not None:
            # Fill in None for any key not returned by the judge
            full_verdicts = {
                "length": None, "keyword": None,
                "start_with": None, "end_with": None,
                "style": None, "topic": None,
            }
            full_verdicts.update(jv)
            result = _verdicts_to_score(full_verdicts)
            result["scorer"] = "judge"
        else:
            if using_judge:
                # Judge output was unparseable — fall back to rule-based for this example
                judge_fallback_count += 1
            result = score_response(response, rule_dict)
            result["scorer"] = "rule" if not using_judge else "rule(fallback)"
        result["id"] = rid
        result["response"] = response
        scored.append(result)

    if using_judge and judge_fallback_count:
        logger.warning(
            f"{judge_fallback_count}/{len(scored)} examples fell back to rule-based "
            "scoring because the judge output could not be parsed."
        )

    total_elapsed = time.time() - t0

    # ── Aggregate metrics ────────────────────────────────────────────────────
    total_passed      = sum(r["passed"] for r in scored)
    total_constraints = sum(r["total"]  for r in scored)
    overall_score     = total_passed / total_constraints if total_constraints > 0 else 0.0
    overall_loss      = round(1.0 - overall_score, 6)

    per_constraint = {}
    for key in ("length", "keyword", "start_with", "end_with", "style", "topic"):
        applicable = [r[key] for r in scored if r.get(key) is not None]
        if applicable:
            pass_rate = sum(applicable) / len(applicable)
            per_constraint[key] = {
                "pass_rate": round(pass_rate, 4),
                "loss":      round(1.0 - pass_rate, 4),
                "n":         len(applicable),
            }

    # ── Print summary ────────────────────────────────────────────────────────
    sep = "=" * 58
    scorer_label = f"judge={args.judge_model}" if using_judge else "rule-based"
    print(f"\n{sep}")
    print(f"  Constraint Validation — {args.model}")
    print(f"  Scorer   : {scorer_label}")
    print(f"  Examples : {len(scored)}   Total time: {total_elapsed:.1f}s")
    if using_judge:
        print(f"  Judge fallbacks: {judge_fallback_count}")
    print(sep)
    print(f"\n  {'Constraint':<15} {'Pass Rate':>10} {'Loss':>8} {'N':>7}")
    print(f"  {'-'*42}")
    for key, m in per_constraint.items():
        print(f"  {key:<15} {m['pass_rate']:>10.4f} {m['loss']:>8.4f} {m['n']:>7}")
    print(f"  {'-'*42}")
    print(f"  {'OVERALL':<15} {overall_score:>10.4f} {overall_loss:>8.4f} {total_constraints:>7}")
    print(f"\n  Overall loss = 1 - ({total_passed}/{total_constraints}) = {overall_loss:.6f}")
    print(f"{sep}\n")

    # ── Save results ─────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = args.model.replace("/", "_").replace("-", "_")
    tag = "judge" if using_judge else "rules"

    csv_path = os.path.join(args.output_dir, f"cv_{tag}_{model_safe}_{ts}.csv")
    fieldnames = ["id", "scorer", "length", "keyword", "start_with", "end_with",
                  "style", "topic", "passed", "total", "score", "loss"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in scored:
            writer.writerow({k: r.get(k) for k in fieldnames})
    logger.info(f"Per-example results saved to {csv_path}")

    meta_path = os.path.join(args.output_dir, f"cv_{tag}_{model_safe}_{ts}_summary.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "model":              args.model,
                "judge_model":        args.judge_model,
                "dataset":            args.dataset,
                "num_examples":       len(scored),
                "scorer":             scorer_label,
                "judge_fallbacks":    judge_fallback_count,
                "gen_elapsed_s":      round(gen_elapsed, 2),
                "judge_elapsed_s":    round(judge_elapsed, 2),
                "total_elapsed_s":    round(total_elapsed, 2),
                "overall_score":      round(overall_score, 6),
                "overall_loss":       overall_loss,
                "total_passed":       total_passed,
                "total_constraints":  total_constraints,
                "per_constraint":     per_constraint,
            },
            f,
            indent=2,
        )
    logger.info(f"Summary saved to {meta_path}")


if __name__ == "__main__":
    main()
