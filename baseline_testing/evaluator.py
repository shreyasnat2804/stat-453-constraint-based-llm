"""
Reusable evaluation module for RECAST constraint-following evaluation.

Works for both baseline and finetuned model evaluation. Core functions:
- evaluate_responses(): run constraint checking on model outputs
- compute_metrics(): aggregate CSR/hard CSR by level and constraint type
- save_results_csv(): write scored CSV with model name, label, and timing
"""

import json
import os
import time
from collections import defaultdict
from datetime import datetime

import pandas as pd

from constraint_checker import ConstraintChecker


def evaluate_responses(results: list[dict]) -> list[dict]:
    """Run deterministic constraint checking on all inference results.

    Args:
        results: List of dicts, each with keys "response" and "constraints".

    Returns:
        The same list, mutated in-place with constraint check fields added
        (results, num_constraints, num_checked, num_passed,
        per_constraint_csr, hard_csr).
    """
    checker = ConstraintChecker()
    for item in results:
        eval_result = checker.check_all(item["response"], item["constraints"])
        item.update(eval_result)
    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute per-level and per-type metrics from evaluated results.

    Args:
        results: List of dicts that have already been through evaluate_responses().

    Returns:
        Dict with keys:
        - "by_level": {level: {"csr", "hard_csr", "count"}} for L1-L4 + Overall
        - "per_type": {constraint_type: pass_rate}
    """
    # Group by difficulty level
    by_level = defaultdict(list)
    for r in results:
        if r.get("num_checked", 0) > 0:
            by_level[r["difficulty_level"]].append(r)

    metrics_by_level = {}
    for level in ["L1", "L2", "L3", "L4"]:
        items = by_level.get(level, [])
        if items:
            csr = sum(r["per_constraint_csr"] for r in items) / len(items)
            hard = sum(1 for r in items if r["hard_csr"]) / len(items)
        else:
            csr, hard = 0.0, 0.0
        metrics_by_level[level] = {
            "csr": round(csr, 4),
            "hard_csr": round(hard, 4),
            "count": len(items),
        }

    # Overall
    all_checked = [r for r in results if r.get("num_checked", 0) > 0]
    if all_checked:
        overall_csr = sum(r["per_constraint_csr"] for r in all_checked) / len(all_checked)
        overall_hard = sum(1 for r in all_checked if r["hard_csr"]) / len(all_checked)
    else:
        overall_csr, overall_hard = 0.0, 0.0
    metrics_by_level["Overall"] = {
        "csr": round(overall_csr, 4),
        "hard_csr": round(overall_hard, 4),
        "count": len(all_checked),
    }

    # Per constraint type pass rate
    type_pass = defaultdict(lambda: {"passed": 0, "total": 0})
    for r in results:
        for cr in r.get("results", []):
            if cr["passed"] is not None:
                type_pass[cr["type"]]["total"] += 1
                if cr["passed"]:
                    type_pass[cr["type"]]["passed"] += 1

    per_type_rates = {}
    for ctype, counts in sorted(type_pass.items()):
        rate = counts["passed"] / counts["total"] if counts["total"] > 0 else 0
        per_type_rates[ctype] = round(rate, 4)

    return {"by_level": metrics_by_level, "per_type": per_type_rates}


def save_results_csv(
    results: list[dict],
    metrics: dict,
    output_dir: str,
    model_name: str,
    label: str = "baseline",
    elapsed_seconds: float = 0.0,
) -> str:
    """Save evaluation results to a CSV in output_dir.

    The CSV contains one row per instance with columns:
        id, difficulty_level, num_constraints, num_checked, num_passed,
        per_constraint_csr, hard_csr

    A summary row is appended with overall metrics, model name, label, and
    elapsed time.

    Args:
        results: Evaluated result dicts.
        metrics: Output of compute_metrics().
        output_dir: Directory to write the CSV into.
        model_name: HuggingFace model ID or display name.
        label: "baseline" or "finetuned" (appears in filename and summary).
        elapsed_seconds: Wall-clock time for inference + evaluation.

    Returns:
        Path to the saved CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_safe = model_name.replace("/", "_").replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{model_safe}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    # Per-instance rows
    rows = []
    for r in results:
        rows.append({
            "id": r.get("id", ""),
            "difficulty_level": r.get("difficulty_level", ""),
            "num_constraints": r.get("num_constraints", 0),
            "num_checked": r.get("num_checked", 0),
            "num_passed": r.get("num_passed", 0),
            "per_constraint_csr": r.get("per_constraint_csr", 0.0),
            "hard_csr": r.get("hard_csr", False),
        })
    df = pd.DataFrame(rows)

    # Summary row
    by_level = metrics["by_level"]
    summary = {
        "id": "SUMMARY",
        "difficulty_level": "Overall",
        "num_constraints": sum(r.get("num_constraints", 0) for r in results),
        "num_checked": sum(r.get("num_checked", 0) for r in results),
        "num_passed": sum(r.get("num_passed", 0) for r in results),
        "per_constraint_csr": by_level["Overall"]["csr"],
        "hard_csr": by_level["Overall"]["hard_csr"],
    }
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    df.to_csv(filepath, index=False)

    # Also save a companion metadata JSON
    meta = {
        "model": model_name,
        "label": label,
        "timestamp": timestamp,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "num_instances": len(results),
        "by_level": by_level,
        "per_type": metrics["per_type"],
    }
    meta_path = os.path.join(output_dir, f"{label}_{model_safe}_{timestamp}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return filepath


def print_summary(metrics: dict, model_name: str, label: str, elapsed_seconds: float = 0.0):
    """Print a human-readable summary of evaluation metrics."""
    by_level = metrics["by_level"]
    print(f"\n{'='*60}")
    print(f"  {label.upper()} Evaluation: {model_name}")
    if elapsed_seconds > 0:
        mins = int(elapsed_seconds // 60)
        secs = elapsed_seconds % 60
        print(f"  Time: {mins}m {secs:.1f}s")
    print(f"{'='*60}")

    print(f"\n  {'Level':<10} {'CSR':>8} {'Hard CSR':>10} {'Count':>8}")
    print(f"  {'-'*36}")
    for level in ["L1", "L2", "L3", "L4", "Overall"]:
        m = by_level[level]
        print(f"  {level:<10} {m['csr']:>8.4f} {m['hard_csr']:>10.4f} {m['count']:>8}")

    print(f"\n  Per-type pass rates:")
    for ctype, rate in sorted(metrics["per_type"].items(), key=lambda x: x[1], reverse=True):
        print(f"    {ctype:<45} {rate:.4f}")
    print()
