"""
cluster_dataset.py

Groups RECAST dataset records into clusters based on the constraint categories
already present in the `added_constraint` field, then writes tagged output files.

Constraint categories found in RECAST data:
    Length, Keyword, Format, Style, Background Info, Helpfulness, Topic,
    Factuality, Example, Language, Numerical Constraints, Strat_With, End_With,
    Tone, Situation, Role Playing, Emotion, No_Commas, All_Lower

Each record is tagged with:
    constraint_categories  : sorted list of all categories present (multi-label)
    num_constraint_types   : number of distinct categories
    primary_cluster        : dominant category (most constraints in it);
                             "mixed" if two or more categories tie for first
    cluster_label          : "+"-joined sorted categories
                             (e.g. "Format+Length+Style")

Supports both dataset schemas found in RECAST:
  Schema A (test splits)  — keys: id, prompt/input, response/output,
                            added_constraint (dict of category → list[str])
  Schema B (instruction)  — keys: instruction, input, output, id,
                            added_constraint (dict)

Usage:
    python3 cluster_dataset.py [--input PATH] [--output PATH]

    Defaults:
        input  : datasets/RECAST-30K.json
        output : datasets/recast_clustered.json
                 datasets/cluster_summary.json
"""

import argparse
import json
import os
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Normalise category names to a clean, consistent form
# ---------------------------------------------------------------------------

CATEGORY_ALIASES = {
    "strat_with":           "Start_With",
    "start_with":           "Start_With",
    "end_with":             "End_With",
    "no_commas":            "No_Commas",
    "all_lower":            "All_Lower",
    "numerical constraints": "Numerical_Constraints",
    "numerical_constraints": "Numerical_Constraints",
    "background info":      "Background_Info",
    "background_info":      "Background_Info",
    "role playing":         "Role_Playing",
    "role_playing":         "Role_Playing",
}


def normalise_category(raw: str) -> str:
    """Lowercase-strip, apply aliases, then title-case."""
    key = raw.strip().lower()
    if key in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[key]
    # Title-case with underscores preserved
    return raw.strip().replace(" ", "_")


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------

def extract_constraint_categories(record: dict) -> Counter:
    """
    Return a Counter of {category: num_constraints_in_that_category}.

    Handles:
      - added_constraint as dict  → {category: [str, ...]}
      - added_constraint as list  → [{"type": ..., ...}, ...]
      - constraints field as list → [{"type": ...}, ...]
      - no constraint field       → empty Counter
    """
    cat_counter: Counter = Counter()

    ac = record.get("added_constraint") or record.get("constraints")

    if ac is None:
        return cat_counter

    if isinstance(ac, str):
        try:
            ac = json.loads(ac)
        except json.JSONDecodeError:
            return cat_counter

    if isinstance(ac, dict):
        for raw_cat, items in ac.items():
            cat = normalise_category(raw_cat)
            count = len(items) if isinstance(items, list) else 1
            cat_counter[cat] += count

    elif isinstance(ac, list):
        for item in ac:
            if isinstance(item, dict):
                raw = item.get("type", item.get("constraint_type", "other"))
            else:
                raw = str(item)
            cat_counter[normalise_category(raw)] += 1

    return cat_counter


def assign_clusters(record: dict) -> dict:
    """Add cluster tags to a record in-place and return it."""
    cat_counter = extract_constraint_categories(record)

    if not cat_counter:
        categories = ["unconstrained"]
        primary = "unconstrained"
    else:
        categories = sorted(cat_counter.keys())
        max_count = max(cat_counter.values())
        top_cats = [c for c, n in cat_counter.items() if n == max_count]
        primary = top_cats[0] if len(top_cats) == 1 else "mixed"

    record["constraint_categories"] = categories
    record["num_constraint_types"] = len(categories)
    record["primary_cluster"] = primary
    record["cluster_label"] = "+".join(categories)
    return record


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list:
    """Load a JSON array or JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"File is empty: {path}")

    if content.startswith("version https://git-lfs.github.com"):
        raise ValueError(
            f"'{path}' is a Git LFS pointer — the actual file has not been downloaded.\n"
            "Place the real RECAST-30K.json in the datasets/ folder and retry."
        )

    # Try JSON first
    try:
        loaded = json.loads(content)
        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, dict):
            for key in ("data", "instances", "examples", "records"):
                if key in loaded and isinstance(loaded[key], list):
                    return loaded[key]
            return [loaded]
    except json.JSONDecodeError:
        pass

    # JSONL fallback
    records = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def build_summary(records: list) -> dict:
    primary_counter: Counter = Counter()
    label_counter:   Counter = Counter()
    category_counter: Counter = Counter()
    level_by_cluster: dict = defaultdict(Counter)
    type_count_dist:  Counter = Counter()

    for r in records:
        pc    = r.get("primary_cluster", "unknown")
        cl    = r.get("cluster_label", "unknown")
        cats  = r.get("constraint_categories", [])
        level = r.get("difficulty_level", r.get("added_constraint_num", "unknown"))
        ntypes = r.get("num_constraint_types", 0)

        primary_counter[pc] += 1
        label_counter[cl] += 1
        level_by_cluster[pc][str(level)] += 1
        type_count_dist[ntypes] += 1
        for cat in cats:
            category_counter[cat] += 1

    total = len(records)
    return {
        "total_records": total,
        "by_primary_cluster": {
            k: {
                "count": v,
                "pct": round(100 * v / total, 2),
            }
            for k, v in sorted(primary_counter.items(), key=lambda x: -x[1])
        },
        "category_presence": {
            k: {"count": v, "pct": round(100 * v / total, 2)}
            for k, v in sorted(category_counter.items(), key=lambda x: -x[1])
        },
        "num_constraint_types_distribution": {
            k: v for k, v in sorted(type_count_dist.items())
        },
        "top_cluster_labels": {
            k: {"count": v, "pct": round(100 * v / total, 2)}
            for k, v in sorted(label_counter.items(), key=lambda x: -x[1])[:20]
        },
    }


def print_summary(summary: dict):
    total = summary["total_records"]
    print(f"\n{'='*62}")
    print(f"  RECAST Constraint-Type Clustering")
    print(f"  Total records: {total:,}")
    print(f"{'='*62}")

    print(f"\n  Primary cluster distribution:")
    print(f"  {'Cluster':<25} {'Count':>8} {'%':>7}")
    print(f"  {'-'*42}")
    for cluster, info in summary["by_primary_cluster"].items():
        print(f"  {cluster:<25} {info['count']:>8,} {info['pct']:>6.1f}%")

    print(f"\n  Category presence (records containing each category):")
    print(f"  {'Category':<25} {'Count':>8} {'%':>7}")
    print(f"  {'-'*42}")
    for cat, info in summary["category_presence"].items():
        print(f"  {cat:<25} {info['count']:>8,} {info['pct']:>6.1f}%")

    print(f"\n  Distinct constraint-type counts per record:")
    for n, count in summary["num_constraint_types_distribution"].items():
        print(f"    {n} type(s): {count:,} records")

    print(f"\n  Top-10 cluster label combinations:")
    print(f"  {'Label':<50} {'Count':>7}")
    print(f"  {'-'*59}")
    for label, info in list(summary["top_cluster_labels"].items())[:10]:
        print(f"  {label:<50} {info['count']:>7,}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cluster RECAST dataset by constraint type.")
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "RECAST-30K.json"),
        help="Path to input JSON/JSONL dataset file",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "recast_clustered.json"),
        help="Path for tagged output JSON file",
    )
    args = parser.parse_args()

    summary_path = os.path.splitext(args.output)[0] + "_summary.json"

    print(f"Loading dataset from: {args.input}")
    records = load_dataset(args.input)
    print(f"Loaded {len(records):,} records.")

    print("Assigning cluster tags...")
    for record in records:
        assign_clusters(record)

    summary = build_summary(records)
    print_summary(summary)

    print(f"Writing tagged dataset to:  {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Writing cluster summary to: {summary_path}")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
