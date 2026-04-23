"""Emit a compact summary of a (cleaned) RECAST JSONL file.

Reports record count, field coverage, text-length distributions, and the
constraint-category distribution across kept records.
"""

import argparse
import json
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path

TEXT_FIELDS = ("winner_prompt", "response_of_winner_prompt")
CONSTRAINT_FIELD = "added_constraint"
CONSTRAINT_NUM_FIELD = "added_constraint_num"


def iter_jsonl(path: Path):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".jsonl"))
            with zf.open(name) as f:
                for raw in f:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
    else:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def describe(xs: list[int]) -> str:
    if not xs:
        return "n/a"
    return (f"min={min(xs)}  median={int(statistics.median(xs))}  "
            f"mean={statistics.mean(xs):.1f}  max={max(xs)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    path = Path(args.input)
    total = 0
    prompt_word_counts: list[int] = []
    response_word_counts: list[int] = []
    field_presence: Counter = Counter()
    cat_record_counts: Counter = Counter()
    total_constraints = 0
    all_top_level_fields: Counter = Counter()

    for rec in iter_jsonl(path):
        total += 1
        for k in rec:
            all_top_level_fields[k] += 1
        for f in TEXT_FIELDS:
            val = rec.get(f)
            if isinstance(val, str) and val.strip():
                field_presence[f] += 1
        prompt = rec.get("winner_prompt") or ""
        response = rec.get("response_of_winner_prompt") or ""
        if prompt:
            prompt_word_counts.append(len(prompt.split()))
        if response:
            response_word_counts.append(len(response.split()))
        constraints = rec.get(CONSTRAINT_FIELD) or {}
        if isinstance(constraints, dict):
            for cat, items in constraints.items():
                if isinstance(items, list) and items:
                    cat_record_counts[cat] += 1
                    total_constraints += len(items)

    lines: list[str] = []
    push = lines.append

    push(f"# Dataset summary")
    push(f"Source : {path.name}")
    push(f"Size   : {path.stat().st_size / (1024**2):.1f} MB")
    push(f"")
    push(f"Total records                               : {total:,}")
    push(f"Total constraint entries (across all records): {total_constraints:,}")
    push(f"Mean constraints per record                 : "
         f"{total_constraints / max(total, 1):.2f}")
    push(f"")
    push(f"## Top-level fields present (count of records where key is present)")
    for k, v in all_top_level_fields.most_common():
        push(f"  {k:<32}: {v:>7,}")
    push(f"")
    push(f"## Text-field coverage (non-empty)")
    for f in TEXT_FIELDS:
        push(f"  {f:<32}: {field_presence.get(f, 0):>7,}")
    push(f"")
    push(f"## Word-length distributions")
    push(f"  winner_prompt                 : {describe(prompt_word_counts)}")
    push(f"  response_of_winner_prompt     : {describe(response_word_counts)}")
    push(f"")
    push(f"## Constraint categories (records that use the category)")
    cat_total = sum(cat_record_counts.values())
    for cat, cnt in sorted(cat_record_counts.items(), key=lambda x: -x[1]):
        share = 100 * cnt / max(cat_total, 1)
        push(f"  {cat:<20}: {cnt:>7,}  ({share:5.1f}% of category mentions)")

    report = "\n".join(lines)
    print(report)
    if args.report:
        Path(args.report).write_text(report + "\n", encoding="utf-8")
        print(f"\n[wrote {args.report}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
