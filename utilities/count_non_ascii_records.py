"""Audit non-ASCII content in a (cleaned) RECAST JSONL file.

Reports:
  * total records scanned
  * records containing >=1 non-ASCII char across the text fields
  * per-field breakdown (which fields carry the non-ASCII content)
  * total non-ASCII char count
  * top-20 most common non-ASCII chars (with Unicode name and category)

Accepts a .jsonl or .jsonl.zip path.
"""

import argparse
import json
import sys
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

TEXT_FIELDS = (
    "winner_prompt",
    "response_of_winner_prompt",
    "prompt_winner",
    "response_winner",
)


def iter_jsonl(path: Path):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".jsonl"))
            with zf.open(name) as f:
                for raw in f:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    else:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def non_ascii_chars(text: str) -> list[str]:
    return [c for c in text if ord(c) > 0x7F]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="path to .jsonl or .jsonl.zip")
    ap.add_argument("--report", default=None,
                    help="optional path to write the full report; else stdout only")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: not found: {path}", file=sys.stderr)
        return 2

    total_records = 0
    records_with_any = 0
    field_hits: Counter = Counter()
    char_counts: Counter = Counter()
    total_non_ascii_chars = 0

    for rec in iter_jsonl(path):
        total_records += 1
        any_in_rec = False
        for field in TEXT_FIELDS:
            val = rec.get(field)
            if not isinstance(val, str):
                continue
            chars = non_ascii_chars(val)
            if chars:
                any_in_rec = True
                field_hits[field] += 1
                total_non_ascii_chars += len(chars)
                char_counts.update(chars)
        if any_in_rec:
            records_with_any += 1
        if total_records % 5000 == 0:
            print(f"...{total_records:,} scanned | {records_with_any:,} with non-ASCII",
                  file=sys.stderr)

    lines: list[str] = []
    push = lines.append

    push(f"# Non-ASCII audit")
    push(f"Source : {path.name}")
    push(f"")
    push(f"Total records scanned                         : {total_records:,}")
    push(f"Records with any non-ASCII character          : {records_with_any:,}  "
         f"({100 * records_with_any / max(total_records, 1):.1f}%)")
    push(f"Total non-ASCII characters across all records : {total_non_ascii_chars:,}")
    push(f"")
    push(f"## Records per field (records where THIS field contained non-ASCII)")
    for field in TEXT_FIELDS:
        if field_hits.get(field, 0):
            push(f"  {field:<30}: {field_hits[field]:>7,}")
    push(f"")
    push(f"## Top 20 non-ASCII characters")
    push(f"{'char':<6} {'codepoint':<10} {'count':>10}  {'category':<4}  name")
    for ch, cnt in char_counts.most_common(20):
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = "<unnamed>"
        cat = unicodedata.category(ch)
        push(f"{ch!r:<6} U+{ord(ch):04X}    {cnt:>10,}  {cat:<4}  {name}")

    report = "\n".join(lines)
    print(report)

    if args.report:
        Path(args.report).write_text(report + "\n", encoding="utf-8")
        print(f"\n[wrote {args.report}]", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
