# Utilities

Small audit / reporting scripts that run *after* the preprocessing pipeline
(`src/crllm/dataset/preprocess/preprocess.py`) to verify and document what the
cleaned dataset actually looks like.

---

## Scripts

| Script | Purpose |
|---|---|
| `count_non_ascii_records.py` | Count records in a cleaned JSONL that still contain any non-ASCII character. Reports per-field hits, total count, and the top-20 most frequent non-ASCII characters (with Unicode name + category). |
| `dataset_summary.py` | Overall post-cleaning stats: record count, top-level field coverage, word-length distributions for prompt/response, and the distribution of constraint categories. |

### Usage

```bash
# Non-ASCII audit
python utilities/count_non_ascii_records.py \
    --input datasets/recast_30k_clean.jsonl \
    --report utilities/non_ascii_report.txt

# Dataset summary
python utilities/dataset_summary.py \
    --input datasets/recast_30k_clean.jsonl \
    --report utilities/dataset_summary.txt
```

Both scripts accept a `.jsonl` or `.jsonl.zip` input.

Results from the latest run are saved alongside the scripts as
`non_ascii_report.txt` and `dataset_summary.txt`.

---

## Cleaned dataset — current state

Source file: `datasets/recast_30k_clean.jsonl` (607 MB)

### Size

| | Count |
|---|---:|
| Total records | **27,349** |
| Total constraint entries across records | 375,080 |
| Mean constraints per record | 13.71 |

### Field schema (all fields present in every record)

`prompt_winner`, `winner_prompt`, `response_winner`, `winner_response`,
`response_of_winner_prompt`, `id`, `added_constraint`, `added_constraint_num`,
`added_constraint_from_LLM`, `added_constraint_from_rule`,
`added_constraint_num_from_LLM`, `added_constraint_num_from_rule`,
`rule_evaluate_dict`.

The cleaning pipeline only rewrites the four text fields — `winner_prompt`,
`response_of_winner_prompt`, `prompt_winner`, `response_winner` — and
validates the `added_constraint` / `added_constraint_num` fields. All other
keys pass through untouched.

### Text length (word counts)

| Field | min | median | mean | max |
|---|---:|---:|---:|---:|
| `winner_prompt` (post-stopword-removal) | 10 | 106 | 109.3 | 577 |
| `response_of_winner_prompt` | 1 | 220 | 267.4 | 5,207 |

Note: the 15-word `--min_length` gate is checked *before* stopword removal,
so final prompt lengths can sit below 15 once stopwords are stripped.

### Constraint-category coverage (records using each category)

| Category | Records |
|---|---:|
| Length | 26,787 |
| Keyword | 25,494 |
| Style | 20,366 |
| Background Info | 19,815 |
| Format | 18,880 |
| Helpfulness | 17,479 |
| Factuality | 16,759 |
| Strat_With | 16,700 |
| End_With | 15,491 |
| Topic | 15,275 |
| Language | 13,183 |
| Example | 12,163 |
| Numerical Constraints | 11,260 |
| Tone | 6,716 |
| Situation | 4,586 |
| Role Playing | 2,442 |
| Emotion | 2,268 |

### Non-ASCII content after cleaning

| | Count |
|---|---:|
| Records containing ≥1 non-ASCII character | **2,670 (9.8%)** |
| Total non-ASCII characters across all records | 127,275 |
| Records where `winner_prompt` has non-ASCII | 1,236 |
| Records where `response_of_winner_prompt` has non-ASCII | 2,639 |

**Top surviving non-ASCII characters** — all are Unicode **letters**
(category `Ll` / `Lo`), which confirms the symbol-normalisation filter is
working: typographic symbols got mapped to ASCII or dropped, while letters
from other scripts (French accents, Russian, Arabic, etc.) were preserved
as intended.

| Char | Codepoint | Count | Category | Name |
|---|---|---:|---|---|
| `é` | U+00E9 | 17,750 | Ll | LATIN SMALL LETTER E WITH ACUTE |
| `ó` | U+00F3 | 12,156 | Ll | LATIN SMALL LETTER O WITH ACUTE |
| `í` | U+00ED | 6,401 | Ll | LATIN SMALL LETTER I WITH ACUTE |
| `á` | U+00E1 | 5,966 | Ll | LATIN SMALL LETTER A WITH ACUTE |
| `а` | U+0430 | 2,182 | Ll | CYRILLIC SMALL LETTER A |
| `о` | U+043E | 2,044 | Ll | CYRILLIC SMALL LETTER O |
| `и` | U+0438 | 2,009 | Ll | CYRILLIC SMALL LETTER I |
| `ñ` | U+00F1 | 1,791 | Ll | LATIN SMALL LETTER N WITH TILDE |
| `à` | U+00E0 | 1,777 | Ll | LATIN SMALL LETTER A WITH GRAVE |
| `ا` | U+0627 | 1,749 | Lo | ARABIC LETTER ALEF |
| `е` | U+0435 | 1,720 | Ll | CYRILLIC SMALL LETTER IE |
| `è` | U+00E8 | 1,642 | Ll | LATIN SMALL LETTER E WITH GRAVE |
| `ú` | U+00FA | 1,554 | Ll | LATIN SMALL LETTER U WITH ACUTE |
| `н` | U+043D | 1,533 | Ll | CYRILLIC SMALL LETTER EN |
| `т` | U+0442 | 1,438 | Ll | CYRILLIC SMALL LETTER TE |
| `ä` | U+00E4 | 1,310 | Ll | LATIN SMALL LETTER A WITH DIAERESIS |
| `р` | U+0440 | 1,256 | Ll | CYRILLIC SMALL LETTER ER |
| `с` | U+0441 | 1,241 | Ll | CYRILLIC SMALL LETTER ES |
| `ü` | U+00FC | 1,077 | Ll | LATIN SMALL LETTER U WITH DIAERESIS |
| `ل` | U+0644 | 1,039 | Lo | ARABIC LETTER LAM |

### How to regenerate these numbers

Re-run both scripts with the same `--report` paths and the tables above will
refresh from the new `.txt` outputs.
