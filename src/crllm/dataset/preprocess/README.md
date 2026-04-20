# Preprocess

---

## Overview

`recast_data_cleaning.py` is the data preprocessing script for the RECAST-30K dataset. It takes the raw RECAST JSONL file, runs a 9-step cleaning and validation pipeline, and writes a cleaned JSONL ready to hand off to the augmentation stage.

---

## Installation

```bash
pip install langdetect emoji datasketch
```

> **NLTK** is an optional dependency. If available, its full English stopword list is used; otherwise the script falls back to a built-in list with no loss of functionality.

---

## Usage

```bash
$ chmod +x run_preprocess.sh
$ ./run_preprocess.sh
```
OR
```bash
python preprocess.py \
        --input RECAST-30K.jsonl \
        --output recast_30k_clean.jsonl \
        --min_length 15 \
        --dedup_threshold 0.85 \
        --imbalance_threshold 0.5
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to the raw RECAST JSONL file |
| `--output` | *(required)* | Path for the cleaned output JSONL |
| `--min_length` | `15` | Minimum word count in `winner_prompt` after cleaning |
| `--dedup_threshold` | `0.85` | Jaccard similarity threshold for near-duplicate detection (0–1) |
| `--imbalance_threshold` | `0.5` | Warn if any constraint category falls below this fraction of the expected uniform share |

---

## Input Schema

The script is matched to the actual RECAST-30K field names:

| Field | Type | Description |
|---|---|---|
| `winner_prompt` | `str` | The instruction/prompt — used as the training input |
| `response_of_winner_prompt` | `str` | The winning response — used as the training target |
| `added_constraint` | `dict[str, list[str]]` | Constraint descriptions grouped by category |
| `added_constraint_num` | `int` | Total constraint count (must match actual count) |
| `rule_evaluate_dict` | `dict` | Rule-based validator specs used for alignment checking |
| `id` | `str` | Unique record identifier |

**Constraint categories** in the data:

- LLM-generated (soft): `Style`, `Topic`, `Language`, `Helpfulness`
- Rule-based (validated): `Length`, `Keyword`, `End_With`, `Strat_With`, `Format`, `Tone`, `Numerical`

---

## Cleaning Pipeline

Records pass through 9 sequential steps. A record is dropped the moment it fails any step.

### Step i — Language Detection
Runs `langdetect` on `winner_prompt`. Any non-English record is dropped. Requires the `langdetect` package; if unavailable, this step is skipped.

### Step ii — HTML Removal
Strips HTML tags (`<b>`, `</b>`, `<!-- ... -->`, etc.) and decodes HTML entities (`&amp;`, `&#39;`, etc.) from both `winner_prompt` and `response_of_winner_prompt`.

### Step iii — Emoji & Character Cleaning
- Replaces emoji characters with whitespace
- Removes control characters, zero-width/invisible unicode, and surrogate pairs
- Collapses runs of 4+ consecutive punctuation/symbol characters

### Step iv — Stopword Removal
Strips common English stopwords (articles, conjunctions, prepositions) from `winner_prompt` **only**. The response is never touched — it must stay fluent since the model learns to produce it. Applied **last** in the pipeline so it does not interfere with the length quality gate.

### Step v — Quality Gates
Two checks on the cleaned `winner_prompt`:
- **Minimum length**: fewer than `--min_length` words → dropped
- **Printability**: fewer than 85% printable characters → dropped (catches garbled encodings)

Also drops any record where `response_of_winner_prompt` is empty after cleaning.

### Step vi — Constraint Field Integrity
Validates the `added_constraint` dict:
- Must exist and be a non-empty dict
- Every category must map to a non-empty list of non-empty strings
- `added_constraint_num` must exactly match the actual total count of constraint descriptions across all categories

### Step vii — Instruction-Response Alignment
Re-runs the rule-based validators from `rule_evaluate_dict` against `response_of_winner_prompt` (not the reference response stored in `func_input[0]`, which is metadata from a different model). Validators implemented:

| Validator function | What it checks |
|---|---|
| `evaluate_word_length` | Word count within `[min, max]` |
| `evaluate_sentence_length` | Sentence count ≈ target (±3 tolerance) |
| `evaluate_keyword` | Each required keyword appears the required number of times |
| `evaluate_start_with` | Response starts with the specified word |
| `evaluate_end_with` | Response ends with the specified word |
| `check_english_uppercase` | At most 1 all-caps word in the response |
| `check_english_lowercase` | Response is entirely lowercase (≤2 uppercase chars tolerated) |
| `contains_no_punctuation` | Response contains no commas |
| `evaluate_format` | Structural format check (passthrough — format type not encoded in func_input) |

### Step viii — Deduplication
Two-level dedup on `winner_prompt`:
1. **Exact** — SHA-256 fingerprint of lowercased, whitespace-normalised prompt
2. **Fuzzy** — MinHash LSH with character 5-shingles at the configured Jaccard threshold (requires `datasketch`; falls back to exact-only if unavailable)

### Step ix — Distribution Audit
After processing, counts how many kept records belong to each constraint category. Logs a warning for any category whose share falls below `--imbalance_threshold × (1 / N_categories)`. Does not drop records — informational only, to flag skew before augmentation.

---

## Output

The output JSONL has the same schema as the input. Fields are unchanged except:
- `winner_prompt` — HTML/emoji/char cleaned, then stopwords removed
- `response_of_winner_prompt` — HTML/emoji/char cleaned only (no stopword removal)
- Other string fields (`prompt_winner`, `response_winner`) — HTML/emoji/char cleaned

Each output record is guaranteed to:
- Be in English
- Have a non-empty, printable prompt of at least `--min_length` words
- Have a non-empty response
- Have a valid, internally-consistent `added_constraint` dict
- Pass all rule-based constraint validators in `rule_evaluate_dict`
- Be unique (not a near-duplicate of any earlier record)

---

## Sample Output Log

```
09:12:01 [INFO] =================================================================
09:12:01 [INFO]   RECAST Full Cleaning Pipeline — STAT 453 Team 15
09:12:01 [INFO] =================================================================
09:12:01 [INFO]   Input              : recast_30k.jsonl
09:12:01 [INFO]   Output             : recast_30k_clean.jsonl
09:12:01 [INFO]   Min prompt length  : 15 words
09:12:01 [INFO]   Dedup threshold    : 0.85
09:12:01 [INFO]   langdetect         : ✓
09:12:01 [INFO]   MinHash dedup      : ✓
...
09:14:33 [INFO]   CLEANING COMPLETE — Summary
09:14:33 [INFO]   Total records processed        :   30,000
09:14:33 [INFO]   Records kept (→ augmentation)  :   27,412  (91.4%)
09:14:33 [INFO]   Skipped — non-English          :       83
09:14:33 [INFO]   Skipped — too short            :      142
09:14:33 [INFO]   Skipped — bad constraint       :      210
09:14:33 [INFO]   Skipped — alignment fail       :    1,847
09:14:33 [INFO]   Skipped — duplicate            :      306
...
09:14:33 [INFO]   Constraint category distribution (kept records):
09:14:33 [INFO]     Length               :  8,241  (22.3%)
09:14:33 [INFO]     Keyword              :  7,918  (21.4%)
09:14:33 [INFO]     Style                :  6,102  (16.5%)
...
```
*(Numbers are illustrative — actual counts depend on the dataset split.)*

---

## File Structure

```
recast_data_cleaning.py   ← this script
recast_30k.jsonl          ← raw input (from RECAST authors)
recast_30k_clean.jsonl    ← cleaned output → pass to augmentation
```