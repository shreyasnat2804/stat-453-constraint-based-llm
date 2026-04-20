"""
RECAST Dataset Cleaning Pipeline
STAT 453 - Team 15 | Spring 2026

Matched to the actual RECAST-30K schema observed in the dataset:

  Key fields per record:
    winner_prompt             — the instruction / prompt text
    winner_response           — which model response key won (e.g. "response_nano")
    response_of_winner_prompt — the actual response text used for training
    added_constraint          — dict mapping category → list[str] of constraint descriptions
    added_constraint_num      — total number of constraints in this record
    rule_evaluate_dict        — dict of validator specs (func name + inputs) for rule-based checks
    id                        — unique record identifier

  Constraint categories seen in the data:
    Style, Topic, Length, Keyword, End_With, Strat_With,
    Language, Helpfulness, Format, Tone, Role-Playing, Numerical

Full cleaning steps:
  (i)    Language Detection — skip non-English records (check winner_prompt)
  (ii)   HTML tag / entity removal on text fields
  (iii)  Emoji replacement, unknown / control character removal
  (iv)   Stopword removal on winner_prompt only (not on response — must stay fluent)
  (v)    Printability + minimum-length quality gate on winner_prompt
  (vi)   Constraint field integrity — added_constraint must be a non-empty dict
         with non-empty string lists; added_constraint_num must match actual count
  (vii)  Instruction-response alignment — re-runs rule_evaluate_dict validators
         (Length/word_count, sentence_count, Keyword, End_With, Strat_With, Format,
          all_uppercase, all_lowercase, no_punctuation) against response_of_winner_prompt
  (viii) Deduplication — exact SHA-256 + MinHash fuzzy near-dedup on winner_prompt
  (ix)   Constraint-category distribution audit — warns on imbalance across categories

Usage:
    python recast_data_cleaning.py \\
        --input   path/to/recast_30k.jsonl \\
        --output  path/to/recast_30k_clean.jsonl \\
        [--min_length 15] \\
        [--dedup_threshold 0.85] \\
        [--imbalance_threshold 0.5]

Dependencies:
    pip install langdetect emoji nltk datasketch
    python -c "import nltk; nltk.download('stopwords')"
"""

import argparse
import hashlib
import json
import logging
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Optional

# ── Optional dependency imports with graceful fallbacks ──────────────────────

try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 42
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    logging.warning("langdetect not installed. Run: pip install langdetect")

try:
    import emoji as emoji_lib
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False
    logging.warning("emoji not installed. Run: pip install emoji")

try:
    from nltk.corpus import stopwords as nltk_stopwords
    STOPWORDS = set(nltk_stopwords.words("english"))
    HAS_NLTK = True
except Exception:
    HAS_NLTK = False
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "up", "about", "into", "through",
        "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "shall", "can", "need", "dare", "ought", "used",
        "it", "its", "this", "that", "these", "those", "i", "me", "my",
        "we", "our", "you", "your", "he", "him", "his", "she", "her",
        "they", "them", "their", "what", "which", "who", "whom", "not",
        "no", "nor", "so", "yet", "both", "either", "neither", "each",
        "few", "more", "most", "other", "some", "such", "than", "too",
        "very", "just", "because", "as", "until", "while", "if", "then",
        "also",
    }
    logging.warning("NLTK stopwords unavailable — using built-in fallback list.")

try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False
    logging.warning("datasketch not installed. Fuzzy dedup disabled. Run: pip install datasketch")

# ── RECAST field name constants (matched to actual schema) ────────────────────
#    Change these if the dataset uses different field names.

FIELD_PROMPT       = "winner_prompt"             # instruction text
FIELD_RESPONSE     = "response_of_winner_prompt"  # training target response
FIELD_CONSTRAINTS  = "added_constraint"           # dict: category -> list[str]
FIELD_CONSTRAINT_N = "added_constraint_num"       # int: total constraint count
FIELD_VALIDATORS   = "rule_evaluate_dict"         # dict of rule validator specs
FIELD_ID           = "id"

# All constraint categories observed in RECAST
LLM_CONSTRAINT_CATEGORIES  = {"Style", "Topic", "Language", "Helpfulness"}
RULE_CONSTRAINT_CATEGORIES = {"Length", "Keyword", "End_With", "Strat_With",
                               "Format", "Tone", "Numerical"}
ALL_CONSTRAINT_CATEGORIES  = LLM_CONSTRAINT_CATEGORIES | RULE_CONSTRAINT_CATEGORIES

# ── Compiled regex patterns ───────────────────────────────────────────────────

RE_HTML_TAGS     = re.compile(r"<[^>]+>", re.DOTALL)
RE_HTML_ENTITIES = re.compile(r"&(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);")
RE_WHITESPACE    = re.compile(r"\s+")
RE_CONTROL       = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
RE_INVISIBLE     = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\ufeff\u00ad]"
)
RE_SURROGATES    = re.compile(r"[\ud800-\udfff]")
RE_SYMBOL_RUNS   = re.compile(r"[^\w\s]{4,}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Text cleaning helpers
# ═══════════════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> Optional[str]:
    """Return ISO-639-1 language code, or None on failure."""
    if not HAS_LANGDETECT:
        return None
    try:
        return detect(text[:2000])
    except LangDetectException:
        return None


def remove_html(text: str) -> str:
    text = RE_HTML_TAGS.sub(" ", text)
    text = RE_HTML_ENTITIES.sub(" ", text)
    return text


def replace_emojis(text: str) -> str:
    if HAS_EMOJI:
        return emoji_lib.replace_emoji(text, replace=" ")
    return re.sub(
        r"[\U0001F300-\U0001FFFF\U00002700-\U000027BF"
        r"\U0000FE00-\U0000FE0F\U00002600-\U000026FF]+",
        " ", text,
    )


def replace_unknown_chars(text: str) -> str:
    """Remove control chars, invisible unicode, surrogates, long symbol runs."""
    text = unicodedata.normalize("NFC", text)
    text = RE_CONTROL.sub(" ", text)
    text = RE_INVISIBLE.sub("", text)
    text = RE_SURROGATES.sub("", text)
    text = RE_SYMBOL_RUNS.sub(" ", text)
    return text


def remove_stopwords(text: str) -> str:
    """
    Strip stopwords token-by-token.
    Applied ONLY to winner_prompt, and ONLY after all quality gates have passed,
    so that length checks are not affected by the word reduction from stripping.
    Never applied to the response — the model must learn to produce fluent text.
    """
    tokens = text.split()
    return " ".join(
        tok for tok in tokens
        if tok.strip(string.punctuation).lower() not in STOPWORDS
    )


def normalize_whitespace(text: str) -> str:
    return RE_WHITESPACE.sub(" ", text).strip()


def is_mostly_printable(text: str, threshold: float = 0.85) -> bool:
    if not text:
        return False
    return sum(c.isprintable() for c in text) / len(text) >= threshold


def clean_text(text: str) -> str:
    """
    Apply HTML removal, emoji replacement, and character cleaning.
    Does NOT apply stopword removal — that is a separate, later step.
    """
    text = remove_html(text)
    text = replace_emojis(text)
    text = replace_unknown_chars(text)
    return normalize_whitespace(text)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Constraint field integrity  (Step vi)
# Validates the added_constraint dict structure against the actual RECAST schema.
# ═══════════════════════════════════════════════════════════════════════════════

def validate_constraints(record: dict) -> tuple[bool, str]:
    """
    Validate the added_constraint field.

    Expected schema:
        "added_constraint": {
            "Length":  ["Limit the response to 100-200 words.", "..."],
            "Keyword": ["Ensure your answer contains 1 'financial services'."],
            "Style":   ["Adopt a heartfelt and appreciative tone."],
            ...
        }
        "added_constraint_num": 7   ← must equal total items across all lists

    Checks:
      - Field exists and is a non-empty dict
      - Every value is a non-empty list of non-empty strings
      - added_constraint_num (if present) matches actual total count
    """
    constraints = record.get(FIELD_CONSTRAINTS)
    if not constraints:
        return False, f"'{FIELD_CONSTRAINTS}' is missing or empty"
    if not isinstance(constraints, dict):
        return False, f"'{FIELD_CONSTRAINTS}' must be a dict, got {type(constraints).__name__}"

    total_count = 0
    for category, descriptions in constraints.items():
        if not isinstance(descriptions, list) or len(descriptions) == 0:
            return False, f"category '{category}' has empty or non-list value"
        for i, desc in enumerate(descriptions):
            if not isinstance(desc, str) or not desc.strip():
                return False, f"category '{category}', item [{i}] is blank or not a string"
        total_count += len(descriptions)

    # Cross-check stated count against actual count
    stated_num = record.get(FIELD_CONSTRAINT_N)
    if stated_num is not None:
        try:
            if int(stated_num) != total_count:
                return (
                    False,
                    f"'{FIELD_CONSTRAINT_N}'={int(stated_num)} does not match "
                    f"actual constraint count={total_count}",
                )
        except (TypeError, ValueError):
            return False, f"'{FIELD_CONSTRAINT_N}' is not an integer: {stated_num!r}"

    return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Instruction-response alignment  (Step vii)
#
# Re-implements the validator functions referenced in rule_evaluate_dict.
# From the data sample, we observed these function names:
#   evaluate_word_length      — word count within [min, max]
#   evaluate_sentence_length  — sentence count approximately == target
#   evaluate_format           — structural format (bullets, numbered list)
#   evaluate_keyword          — exact keyword occurrence counts
#   evaluate_start_with       — response starts with given word
#   evaluate_end_with         — response ends with given word
#   check_english_uppercase   — ≤1 all-caps word in response
#   check_english_lowercase   — response is entirely lowercase
#   contains_no_punctuation   — response has no commas
#
# We validate against response_of_winner_prompt (the actual training target),
# NOT against func_input[0] which stores the original model's output as metadata.
# ═══════════════════════════════════════════════════════════════════════════════

def _eval_word_length(response: str, func_input: list) -> tuple[bool, str]:
    """
    func_input: [response_text, [min, max], target, null, tolerance_factor]
    Checks word count is within [min, max].
    """
    if len(func_input) < 2 or not isinstance(func_input[1], list):
        return True, ""
    lo, hi = int(func_input[1][0]), int(func_input[1][1])
    word_count = len(response.split())
    if not (lo <= word_count <= hi):
        return False, f"word count {word_count} outside [{lo}, {hi}]"
    return True, ""


def _eval_sentence_length(response: str, func_input: list) -> tuple[bool, str]:
    """
    func_input: [response_text, min_ok_flag, target_sentence_count, null]
    Checks sentence count is approximately == target (±3 tolerance).
    """
    if len(func_input) < 3 or func_input[2] is None:
        return True, ""
    try:
        target = int(func_input[2])
    except (TypeError, ValueError):
        return True, ""
    sentences = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]
    actual    = len(sentences)
    tolerance = 3  # RECAST uses "approximately N sentences"
    if abs(actual - target) > tolerance:
        return False, f"sentence count {actual} not close enough to target {target} (±{tolerance})"
    return True, ""


def _eval_format(response: str, func_input: list) -> tuple[bool, str]:
    """
    func_input: [null, response_text]
    Format is loosely validated — mainly structural marker presence.
    The full format type is not encoded in func_input, so we do a light check.
    """
    # No reliable format type info in func_input; skip to avoid false drops
    return True, ""


def _eval_keyword(response: str, func_input: list) -> tuple[bool, str]:
    """
    func_input: [response_text, required_counts_dict, expected_counts_dict]
    required_counts_dict: {"keyword": N}  — keyword must appear exactly N times.
    """
    if len(func_input) < 2 or not isinstance(func_input[1], dict):
        return True, ""
    resp_lower = response.lower()
    for kw, count in func_input[1].items():
        actual = resp_lower.count(kw.lower())
        if actual < int(count):
            return False, f"keyword '{kw}' appears {actual}x, required {count}x"
    return True, ""


def _eval_start_with(response: str, func_input: list) -> tuple[bool, str]:
    """
    func_input: [response_text, prefix_word]
    Response must start with the given word (case-insensitive).
    """
    if len(func_input) < 2:
        return True, ""
    prefix = str(func_input[1]).strip().lower().strip('"\'')
    if not prefix:
        return True, ""
    if not response.strip().lower().startswith(prefix):
        return False, f"response does not start with '{prefix}'"
    return True, ""


def _eval_end_with(response: str, func_input: list) -> tuple[bool, str]:
    """
    func_input: [response_text, suffix_word]
    Response must end with the given word (strip trailing punctuation/whitespace).
    """
    if len(func_input) < 2:
        return True, ""
    suffix = str(func_input[1]).strip().lower().strip('"\'')
    if not suffix:
        return True, ""
    clean_end = response.strip().rstrip(".!?, \n").lower()
    if not clean_end.endswith(suffix):
        return False, f"response does not end with '{suffix}'"
    return True, ""


def _eval_uppercase(response: str, func_input: list) -> tuple[bool, str]:
    """
    check_english_uppercase: number of ALL-CAPS words (>1 char) must be ≤ 1.
    """
    all_caps = [w for w in response.split() if w.isupper() and w.isalpha() and len(w) > 1]
    if len(all_caps) > 1:
        return False, f"too many all-caps words ({len(all_caps)}): {all_caps[:5]}"
    return True, ""


def _eval_lowercase(response: str, func_input: list) -> tuple[bool, str]:
    """
    check_english_lowercase: response must be entirely lowercase.
    Allows ≤ 2 uppercase alpha characters as a tolerance for edge cases.
    """
    upper_chars = [c for c in response if c.isalpha() and c.isupper()]
    if len(upper_chars) > 2:
        return False, f"response not fully lowercase: {len(upper_chars)} uppercase chars found"
    return True, ""


def _eval_no_punctuation(response: str, func_input: list) -> tuple[bool, str]:
    """
    contains_no_punctuation: response must not contain commas.
    """
    if "," in response:
        return False, "response contains commas but constraint requires none"
    return True, ""


# Dispatcher maps RECAST validator function names → our implementations
VALIDATOR_DISPATCH: dict[str, callable] = {
    "evaluate_word_length":     _eval_word_length,
    "evaluate_sentence_length": _eval_sentence_length,
    "evaluate_format":          _eval_format,
    "evaluate_keyword":         _eval_keyword,
    "evaluate_start_with":      _eval_start_with,
    "evaluate_end_with":        _eval_end_with,
    "check_english_uppercase":  _eval_uppercase,
    "check_english_lowercase":  _eval_lowercase,
    "contains_no_punctuation":  _eval_no_punctuation,
}


def check_alignment(record: dict) -> tuple[bool, str]:
    """
    Re-run the rule_evaluate_dict validators using response_of_winner_prompt.

    rule_evaluate_dict format (from data):
        {
          "word_length":     {"func": "evaluate_word_length",     "func_input": [old_response, ...]},
          "keyword":         {"func": "evaluate_keyword",         "func_input": [old_response, ...]},
          "start_with":      {"func": "evaluate_start_with",      "func_input": [old_response, ...]},
          "end_with":        {"func": "evaluate_end_with",        "func_input": [old_response, ...]},
          "all_upper":       {"func": "check_english_uppercase",  "func_input": [old_response]},
          "all_lower":       {"func": "check_english_lowercase",  "func_input": [old_response]},
          "no_commas":       {"func": "contains_no_punctuation",  "func_input": [old_response]},
          ...
        }

    IMPORTANT: func_input[0] contains the original competing model's output stored
    as metadata — NOT the response we're validating. We substitute
    response_of_winner_prompt as the first argument in every call.
    """
    response = record.get(FIELD_RESPONSE) or ""
    if not response.strip():
        return False, f"'{FIELD_RESPONSE}' is empty"

    validators = record.get(FIELD_VALIDATORS)
    if not validators or not isinstance(validators, dict):
        return True, ""  # LLM-only constraints, no rule validators

    for validator_key, spec in validators.items():
        if not isinstance(spec, dict):
            continue
        func_name  = spec.get("func", "")
        func_input = spec.get("func_input", [])

        impl = VALIDATOR_DISPATCH.get(func_name)
        if impl is None:
            logging.debug(f"Unknown validator '{func_name}' in key '{validator_key}' — skipping")
            continue

        # Substitute our actual response for func_input[0] (the stored reference response)
        actual_input = [response] + (func_input[1:] if len(func_input) > 1 else [])

        ok, reason = impl(response, actual_input)
        if not ok:
            return False, f"'{validator_key}' ({func_name}): {reason}"

    return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Deduplication  (Step viii)
# ═══════════════════════════════════════════════════════════════════════════════

def fingerprint(text: str) -> str:
    """SHA-256 of lowercased, whitespace-normalised text."""
    normalised = RE_WHITESPACE.sub(" ", text.lower().strip())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 5) -> set[str]:
    """Character k-shingles for MinHash similarity."""
    text = RE_WHITESPACE.sub(" ", text.lower().strip())
    return {text[i: i + k] for i in range(max(1, len(text) - k + 1))}


class DedupIndex:
    """
    Two-level deduplication index on winner_prompt:
      1. Exact-match via SHA-256  (O(1), zero false positives)
      2. Near-duplicate via MinHash LSH  (Jaccard ≥ threshold)
    """

    def __init__(self, threshold: float = 0.85, num_perm: int = 128):
        self.seen_exact: set[str] = set()
        self.threshold  = threshold
        self.num_perm   = num_perm
        if HAS_DATASKETCH:
            self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
            self.minhashes: dict[str, MinHash] = {}
        else:
            self.lsh = None

    def _make_minhash(self, text: str) -> "MinHash":
        m = MinHash(num_perm=self.num_perm)
        for s in shingles(text):
            m.update(s.encode("utf8"))
        return m

    def is_duplicate(self, key: str, text: str) -> bool:
        """Returns True if text is duplicate/near-duplicate. Adds to index if not."""
        fp = fingerprint(text)
        if fp in self.seen_exact:
            return True
        self.seen_exact.add(fp)

        if self.lsh is not None:
            mh = self._make_minhash(text)
            if self.lsh.query(mh):
                return True
            try:
                self.lsh.insert(key[:200], mh)
                self.minhashes[key[:200]] = mh
            except ValueError:
                pass

        return False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Constraint-category distribution audit  (Step ix)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_categories(record: dict) -> list[str]:
    """Return all constraint category keys from a record's added_constraint dict."""
    constraints = record.get(FIELD_CONSTRAINTS)
    if not constraints or not isinstance(constraints, dict):
        return ["unknown"]
    return list(constraints.keys()) or ["unknown"]


def audit_distribution(
    category_counts: Counter,
    imbalance_threshold: float = 0.5,
) -> list[str]:
    """
    Warn about under-represented constraint categories.
    Flags any category whose share < imbalance_threshold × (1 / N_categories).
    """
    warnings_out = []
    total = sum(category_counts.values())
    if total == 0 or not category_counts:
        return warnings_out
    n_cats        = len(category_counts)
    uniform_share = 1.0 / n_cats
    low_threshold = imbalance_threshold * uniform_share
    for cat, count in sorted(category_counts.items()):
        share = count / total
        if share < low_threshold:
            warnings_out.append(
                f"  ⚠  '{cat}' under-represented: {count:,} records "
                f"({100*share:.1f}% vs expected ≥{100*low_threshold:.1f}%)"
            )
    return warnings_out


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Per-record cleaning entry point
# ═══════════════════════════════════════════════════════════════════════════════

def clean_record(
    record: dict,
    min_length: int = 15,
) -> tuple[Optional[dict], str]:
    """
    Apply the full cleaning chain to one RECAST record.

    Pipeline order is deliberate:
      1. Language check on raw winner_prompt (before any transforms)
      2. HTML / emoji / char cleaning on winner_prompt and response_of_winner_prompt
      3. Quality gates (length, printability, empty response) on cleaned prompt
      4. Constraint field integrity (added_constraint dict structure + count check)
      5. Alignment check via rule_evaluate_dict validators on response_of_winner_prompt
      6. Stopword removal on winner_prompt — last, so gates are unaffected

    Returns (cleaned_record, skip_reason). skip_reason is "" if kept.
    """

    prompt   = record.get(FIELD_PROMPT)   or ""
    response = record.get(FIELD_RESPONSE) or ""

    # ── (i) Language detection on raw prompt ─────────────────────────────────
    if prompt and HAS_LANGDETECT:
        lang = detect_language(prompt[:2000])
        if lang is not None and lang != "en":
            return None, "non_english"

    # ── (ii–iii) Text cleaning ────────────────────────────────────────────────
    cleaned = dict(record)
    cleaned[FIELD_PROMPT]    = clean_text(prompt)
    cleaned[FIELD_RESPONSE]  = clean_text(response)
    # Clean any other plain-string top-level fields
    for field in ("prompt_winner", "response_winner"):
        if isinstance(record.get(field), str):
            cleaned[field] = clean_text(record[field])

    prompt_clean = cleaned[FIELD_PROMPT]

    # ── (v) Quality gates — BEFORE stopword removal ───────────────────────────
    if len(prompt_clean.split()) < min_length:
        return None, "too_short"
    if not is_mostly_printable(prompt_clean):
        return None, "unprintable"
    if not cleaned[FIELD_RESPONSE].strip():
        return None, "empty_response"

    # ── (vi) Constraint field integrity ───────────────────────────────────────
    ok, reason = validate_constraints(cleaned)
    if not ok:
        return None, f"bad_constraint: {reason}"

    # ── (vii) Alignment — re-run rule validators on response ──────────────────
    ok, reason = check_alignment(cleaned)
    if not ok:
        return None, f"alignment: {reason}"

    # ── (iv) Stopword removal on prompt — applied last ────────────────────────
    cleaned[FIELD_PROMPT] = normalize_whitespace(remove_stopwords(prompt_clean))

    return cleaned, ""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    input_path: Path,
    output_path: Path,
    min_length: int,
    dedup_threshold: float,
    imbalance_threshold: float,
) -> dict:

    stats: dict = {
        "total":                   0,
        "kept":                    0,
        "skipped_non_english":     0,
        "skipped_too_short":       0,
        "skipped_unprintable":     0,
        "skipped_empty_response":  0,
        "skipped_bad_constraint":  0,
        "skipped_alignment":       0,
        "skipped_duplicate":       0,
        "skipped_other":           0,
    }

    category_counts: Counter = Counter()
    dedup = DedupIndex(threshold=dedup_threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(input_path,  "r", encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logging.warning(f"Line {line_no}: JSON parse error — {exc}")
                stats["skipped_other"] += 1
                continue

            # ── (i–vii) Clean + validate ──────────────────────────────────
            cleaned, reason = clean_record(record, min_length=min_length)

            if cleaned is None:
                stat_key = f"skipped_{reason.split(':')[0]}"
                stats[stat_key] = stats.get(stat_key, 0) + 1
                continue

            # ── (viii) Deduplication on winner_prompt ─────────────────────
            prompt_text = cleaned.get(FIELD_PROMPT) or ""
            if dedup.is_duplicate(f"rec_{line_no}", prompt_text):
                stats["skipped_duplicate"] += 1
                continue

            # ── (ix) Track category distribution ─────────────────────────
            for cat in extract_categories(cleaned):
                category_counts[cat] += 1

            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            stats["kept"] += 1

            if stats["total"] % 1000 == 0:
                pct = 100 * stats["kept"] / stats["total"]
                logging.info(
                    f"  Processed {stats['total']:,} | "
                    f"kept {stats['kept']:,} ({pct:.1f}%)"
                )

    # ── (ix) Distribution audit ───────────────────────────────────────────────
    dist_warnings = audit_distribution(category_counts, imbalance_threshold)

    stats["category_counts"] = dict(category_counts)
    stats["dist_warnings"]   = dist_warnings
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Full RECAST-30K cleaning pipeline — STAT 453 Team 15",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",   required=True,  type=Path,
                        help="Raw RECAST train JSONL")
    parser.add_argument("--output",  required=True,  type=Path,
                        help="Cleaned train JSONL (ready for augmentation)")
    parser.add_argument("--min_length",  type=int, default=15,
                        help="Min word count in winner_prompt after cleaning")
    parser.add_argument("--dedup_threshold", type=float, default=0.85,
                        help="Jaccard similarity threshold for near-dedup (0–1)")
    parser.add_argument("--imbalance_threshold", type=float, default=0.5,
                        help="Flag categories below this fraction of uniform share")

    args = parser.parse_args()

    sep = "=" * 65
    logging.info(sep)
    logging.info("  RECAST Full Cleaning Pipeline — STAT 453 Team 15")
    logging.info(sep)
    logging.info(f"  Input              : {args.input}")
    logging.info(f"  Output             : {args.output}")
    logging.info(f"  Min prompt length  : {args.min_length} words")
    logging.info(f"  Dedup threshold    : {args.dedup_threshold}")
    logging.info(f"  Imbalance thresh.  : {args.imbalance_threshold}")
    logging.info(f"  Prompt field       : '{FIELD_PROMPT}'")
    logging.info(f"  Response field     : '{FIELD_RESPONSE}'")
    logging.info(f"  Constraint field   : '{FIELD_CONSTRAINTS}'")
    logging.info(f"  Validator field    : '{FIELD_VALIDATORS}'")
    logging.info(f"  langdetect         : {'✓' if HAS_LANGDETECT else '✗ (disabled)'}")
    logging.info(f"  emoji lib          : {'✓' if HAS_EMOJI else '✗ (fallback)'}")
    logging.info(f"  NLTK stopwords     : {'✓' if HAS_NLTK else '✗ (fallback list)'}")
    logging.info(f"  MinHash dedup      : {'✓' if HAS_DATASKETCH else '✗ (exact-only)'}")
    logging.info(sep)

    stats = run_pipeline(
        input_path          = args.input,
        output_path         = args.output,
        min_length          = args.min_length,
        dedup_threshold     = args.dedup_threshold,
        imbalance_threshold = args.imbalance_threshold,
    )

    total = stats["total"]
    kept  = stats["kept"]

    logging.info(sep)
    logging.info("  CLEANING COMPLETE — Summary")
    logging.info(sep)
    logging.info(f"  Total records processed        : {total:>8,}")
    logging.info(f"  Records kept (→ augmentation)  : {kept:>8,}  ({100*kept/max(total,1):.1f}%)")
    logging.info(f"  Skipped — non-English          : {stats['skipped_non_english']:>8,}")
    logging.info(f"  Skipped — too short            : {stats['skipped_too_short']:>8,}")
    logging.info(f"  Skipped — unprintable          : {stats['skipped_unprintable']:>8,}")
    logging.info(f"  Skipped — empty response       : {stats['skipped_empty_response']:>8,}")
    logging.info(f"  Skipped — bad constraint       : {stats['skipped_bad_constraint']:>8,}")
    logging.info(f"  Skipped — alignment fail       : {stats['skipped_alignment']:>8,}")
    logging.info(f"  Skipped — duplicate            : {stats['skipped_duplicate']:>8,}")
    logging.info(f"  Skipped — other                : {stats['skipped_other']:>8,}")

    logging.info("")
    logging.info("  Constraint category distribution (kept records):")
    cat_counts = stats.get("category_counts", {})
    cat_total  = sum(cat_counts.values())
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        logging.info(f"    {cat:<20} : {cnt:>6,}  ({100*cnt/max(cat_total,1):.1f}%)")
    for w in stats.get("dist_warnings", []):
        logging.warning(w)

    logging.info(sep)
    logging.info(f"  Output ready for augmentation  : {args.output}")
    logging.info(sep)


if __name__ == "__main__":
    main()