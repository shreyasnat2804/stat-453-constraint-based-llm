"""
Lexical Editing Augmentation for RECAST-30K
STAT 453 - Team 15 | Spring 2026

EDA-style (Easy Data Augmentation) lexical perturbations applied to prompts
only. Uses NLTK WordNet for synonym lookup. Constraint tokens and format
keywords are protected from modification.

Operations:
  1. Synonym Replacement (SR) — replace non-protected words with WordNet synonyms
  2. Random Insertion (RI)    — insert a synonym of a random word at a random position
  3. Random Swap (RS)         — swap two non-protected words
  4. Random Deletion (RD)     — delete non-protected words with probability p

Usage:
    python3 -m src.crllm.dataset.augmentation.lexical_edit \\
        --input  path/to/cleaned.jsonl \\
        --output path/to/augmented.jsonl \\
        [--alpha_sr 0.1] [--alpha_ri 0.1] [--alpha_rs 0.1] [--alpha_rd 0.05] \\
        [--seed 42] [--force]
"""

import argparse
import json
import logging
import random
import re
from pathlib import Path

# ── NLTK WordNet setup ───────────────────────────────────────────────────────

try:
    from nltk.corpus import wordnet
    wordnet.synsets("test")
except LookupError:
    import nltk
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    from nltk.corpus import wordnet

# ── Constants ────────────────────────────────────────────────────────────────

FORMAT_KEYWORDS = frozenset({
    "json", "bullet", "numbered", "paragraph", "sentence", "word", "words",
    "uppercase", "lowercase", "bold", "italic", "heading", "section", "list",
})

RE_NUMBERS = re.compile(r"\b\d+\b")
RE_QUOTED = re.compile(r"""(['"])(.*?)\1""")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Helper functions
# ═════════════════════════════════════════════════════════════════════════════

def detect_prompt_field(record: dict) -> str:
    """Return the prompt field name present in the record."""
    if "winner_prompt" in record:
        return "winner_prompt"
    if "instruction" in record:
        return "instruction"
    raise KeyError("Record has neither 'winner_prompt' nor 'instruction'")


def get_wordnet_synonyms(word: str) -> list[str]:
    """Return WordNet synonyms for *word*, excluding the word itself and
    multi-word expressions (those containing underscores or spaces)."""
    synonyms: list[str] = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name()
            if "_" in name or " " in name:
                continue
            if name.lower() != word.lower() and name not in synonyms:
                synonyms.append(name)
    return synonyms


def _extract_entities(text: str) -> set[str]:
    """Extract likely proper nouns (capitalised words not at sentence start)."""
    entities: set[str] = set()
    sentences = re.split(r"[.!?]\s+", text)
    for sent in sentences:
        words = sent.split()
        for word in words[1:]:
            cleaned = word.strip(".,;:!?'\"()[]{}*")
            if (
                cleaned
                and cleaned[0].isupper()
                and not cleaned.isupper()
                and not cleaned.isdigit()
                and len(cleaned) >= 2
            ):
                entities.add(cleaned.lower())
    return entities


def extract_protected_tokens(record: dict) -> set[str]:
    """Build the set of tokens that must NOT be modified during augmentation."""
    protected: set[str] = set()

    # Determine prompt field
    try:
        prompt_field = detect_prompt_field(record)
    except KeyError:
        prompt_field = None

    prompt_text = record.get(prompt_field, "") if prompt_field else ""

    # Numbers from prompt
    for match in RE_NUMBERS.finditer(prompt_text):
        protected.add(match.group())

    # Quoted substrings — add individual tokens
    for match in RE_QUOTED.finditer(prompt_text):
        for tok in match.group(2).split():
            protected.add(tok.lower())

    # Constraint description words
    constraints = record.get("added_constraint", {})
    if isinstance(constraints, dict):
        for category, descriptions in constraints.items():
            # Category name itself
            protected.add(category.lower())
            if isinstance(descriptions, list):
                for desc in descriptions:
                    if isinstance(desc, str):
                        for tok in desc.split():
                            protected.add(tok.strip(".,;:!?'\"()[]{}").lower())

    # Hardcoded format keywords
    protected.update(FORMAT_KEYWORDS)

    # Proper nouns / named entities from prompt
    protected.update(_extract_entities(prompt_text))

    return protected


def is_protected(token: str, protected: set[str]) -> bool:
    """Case-insensitive membership check against the protected set."""
    return token.lower() in protected


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EDA operations
# ═════════════════════════════════════════════════════════════════════════════

def synonym_replacement(
    words: list[str],
    protected: set[str],
    n: int,
    rng: random.Random,
) -> list[str]:
    """Replace up to *n* random non-protected words with WordNet synonyms."""
    words = list(words)
    eligible = [i for i, w in enumerate(words) if not is_protected(w, protected)]
    rng.shuffle(eligible)
    replaced = 0
    for idx in eligible:
        if replaced >= n:
            break
        syns = get_wordnet_synonyms(words[idx])
        if syns:
            words[idx] = rng.choice(syns)
            replaced += 1
    return words


def random_insertion(
    words: list[str],
    protected: set[str],
    n: int,
    rng: random.Random,
) -> list[str]:
    """Insert *n* synonyms of random non-protected words at random positions."""
    words = list(words)
    for _ in range(n):
        eligible = [w for w in words if not is_protected(w, protected)]
        if not eligible:
            break
        chosen = rng.choice(eligible)
        syns = get_wordnet_synonyms(chosen)
        if syns:
            synonym = rng.choice(syns)
            insert_pos = rng.randint(0, len(words))
            words.insert(insert_pos, synonym)
    return words


def random_swap(
    words: list[str],
    protected: set[str],
    n: int,
    rng: random.Random,
) -> list[str]:
    """Swap *n* pairs of non-protected words."""
    words = list(words)
    eligible = [i for i, w in enumerate(words) if not is_protected(w, protected)]
    if len(eligible) < 2:
        return words
    for _ in range(n):
        i, j = rng.sample(eligible, 2)
        words[i], words[j] = words[j], words[i]
    return words


def random_deletion(
    words: list[str],
    protected: set[str],
    p: float,
    rng: random.Random,
) -> list[str]:
    """Delete non-protected words with probability *p*. Always keep >= 1 word."""
    if not words:
        return []
    if len(words) == 1:
        return list(words)
    result = []
    for w in words:
        if is_protected(w, protected):
            result.append(w)
        elif rng.random() >= p:
            result.append(w)
    # Guarantee at least one word survives
    if not result:
        result.append(words[rng.randint(0, len(words) - 1)])
    return result


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Combined EDA augmentation
# ═════════════════════════════════════════════════════════════════════════════

def eda_augment(
    text: str,
    protected_tokens: set[str],
    alpha_sr: float = 0.1,
    alpha_ri: float = 0.1,
    alpha_rs: float = 0.1,
    alpha_rd: float = 0.05,
    rng: random.Random | None = None,
) -> str:
    """Apply all four EDA operations sequentially and return augmented text."""
    if rng is None:
        rng = random.Random(42)

    words = text.split()
    if not words:
        return text

    num_words = len(words)
    n_sr = max(1, int(alpha_sr * num_words)) if alpha_sr > 0 else 0
    n_ri = max(1, int(alpha_ri * num_words)) if alpha_ri > 0 else 0
    n_rs = max(1, int(alpha_rs * num_words)) if alpha_rs > 0 else 0

    # Apply operations sequentially
    if n_sr > 0:
        words = synonym_replacement(words, protected_tokens, n_sr, rng)
    if n_ri > 0:
        words = random_insertion(words, protected_tokens, n_ri, rng)
    if n_rs > 0:
        words = random_swap(words, protected_tokens, n_rs, rng)
    if alpha_rd > 0:
        words = random_deletion(words, protected_tokens, alpha_rd, rng)

    return " ".join(words)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run_lexical_edit_pipeline(
    input_path: Path,
    output_path: Path,
    alpha_sr: float = 0.1,
    alpha_ri: float = 0.1,
    alpha_rs: float = 0.1,
    alpha_rd: float = 0.05,
    seed: int = 42,
    force: bool = False,
) -> dict:
    """Read cleaned JSONL, augment each prompt, write augmented JSONL.

    Returns a stats dict with record counts.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Idempotency guard
    if output_path.exists() and not force:
        logging.info(
            "Output %s already exists. Use --force to overwrite. Skipping.",
            output_path,
        )
        return {"status": "skipped", "reason": "output_exists"}

    rng = random.Random(seed)
    stats = {"total": 0, "augmented": 0, "errors": 0}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(input_path, "r", encoding="utf-8") as fin,
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
                logging.warning("Line %d: JSON parse error - %s", line_no, exc)
                stats["errors"] += 1
                continue

            try:
                prompt_field = detect_prompt_field(record)
            except KeyError:
                logging.warning("Line %d: no prompt field found, skipping", line_no)
                stats["errors"] += 1
                continue

            prompt_text = record.get(prompt_field, "")
            protected = extract_protected_tokens(record)

            augmented_text = eda_augment(
                prompt_text, protected,
                alpha_sr=alpha_sr, alpha_ri=alpha_ri,
                alpha_rs=alpha_rs, alpha_rd=alpha_rd,
                rng=rng,
            )

            augmented = dict(record)
            augmented[prompt_field] = augmented_text
            augmented["id"] = record.get("id", f"rec_{line_no}") + "_lex"
            augmented["augmentation_method"] = "lexical_edit"

            fout.write(json.dumps(augmented, ensure_ascii=False) + "\n")
            stats["augmented"] += 1

    logging.info(
        "Lexical edit complete: %d total, %d augmented, %d errors",
        stats["total"], stats["augmented"], stats["errors"],
    )
    return stats


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Lexical editing augmentation for RECAST-30K - STAT 453 Team 15",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to cleaned JSONL")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path for augmented JSONL output")
    parser.add_argument("--alpha_sr", type=float, default=0.1,
                        help="Fraction of words for synonym replacement")
    parser.add_argument("--alpha_ri", type=float, default=0.1,
                        help="Fraction of words for random insertion")
    parser.add_argument("--alpha_rs", type=float, default=0.1,
                        help="Fraction of words for random swap")
    parser.add_argument("--alpha_rd", type=float, default=0.05,
                        help="Fraction of words for random deletion")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite output if it exists")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    sep = "=" * 65
    logging.info(sep)
    logging.info("  Lexical Editing Augmentation - STAT 453 Team 15")
    logging.info(sep)
    logging.info("  Input       : %s", args.input)
    logging.info("  Output      : %s", args.output)
    logging.info("  alpha_sr    : %s", args.alpha_sr)
    logging.info("  alpha_ri    : %s", args.alpha_ri)
    logging.info("  alpha_rs    : %s", args.alpha_rs)
    logging.info("  alpha_rd    : %s", args.alpha_rd)
    logging.info("  seed        : %s", args.seed)
    logging.info("  force       : %s", args.force)
    logging.info(sep)

    stats = run_lexical_edit_pipeline(
        input_path=args.input,
        output_path=args.output,
        alpha_sr=args.alpha_sr,
        alpha_ri=args.alpha_ri,
        alpha_rs=args.alpha_rs,
        alpha_rd=args.alpha_rd,
        seed=args.seed,
        force=args.force,
    )

    logging.info(sep)
    logging.info("  Stats: %s", json.dumps(stats, indent=2))
    logging.info(sep)


if __name__ == "__main__":
    main()
