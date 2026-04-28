"""Back-translation augmentation for RECAST-30K dataset.

Paraphrases prompts via English -> intermediate language -> English
using Helsinki-NLP MarianMT models. Constraint-critical tokens are
verified after translation; records that fail verification fall back
to the original prompt.
"""

import argparse
import json
import logging
import os
import re
import random

try:
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FORMAT_KEYWORDS = frozenset(
    [
        "json",
        "bullet",
        "numbered",
        "paragraph",
        "sentence",
        "word",
        "uppercase",
        "lowercase",
    ]
)


def detect_prompt_field(record: dict) -> str:
    """Return the key used for the prompt text in *record*.

    Raises ``KeyError`` if neither ``winner_prompt`` nor ``instruction``
    is present.
    """
    if "winner_prompt" in record:
        return "winner_prompt"
    if "instruction" in record:
        return "instruction"
    raise KeyError(
        "Record has neither 'winner_prompt' nor 'instruction' field"
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_translation_models(intermediate_lang: str = "de") -> tuple:
    """Load forward (en->lang) and backward (lang->en) MarianMT models.

    Returns (fwd_model, fwd_tokenizer, bwd_model, bwd_tokenizer).
    """
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers and sentencepiece are required for back-translation. "
            "Install with: pip install transformers sentencepiece torch"
        )

    fwd_name = f"Helsinki-NLP/opus-mt-en-{intermediate_lang}"
    bwd_name = f"Helsinki-NLP/opus-mt-{intermediate_lang}-en"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Translation models will run on device: %s", device)

    logger.info("Loading forward model: %s", fwd_name)
    fwd_tokenizer = MarianTokenizer.from_pretrained(fwd_name)
    fwd_model = MarianMTModel.from_pretrained(fwd_name).to(device).eval()

    logger.info("Loading backward model: %s", bwd_name)
    bwd_tokenizer = MarianTokenizer.from_pretrained(bwd_name)
    bwd_model = MarianMTModel.from_pretrained(bwd_name).to(device).eval()

    return fwd_model, fwd_tokenizer, bwd_model, bwd_tokenizer


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def translate_batch(
    texts: list[str],
    model,
    tokenizer,
    max_length: int = 512,
) -> list[str]:
    """Translate a batch of texts using a MarianMT model."""
    if not texts:
        return []

    # Preserve indices of empty strings so we can skip them.
    non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
    non_empty_texts = [texts[i] for i in non_empty_indices]

    if not non_empty_texts:
        return [""] * len(texts)

    encoded = tokenizer(
        non_empty_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)
    with torch.no_grad():
        translated_ids = model.generate(**encoded, max_length=max_length)
    decoded = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)

    results = [""] * len(texts)
    for idx, translated in zip(non_empty_indices, decoded):
        results[idx] = translated
    return results


def back_translate_prompt(
    prompt: str,
    fwd_model,
    fwd_tok,
    bwd_model,
    bwd_tok,
    max_length: int = 512,
) -> str:
    """Back-translate a single prompt: en -> intermediate -> en."""
    if not prompt.strip():
        return ""
    intermediate = translate_batch([prompt], fwd_model, fwd_tok, max_length)
    back = translate_batch(intermediate, bwd_model, bwd_tok, max_length)
    return back[0]


# ---------------------------------------------------------------------------
# Constraint preservation
# ---------------------------------------------------------------------------


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


def extract_constraint_tokens(record: dict) -> set[str]:
    """Extract constraint-critical tokens from a record.

    Gathers:
    - Numbers from the prompt (``\\b\\d+\\b``)
    - All words from ``added_constraint`` description strings
    - Quoted substrings in the prompt
    - Format keywords (json, bullet, etc.)
    - Constraint category names
    - Proper nouns / named entities
    """
    tokens: set[str] = set()

    prompt_field = detect_prompt_field(record)
    prompt = record[prompt_field]

    # Numbers from prompt
    tokens.update(re.findall(r"\b\d+\b", prompt))

    # Quoted substrings in prompt
    for match in re.findall(r"""['"]([^'"]+)['"]""", prompt):
        tokens.add(match.lower())

    # Format keywords present in prompt
    prompt_lower = prompt.lower()
    for kw in FORMAT_KEYWORDS:
        if kw in prompt_lower:
            tokens.add(kw)

    # Tokens from added_constraint
    added = record.get("added_constraint")
    if added and isinstance(added, dict):
        for category, descriptions in added.items():
            # Category name
            tokens.add(category.lower())
            if isinstance(descriptions, list):
                for desc in descriptions:
                    for word in str(desc).split():
                        tokens.add(word.lower())

    # Proper nouns / named entities from prompt
    tokens.update(_extract_entities(prompt))

    return tokens


def verify_constraint_preservation(
    original: str,
    augmented: str,
    constraint_tokens: set[str],
) -> bool:
    """Return True if all *constraint_tokens* appear in *augmented* (case-insensitive)."""
    augmented_lower = augmented.lower()
    for token in constraint_tokens:
        if token.lower() not in augmented_lower:
            return False
    return True


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_back_translation_pipeline(
    input_path: str,
    output_path: str,
    intermediate_lang: str = "de",
    batch_size: int = 32,
    max_length: int = 512,
    seed: int = 42,
    force: bool = False,
) -> dict:
    """Run full back-translation augmentation pipeline.

    Returns a stats dict with keys: total, augmented, fell_back.
    """
    # Idempotency check
    if os.path.exists(output_path) and not force:
        logger.info(
            "Output file %s already exists. Use --force to overwrite. Skipping.",
            output_path,
        )
        return {"total": 0, "augmented": 0, "fell_back": 0, "skipped": True}

    random.seed(seed)

    # Load models
    fwd_model, fwd_tok, bwd_model, bwd_tok = load_translation_models(
        intermediate_lang
    )

    # Read input records
    records: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info("Loaded %d records from %s", len(records), input_path)

    # Extract prompts
    prompt_fields = [detect_prompt_field(r) for r in records]
    prompts = [r[pf] for r, pf in zip(records, prompt_fields)]

    n_batches = (len(prompts) + batch_size - 1) // batch_size
    log_every = max(1, n_batches // 20)  # ~20 progress updates per pass

    # Batch translate: forward pass
    intermediate_texts: list[str] = []
    for bi, i in enumerate(range(0, len(prompts), batch_size)):
        batch = prompts[i : i + batch_size]
        intermediate_texts.extend(
            translate_batch(batch, fwd_model, fwd_tok, max_length)
        )
        if bi % log_every == 0 or bi == n_batches - 1:
            logger.info("  forward %d/%d batches", bi + 1, n_batches)

    # Batch translate: backward pass
    back_translated: list[str] = []
    for bi, i in enumerate(range(0, len(intermediate_texts), batch_size)):
        batch = intermediate_texts[i : i + batch_size]
        back_translated.extend(
            translate_batch(batch, bwd_model, bwd_tok, max_length)
        )
        if bi % log_every == 0 or bi == n_batches - 1:
            logger.info("  backward %d/%d batches", bi + 1, n_batches)

    # Build augmented records with verification
    augmented_count = 0
    fell_back_count = 0
    augmented_records: list[dict] = []

    for idx, record in enumerate(records):
        pf = prompt_fields[idx]
        bt_text = back_translated[idx]
        constraint_tokens = extract_constraint_tokens(record)

        preserved = verify_constraint_preservation(
            prompts[idx], bt_text, constraint_tokens
        )

        augmented = dict(record)
        if preserved and bt_text.strip():
            augmented[pf] = bt_text
            augmented_count += 1
        else:
            logger.warning(
                "Constraint verification failed for record %s; falling back to original",
                record.get("id", idx),
            )
            fell_back_count += 1

        augmented["id"] = str(record.get("id", idx)) + "_bt"
        augmented["augmentation_method"] = "back_translation"
        augmented["augmentation_intermediate_lang"] = intermediate_lang
        augmented_records.append(augmented)

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in augmented_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "total": len(records),
        "augmented": augmented_count,
        "fell_back": fell_back_count,
        "skipped": False,
    }
    logger.info("Pipeline complete: %s", stats)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Back-translation augmentation for RECAST-30K"
    )
    parser.add_argument("--input", required=True, help="Path to cleaned JSONL")
    parser.add_argument(
        "--output", required=True, help="Path for augmented JSONL output"
    )
    parser.add_argument(
        "--intermediate_lang",
        default="de",
        help="Intermediate language for back-translation (default: de)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Translation batch size (default: 32)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max token length for translation (default: 512)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    stats = run_back_translation_pipeline(
        input_path=args.input,
        output_path=args.output,
        intermediate_lang=args.intermediate_lang,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
        force=args.force,
    )
    logger.info("Final stats: %s", stats)


if __name__ == "__main__":
    main()
