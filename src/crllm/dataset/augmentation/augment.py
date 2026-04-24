"""
Augmentation orchestrator for RECAST-30K.
STAT 453 - Team 15 | Spring 2026

Accepts a JSON or JSONL dataset file and runs both augmentation
techniques (lexical editing + back-translation), producing a single
combined output with 3x the original records.

Usage:
    python3 -m src.crllm.dataset.augmentation.augment \
        --input  datasets/RECAST-30K.json \
        --output datasets/recast_30k_augmented.jsonl \
        [--seed 42] [--force] [--batch_size 32]

Callable from a pipeline:
    from src.crllm.dataset.augmentation.augment import run_augmentation
    stats = run_augmentation("datasets/RECAST-30K.json", "datasets/augmented.jsonl")
"""

import argparse
import json
import logging
import tempfile
import zipfile
from pathlib import Path

from src.crllm.dataset.augmentation.back_translate import (
    run_back_translation_pipeline,
)
from src.crllm.dataset.augmentation.lexical_edit import (
    run_lexical_edit_pipeline,
)

logger = logging.getLogger(__name__)


# ── I/O helpers ──────────────────────────────────────────────────────────────


def _load_records(input_path: Path) -> list[dict]:
    """Load records from a JSON array, JSONL, or zipped JSONL file."""
    # Handle zip files — read the first file inside the archive
    if input_path.suffix == ".zip":
        records = []
        with zipfile.ZipFile(input_path) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                for line in f:
                    line = line.decode("utf-8").strip()
                    if line:
                        records.append(json.loads(line))
        return records

    text = input_path.read_text(encoding="utf-8")
    stripped = text.lstrip()

    # JSON array (starts with '[')
    if stripped.startswith("["):
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError(f"Expected JSON array, got {type(records).__name__}")
        return records

    # JSONL (one object per line)
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL file into list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Public API ───────────────────────────────────────────────────────────────


def run_augmentation(
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 42,
    force: bool = False,
    # Lexical edit params
    alpha_sr: float = 0.1,
    alpha_ri: float = 0.1,
    alpha_rs: float = 0.1,
    alpha_rd: float = 0.05,
    # Back-translation params
    intermediate_lang: str = "de",
    batch_size: int = 32,
    max_length: int = 512,
) -> dict:
    """Run both augmentation techniques and combine into a single output.

    Accepts JSON array or JSONL as input. Produces JSONL with:
      - Original records (unchanged)
      - Lexically-edited records (id suffixed with ``_lex``)
      - Back-translated records (id suffixed with ``_bt``)

    Returns a stats dict summarising the run.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Idempotency
    if output_path.exists() and not force:
        logger.info(
            "Output %s already exists. Use force=True to overwrite. Skipping.",
            output_path,
        )
        return {"status": "skipped", "reason": "output_exists"}

    # ── Load & normalise to JSONL ────────────────────────────────────────
    logger.info("Loading records from %s", input_path)
    records = _load_records(input_path)
    logger.info("Loaded %d records", len(records))

    # Write a temp JSONL for the sub-pipelines
    tmp_dir = tempfile.mkdtemp(prefix="recast_aug_")
    tmp_input = Path(tmp_dir) / "input.jsonl"
    tmp_lex = Path(tmp_dir) / "augmented_lex.jsonl"
    tmp_bt = Path(tmp_dir) / "augmented_bt.jsonl"

    _write_jsonl(tmp_input, records)

    # ── Run lexical editing ──────────────────────────────────────────────
    logger.info("Running lexical editing augmentation...")
    lex_stats = run_lexical_edit_pipeline(
        input_path=tmp_input,
        output_path=tmp_lex,
        alpha_sr=alpha_sr,
        alpha_ri=alpha_ri,
        alpha_rs=alpha_rs,
        alpha_rd=alpha_rd,
        seed=seed,
        force=True,
    )
    logger.info("Lexical edit stats: %s", lex_stats)

    # ── Run back-translation ─────────────────────────────────────────────
    logger.info("Running back-translation augmentation...")
    bt_stats = run_back_translation_pipeline(
        input_path=str(tmp_input),
        output_path=str(tmp_bt),
        intermediate_lang=intermediate_lang,
        batch_size=batch_size,
        max_length=max_length,
        seed=seed,
        force=True,
    )
    logger.info("Back-translation stats: %s", bt_stats)

    # ── Combine: original + lex + bt ─────────────────────────────────────
    lex_records = _read_jsonl(tmp_lex)
    bt_records = _read_jsonl(tmp_bt)

    combined = records + lex_records + bt_records
    _write_jsonl(output_path, combined)

    stats = {
        "status": "complete",
        "original": len(records),
        "lexical_edit": len(lex_records),
        "back_translation": len(bt_records),
        "total_output": len(combined),
        "lexical_edit_stats": lex_stats,
        "back_translation_stats": bt_stats,
    }
    logger.info(
        "Augmentation complete: %d original + %d lex + %d bt = %d total",
        len(records), len(lex_records), len(bt_records), len(combined),
    )
    return stats


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="RECAST-30K Augmentation Orchestrator - STAT 453 Team 15",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to dataset (JSON array or JSONL)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path for combined augmented JSONL output")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite output if it exists")
    parser.add_argument("--alpha_sr", type=float, default=0.1,
                        help="Lexical edit: synonym replacement fraction")
    parser.add_argument("--alpha_ri", type=float, default=0.1,
                        help="Lexical edit: random insertion fraction")
    parser.add_argument("--alpha_rs", type=float, default=0.1,
                        help="Lexical edit: random swap fraction")
    parser.add_argument("--alpha_rd", type=float, default=0.05,
                        help="Lexical edit: random deletion fraction")
    parser.add_argument("--intermediate_lang", type=str, default="de",
                        help="Back-translation intermediate language")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Back-translation batch size")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Back-translation max token length")

    args = parser.parse_args()

    sep = "=" * 65
    logger.info(sep)
    logger.info("  RECAST-30K Augmentation Orchestrator - STAT 453 Team 15")
    logger.info(sep)

    stats = run_augmentation(
        input_path=args.input,
        output_path=args.output,
        seed=args.seed,
        force=args.force,
        alpha_sr=args.alpha_sr,
        alpha_ri=args.alpha_ri,
        alpha_rs=args.alpha_rs,
        alpha_rd=args.alpha_rd,
        intermediate_lang=args.intermediate_lang,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    logger.info(sep)
    logger.info("  Final stats: %s", json.dumps(stats, indent=2, default=str))
    logger.info(sep)


if __name__ == "__main__":
    main()
