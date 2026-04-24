# Augmentation

Prompt-level data augmentation for RECAST-30K. Augments only `winner_prompt` (or `instruction`); responses and constraint metadata are preserved unchanged.

## Scripts

### `lexical_edit.py` -- EDA-style lexical perturbations

Synonym replacement, random insertion, random swap, and random deletion using NLTK WordNet. Constraint-critical tokens (numbers, keywords, format specifiers) are protected from modification.

```bash
python3 -m src.crllm.dataset.augmentation.lexical_edit \
    --input  datasets/recast_30k_clean.jsonl \
    --output datasets/recast_30k_lex.jsonl \
    --alpha_sr 0.1 --alpha_ri 0.1 --alpha_rs 0.1 --alpha_rd 0.05 \
    --seed 42
```

### `back_translate.py` -- Paraphrasing via back-translation

English -> German -> English using Helsinki-NLP MarianMT models (~300MB per direction, CPU-friendly). Falls back to original prompt if constraint tokens are lost in translation.

```bash
python3 -m src.crllm.dataset.augmentation.back_translate \
    --input  datasets/recast_30k_clean.jsonl \
    --output datasets/recast_30k_bt.jsonl \
    --intermediate_lang de --batch_size 32 \
    --seed 42
```

## Common Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Cleaned JSONL from preprocessing |
| `--output` | *(required)* | Augmented JSONL output path |
| `--force` | `false` | Overwrite output if it already exists |
| `--seed` | `42` | Random seed for reproducibility |

## Output Schema

Each output record is a copy of the input with:
- Prompt field augmented (all other fields unchanged)
- `id` suffixed with `_lex` or `_bt`
- `augmentation_method` field added (`"lexical_edit"` or `"back_translation"`)

## Testing

```bash
python3 -m pytest tests/test_lexical_edit.py -v
python3 -m pytest tests/test_back_translate.py -v
```
