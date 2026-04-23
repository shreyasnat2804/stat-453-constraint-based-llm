# stat-453 Constraint-Based LLM (Team 15)

Course project exploring instruction-following with explicit constraints on
the RECAST-30K dataset: preprocess the raw corpus, augment it, fine-tune a
base model (LoRA + full), and evaluate constraint satisfaction.

---

## Repository layout

```
stat-453-constraint-based-llm/
├── src/crllm/                    ← all library code lives here (package = crllm)
│   ├── dataset/
│   │   ├── preprocess/           ← step 1: clean raw RECAST-30K
│   │   ├── augmentation/         ← step 2: data augmentation (e.g. back-translation)
│   │   └── clustering/           ← exploratory: cluster by constraint mix
│   ├── training/
│   │   ├── lora_finetune/        ← LoRA fine-tuning pipeline
│   │   └── full_finetune/        ← full fine-tuning pipeline
│   └── evaluation/               ← constraint-checker, LLM-judge, viz
│
├── datasets/                     ← raw + cleaned data (zipped for git)
├── tests/                        ← pytest tests , one test file per module
├── notebooks/                    ← exploratory notebooks (not for final code)
├── utilities/                    ← post-run reports + reusable audit tools
├── docs/                         ← proposal + literature notes
└── pyproject.toml                ← Poetry project config (Add your dev dependencies here)
```

---

## Where does new code go?

Please follow the layout below, the `crllm` package is what gets imported
in tests and notebooks, so library code must stay inside `src/crllm/`.

| If you're adding… | Put it in… |
|---|---|
| A dataset cleaning / augmentation / clustering module | `src/crllm/dataset/<stage>/` |
| A training pipeline or fine-tuning script | `src/crllm/training/<approach>/` |
| An evaluator, judge, or metric | `src/crllm/evaluation/` |
| A unit test for any of the above | `tests/test_<module>.py` |
| A one-off exploratory analysis / audit | `scripts/` |
| A report script whose output belongs with the dataset | `utilities/` |
| Literature review / proposal docs | `docs/` |
| A Jupyter notebook | `notebooks/` (not for shipped code) |

**Don't:**
- Put `.py` modules at the repo root , they won't be importable as `crllm.*`.
- Drop raw data into `src/` , data goes in `datasets/`.
- Mix notebooks with library code , keep notebooks in `notebooks/`.
- Commit unzipped JSONL datasets > 50 MB , zip them (see `datasets/*.zip`).

---

## Running things

```bash
# Install (Poetry manages all deps)
poetry install

# Preprocess raw RECAST
bash src/crllm/dataset/preprocess/run_preprocess.sh

# Run the full test suite
PYTHONPATH=src python -m pytest tests/ -v

# Post-cleaning audits
python utilities/count_non_ascii_records.py -input datasets/recast_30k_clean.jsonl
python utilities/dataset_summary.py        -input datasets/recast_30k_clean.jsonl
```

Each subpackage (`preprocess/`, `evaluation/`, `clustering/`, ...) has its
own `README.md` covering usage and arguments , read those before modifying
that stage's code.

---

## Team

Gayathri Ethiraj · Fnu Rinkle Rose Renny · Mark Aashish Lnu · Shreyas Natarajan
