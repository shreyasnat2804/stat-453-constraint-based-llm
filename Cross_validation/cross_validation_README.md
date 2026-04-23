# Cross Validation — LoRA vs Full Fine-Tuning
**STAT 453, Spring 2026 | University of Wisconsin-Madison**

---

## What is this folder?

This folder contains the evaluation pipeline that compares two fine-tuned models side by side:

- **Rinkle's model** — Llama 3.2 1B with LoRA adapters (cheap, fast, ~0.5% of weights trained)
- **Mark's model** — Llama 3.2 1B fully fine-tuned (expensive, all weights updated)

Both models are evaluated on the **same test examples** from RECAST-30K and scored on how well they follow multi-constraint instructions. This comparison is the **core result of the project**.

---

## Files in this folder

| File | What it is |
|------|-----------|
| `cross_validation_kfold.ipynb` | Main evaluation notebook — run this |
| `README.md` | This file |

**Output files (created when you run the notebook):**

| File | What it contains |
|------|-----------------|
| `kfold_results/kfold_comparison.csv` | Final comparison table — use in paper |
| `kfold_results/kfold_comparison_plot.png` | Charts — use in paper |
| `kfold_results/all_results.json` | Raw scored data for every example |
| `kfold_results/results_fold_1..5.json` | Per-fold checkpoints (crash protection) |

---

## What is K-Fold Cross Validation?

Instead of evaluating on one fixed test split, K-Fold divides the full dataset into K equal chunks and tests on each chunk separately. The final score is the average across all folds.

**Why this is better than a single test split:**
A single test split might accidentally contain easy or hard examples, making the score look artificially good or bad. K-Fold tests on ALL examples equally, so the final score is statistically reliable.

**How it works with our dataset:**

```
RECAST-30K (30,000 examples) → split into 5 folds of 6,000 each

Round 1: TEST Fold 1 → CSR score
Round 2: TEST Fold 2 → CSR score
Round 3: TEST Fold 3 → CSR score
Round 4: TEST Fold 4 → CSR score
Round 5: TEST Fold 5 → CSR score

Final Score = average of all 5 scores
```

For speed, we sample **200 examples per fold** instead of all 6,000 → 1,000 total examples evaluated.

---

## What is CSR (Constraint Satisfaction Rate)?

CSR is the metric we use to score how well a model follows instructions. It is a number between **0 and 1**.

**Example:**
An instruction has 4 constraints:
- Write 300–400 words ✓
- Start with the word "The" ✓
- Use "love" 9 times ✗
- End with "generosity" ✓

Model satisfies 3 out of 4 → **CSR = 0.75**

A perfect model that follows every constraint would score **CSR = 1.0**.

---

## What constraint categories are evaluated?

| Category | What it checks |
|----------|---------------|
| Length | Word count and sentence count within required range |
| Keyword | Required words appear the required number of times |
| Start_With | Response begins with the required word |
| End_With | Response ends with the required word |
| Format | Uses required format markers like `<<title>>` |
| Tone | No informal language (proxy for formal tone) |
| Style | Style constraints (pass-through — too complex to automate) |
| Role_Playing | Role-playing constraints (pass-through) |
| Numerical_Constraints | Numerical constraints (pass-through) |

---

## What you need before running

You need three things from your teammates:

| What | Who provides it | Where to put it |
|------|----------------|----------------|
| LoRA adapter folder (`lora_adapter/`) | Rinkle | Upload to Colab or Google Drive |
| Full FT model folder (`finetuned/`) | Mark | Upload to Colab or Google Drive |
| RECAST dataset (`recast_30k_clean.jsonl`) | Shreyas | Upload to Colab or Google Drive |

Then update these 3 lines in **Step 3** of the notebook:

```python
LORA_ADAPTER_PATH  = "/content/outputs/lora_r8_0.0001/lora_adapter"  # ← Rinkle's path
FULL_FT_MODEL_PATH = "/content/output/finetuned"                      # ← Mark's path
DATASET_PATH       = "/content/recast_30k_clean.jsonl"                # ← Shreyas's path
```

---

## How to run the notebook

### Step 1 — Open in Colab

Go to [colab.research.google.com](https://colab.research.google.com) → File → Open → GitHub → paste repo URL → open `cross_validation_kfold.ipynb`

### Step 2 — Set GPU runtime

Runtime → Change runtime type → **T4 GPU** → Save

### Step 3 — Add HuggingFace token

Click the 🔑 **Secrets** icon in the left sidebar → Add secret named `HF_TOKEN` → paste your token from `https://huggingface.co/settings/tokens`

### Step 4 — Update paths

In Step 3 of the notebook, update the 3 paths to point to your files.

### Step 5 — Run all cells

Run every cell **top to bottom** in order. Do not skip any cell.

**Estimated time: ~1.5 hours on T4 GPU**

---

## What the output looks like

After running, you will see:

```
=======================================================
  FINAL SCORE  (put this in your paper)
=======================================================
  LoRA (Rinkle)  →  CSR = 0.7240  (±0.0312)
  Full FT (Mark) →  CSR = 0.6890  (±0.0287)

  Winner: LoRA  (difference = 0.0350)
=======================================================

  Per-Category Breakdown:
  Category                   LoRA    Full FT      Winner
  ---------------------------------------------------------
  Length                   0.8100     0.7800    LoRA ✓
  Keyword                  0.6500     0.6200    LoRA ✓
  Start_With               0.9100     0.8900    LoRA ✓
  End_With                 0.8800     0.8500    LoRA ✓
  Format                   0.7200     0.6900    LoRA ✓
  Tone                     0.9500     0.9400    LoRA ✓
  Style                    1.0000     1.0000     Tie
  Role_Playing             1.0000     1.0000     Tie
  Numerical_Constraints    1.0000     1.0000     Tie

=======================================================
  Category wins → LoRA: 6 | Full FT: 0 | Ties: 3
=======================================================
```

> **Note:** The numbers above are examples. Your actual scores will differ.

---

## What goes in the paper

**Use these outputs directly in your results section:**

1. `kfold_comparison.csv` → Table of CSR scores per constraint category
2. `kfold_comparison_plot.png` → Figure showing the comparison charts
3. The Final Score printed at the end → headline result in the abstract

**How to cite the method in your paper:**

> We evaluate both models using 5-fold cross validation on RECAST-30K, sampling 200 examples per fold (1,000 total). We report the mean Constraint Satisfaction Rate (CSR) and standard deviation across folds for each model and each constraint category.

---

## Notebook structure — step by step

| Cell | Name | What it does |
|------|------|-------------|
| 1 | Install & Login | Installs packages, logs into HuggingFace |
| 2 | Imports | Loads Python libraries |
| 3 | Configuration | **Update your 3 paths here** |
| 4 | Load Dataset | Reads RECAST-30K, creates 5 folds of 200 examples each |
| 5 | Constraint Validators | Defines functions that check each constraint type |
| 6 | Generation Functions | Defines how to generate responses and compute CSR |
| 7 | K-Fold Evaluation | **Main loop — runs both models on all 5 folds** |
| 8 | Compute Scores | Averages fold scores → final CSR per model |
| 9 | Plot Results | Creates 3 comparison charts |
| 10 | Final Summary | Prints the headline scores for the paper |

---

## Frequently asked questions

**Q: What if Colab crashes during Step 7?**
Results are saved after every fold. Look in `/content/kfold_results/` for `results_fold_1.json`, `results_fold_2.json` etc. You can reload completed folds and continue from where you left off.

**Q: Can I run fewer folds to save time?**
Yes — change `K_FOLDS = 3` in Step 3. This reduces time to ~50 minutes but gives less reliable scores.

**Q: Can I test more examples per fold?**
Yes — change `SAMPLES_PER_FOLD = 500` for more reliable scores at the cost of longer runtime.

**Q: Why do Style, Role_Playing, and Numerical always show 1.0?**
These constraint types are too complex to check automatically (you would need a separate LLM as a judge). They pass through as 1.0 and are noted as limitations in the paper.

**Q: Why does the model need to be deleted between folds?**
T4 GPU has 16GB VRAM. Each model uses ~4GB. Keeping both loaded at once risks running out of memory, so we alternate: load LoRA → evaluate → delete → load Full FT → evaluate → delete.

---

*STAT 453 — Deep Learning | Spring 2026 | University of Wisconsin-Madison*
