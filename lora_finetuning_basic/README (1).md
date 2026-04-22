# LoRA Fine-Tuning Pipeline
**STAT 453, Spring 2026 | University of Wisconsin-Madison**

This folder contains Rinkle's LoRA fine-tuning pipeline for Llama 3.2 1B Instruct on the RECAST-30K dataset. It is one half of the LoRA vs. Full Fine-Tuning comparison — Mark's full fine-tuning pipeline is the other half.

---

## How This Fits Into the Full Project Pipeline

```
Shreyas (Data Prep)
        │
        │  recast_prepared.jsonl
        ▼
Rinkle (LoRA Fine-Tuning)  ──────────────────────────────┐
        │                                                 │
        │  lora_adapter/                                  │
        ▼                                                 ▼
Mark (Full Fine-Tuning)                        Compare & Cross-Validate
        │                                      CSR, Hard CSR per category
        │  full_ft_model/                       (LoRA vs Full FT)
        └──────────────────────────────────────┘
```

**The question we are answering:** Can LoRA's parameter-efficient updates (training <1% of weights) match full fine-tuning on multi-constraint instruction following at 1B scale?

---

## Folder Structure

```
lora_finetuning_basic/
├── README.md           ← this file
├── lora_base.py        ← main training script
└── datalora.json       ← sample data for local smoke testing
```

---

## What This Script Does

1. Loads the prepared RECAST dataset from Shreyas (`input` + `output` fields)
2. Formats every example into Llama 3's chat template
3. Loads Llama 3.2 1B Instruct in bfloat16 and freezes all weights
4. Injects LoRA adapter matrices into `q_proj` and `v_proj` attention layers
5. Trains only the adapter weights (~4–8M parameters, <1% of the model)
6. Saves the adapter folder for evaluation and handoff to Mark

---

## Setup

### Requirements

```bash
pip install transformers peft datasets accelerate trl
```

### HuggingFace Login

Llama 3.2 is a gated model — you need to request access and log in before running.

**Step 1** — Request access at:
```
https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
```
Access is granted instantly after agreeing to Meta's license terms.

**Step 2** — Log in from your terminal:
```bash
hf auth login
```
Get your token from `https://huggingface.co/settings/tokens` and paste it when prompted.

---

## How to Run

### Local smoke test (CPU — just verifies pipeline works)

```bash
python lora_base.py
```

This runs through all 4 stages — dataset loading, model loading, LoRA injection, and a short training loop — to confirm everything is wired up correctly. Use your local `datalora.json` sample for this.

> **Note:** CPU training is extremely slow for a 1B model. Local runs are for testing only. Do all real training on Colab A100.

### Full training run (Colab A100)

1. Upload `lora_base.py` and Shreyas's full `recast_prepared.jsonl` to Colab
2. Set runtime to **A100 GPU** (Runtime → Change runtime type → A100)
3. Install requirements and log in to HuggingFace
4. Update `DATASET_PATH` in Section 1 to point to the full dataset
5. Run the script

---

## Configuration

All settings live at the top of `lora_base.py` in **Section 1**. Change values there — do not hardcode anything deeper in the script.

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `LORA_RANK` | `8` | Adapter expressiveness. Sweep: `8` and `16` |
| `LORA_ALPHA` | `16` (2 × rank) | Scaling factor on adapter output |
| `LORA_DROPOUT` | `0.05` | Regularisation — prevents overfitting |
| `TARGET_MODULES` | `q_proj, v_proj` | Which attention layers to adapt |
| `LEARNING_RATE` | `1e-4` | Sweep: `1e-4`, `5e-5`, `1e-5` |
| `NUM_EPOCHS` | `1` | Full passes through training data |
| `DATASET_PATH` | `./datalora.json` | Path to Shreyas's prepared dataset |

### Hyperparameter sweep runs

```python
# Run 1 — default
LORA_RANK = 8,  LEARNING_RATE = 1e-4

# Run 2
LORA_RANK = 16, LEARNING_RATE = 1e-4

# Run 3
LORA_RANK = 8,  LEARNING_RATE = 5e-5

# Run 4
LORA_RANK = 8,  LEARNING_RATE = 1e-5
```

Each run saves to its own output folder automatically so runs never overwrite each other.

---

## Output

After training, the adapter is saved to:
```
outputs/lora_r{rank}_lr{lr}/lora_adapter/
├── adapter_config.json     ← LoRA settings (rank, alpha, target modules)
├── adapter_model.bin       ← trained A and B matrices (~30–60 MB)
└── tokenizer files
```

> **For Mark:** Share the `lora_adapter/` folder. Mark loads it on top of the same Llama 3.2 1B base to run the LoRA vs. Full FT comparison on identical evaluation data.

Loading the adapter for evaluation:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base  = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = PeftModel.from_pretrained(base, "./outputs/lora_r8_lr1e-4/lora_adapter")
```

---

## Cross-Validation with Mark's Full Fine-Tuning

Once both pipelines have run on the same dataset, compare them using the same evaluation script on the same RECAST test split.

### What to measure

| Metric | Description |
|--------|-------------|
| **CSR** | Constraint Satisfaction Rate — fraction of individual constraints satisfied per response |
| **Hard CSR** | Fraction of responses satisfying ALL constraints simultaneously |
| **Per-category CSR** | CSR broken down across all 8 constraint types (Length, Format, Keyword, Style, Tone, Role-Playing, Start/End-With, Numerical) |
| **IFEval score** | General instruction-following — checks neither pipeline forgot how to follow basic instructions |

### Comparison table to fill in

| Configuration | CSR | Hard CSR | IFEval | GPU Mem | Train Time |
|---------------|-----|----------|--------|---------|------------|
| LoRA rank=8, lr=1e-4 | | | | | |
| LoRA rank=16, lr=1e-4 | | | | | |
| LoRA rank=8, lr=5e-5 | | | | | |
| Full Fine-Tuning (Mark) | | | | | |

### How to keep the comparison fair

- Both pipelines must use the **exact same train/test split** — use `seed=42` everywhere
- Both pipelines must run on the **same base model** — `meta-llama/Llama-3.2-1B-Instruct`
- Evaluation must use the **same evaluation script** on the **same test examples**
- Generation must use **greedy decoding** (`do_sample=False`) for reproducibility

---

## Expected Training Time (Colab A100 40GB)

| Configuration | Memory | Time per epoch |
|---------------|--------|----------------|
| LoRA rank=8 | ~7 GB | ~2–3 hours |
| LoRA rank=16 | ~7 GB | ~3–4 hours |
| Full fine-tuning (Mark) | ~16 GB | ~4–6 hours |

---

## Common Errors and Fixes

| Error | Fix |
|-------|-----|
| `DatasetNotFoundError` | Add `"json", data_files=DATASET_PATH` to `load_dataset()` call |
| `GatedRepoError 401` | Run `hf auth login` and paste your HF token |
| `evaluation_strategy` unexpected keyword | Rename to `eval_strategy` |
| `tokenizer` unexpected keyword in SFTTrainer | Rename to `processing_class` |
| `dataset_text_field` unexpected keyword | Remove it — newer trl uses `text` column by default |
| `bf16` error on CPU | Remove `bf16=True` — only works on A100 GPU |
| `warmup_ratio` deprecated | Replace with `warmup_steps=10` |

---

## Key Concepts

**LoRA (Low-Rank Adaptation):** Instead of updating all 1.24B weights, LoRA freezes the base model and trains two small matrices A and B per target layer. The effective weight update is `ΔW = B × A`. With rank=8, we train ~4–8M parameters (<1% of the model).

**Why this matters:** LoRA has been shown to match full fine-tuning at 7B+ scale. Whether it holds at 1B on multi-constraint tasks is the open research question this project addresses.

**CSR vs Hard CSR:** CSR measures partial credit (2 out of 3 constraints = 0.67). Hard CSR is all-or-nothing (all 3 constraints must pass = 1.0, otherwise 0.0). Hard CSR is the stricter and more meaningful metric for real-world use.

---

*STAT 453 — Deep Learning | Spring 2026 | University of Wisconsin-Madison*
