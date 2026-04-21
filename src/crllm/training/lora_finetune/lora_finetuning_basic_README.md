# LoRA Fine-Tuning — Team Reference Guide
**STAT 453, Spring 2026 | University of Wisconsin-Madison**  
*For: Shreyas, Gayathri, Mark, Rinkle*

---

## Table of Contents
1. [Team Division of Work](#1-team-division-of-work)
2. [What is LoRA?](#2-what-is-lora)
3. [What Each Part of the Script Does](#3-what-each-part-of-the-script-does)
4. [LoRA Parameters Reference](#4-lora-parameters-reference)
5. [How to Run](#5-how-to-run)
6. [After Training — What to Report](#6-after-training--what-to-report)
7. [Glossary](#7-glossary)

---

## 1. Team Division of Work

The pipeline flows left to right — each person's output becomes the next person's input.

| Name | Role | Responsibility |
|------|------|----------------|
| **Shreyas** | Data preparation | Clean and format RECAST-30K into `instruction` / `response` pairs |
| **Gayathri** | Data augmentation | Augment data and mix with Tulu for generalization |
| **Mark** | Full fine-tuning base | Build base training setup for full FT (comparison baseline) |
| **Rinkle** | LoRA fine-tuning | Build LoRA base, run rank/LR sweep, evaluate CSR |

> **Handoff chain:**  
> Shreyas formats RECAST-30K → Gayathri augments it → Rinkle (LoRA) and Mark (Full FT) both train on that same dataset → results are compared together for the paper.

---

## 2. What is LoRA?

**LoRA** stands for **Low-Rank Adaptation**. It is a fine-tuning technique that avoids updating every weight in a large model. Instead, it freezes the original model completely and trains a pair of tiny add-on matrices called **adapters**.

### The Core Idea

Every Transformer layer has large weight matrices. For example, the query projection matrix `W` in the attention mechanism might be shape `2048 × 2048`. Normally, fine-tuning updates all of these — hundreds of millions of parameters.

LoRA instead learns two small matrices **A** and **B**, where:

```
Effective update  =  B × A   (scaled by alpha / rank)
```

The original `W` is **frozen** (never updated). Only `A` and `B` are trained. Because rank is small (8 or 16), `A` and `B` together are tiny compared to `W`.

### Concrete Numbers for Our Project

> Llama 3.2 1B has **1.24 billion** total parameters.  
> With LoRA `rank=8` on `q_proj` and `v_proj`, we train roughly **4–8 million** parameters — less than 1% of the model.  
> This fits on a **single A100 40GB** and trains in **2–3 hours** instead of 4–6 hours for full fine-tuning.

### Why LoRA Instead of Training Everything?

- **Much cheaper** — fits on one GPU, trains faster, uses less memory
- **Less forgetting** — the base model is frozen, so it retains general language ability
- **Portable** — the adapter file is ~30–60 MB; you share that, not a 2 GB model
- **Comparable quality** — at 7B+ scale, LoRA matches full fine-tuning on many tasks. At 1B (our scale), that comparison is one of the open questions we are investigating

### The Open Question We Are Answering

RECAST fine-tuning has been tested at 7B–8B scale and works well. Nobody has tested it at 1B. Our project asks: **can LoRA's rank-8 or rank-16 updates capture the multi-dimensional constraint-following behavior that RECAST teaches?** That is the scientific contribution of this project.

---

## 3. What Each Part of the Script Does

The LoRA base script (`lora_base.py`) is divided into 7 sections.

### Section 1 — Configuration
All the numbers that control training: rank, learning rate, batch size, output folder. **You only ever need to change this section** — nothing else. The whole script reads from these variables at the top.

### Section 2 — Prompt Formatting
Llama 3 expects a specific chat format with special tokens like `<|start_header_id|>`. If the format is wrong, the model does not know where the instruction ends and the response begins. This section wraps every RECAST example into that format correctly.

```
<|begin_of_text|>
<|start_header_id|>user<|end_header_id|>

{instruction}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{response}
<|eot_id|>
```

> **What Shreyas needs to provide:**  
> Each row in the prepared RECAST dataset should have two fields: `instruction` (the constraint-following prompt) and `response` (the correct output). If field names differ, update the `.get()` calls in `format_example()`.

### Section 3 — Load Dataset
Loads RECAST-30K from Hugging Face (or a local path), applies the prompt format to every row, shuffles, and splits **90% for training** and **10% for validation**.

### Section 4 — Load Model and Tokenizer
Downloads Llama 3.2 1B Instruct in `bfloat16` (half-precision) and places it on the GPU. Also loads the tokenizer, which converts text into token IDs the model understands. The padding token is set explicitly because Llama 3 does not have a default one.

### Section 5 — Inject LoRA Adapters
This is the core LoRA step. It calls `get_peft_model()` which:
- Freezes all original model weights
- Adds the `A` and `B` adapter matrices to `q_proj` and `v_proj` in every attention layer

After this, **only the adapter matrices receive gradients** during training.

```python
model.print_trainable_parameters()
# trainable params: 6,291,456 || all params: 1,236,174,848 || trainable%: 0.51%
```

### Section 6 — Train
The actual training loop. `SFTTrainer` from the `trl` library handles tokenization, loss computation, gradient updates, validation, and checkpointing automatically.

**What to watch during training:**
- Train loss should decrease steadily across steps
- Validation loss should also decrease, slightly slower
- If validation loss starts **rising** while train loss keeps falling → that is overfitting → stop early

### Section 7 — Save Adapter Weights
Saves only the trained `A` and `B` matrices — not the full 2 GB base model. The output folder is ~30–60 MB and contains:

```
lora_adapter/
├── adapter_config.json     ← LoRA settings (rank, alpha, target modules)
├── adapter_model.bin       ← the trained A and B matrices
└── tokenizer files         ← so the adapter can be loaded standalone
```

> **Share this folder with Mark** for the LoRA vs. Full FT comparison evaluation.

---

## 4. LoRA Parameters Reference

| Parameter | Value | What it means |
|-----------|-------|----------------|
| `rank (r)` | 8 and 16 | Controls how many dimensions the adapter uses. Higher = richer updates, more memory. |
| `alpha` | 2 × rank (16 or 32) | Scaling factor on the adapter output. Standard practice is `alpha = 2r`. |
| `dropout` | 0.05 | Randomly zeroes 5% of adapter weights each step to reduce overfitting. |
| `target_modules` | `q_proj`, `v_proj` | The attention weight matrices we adapt. Query + Value projections are most impactful. |
| `learning_rate` | `1e-4`, `5e-5`, `1e-5` | How fast adapter weights update. We sweep three values to find the best. |
| `epochs` | 1–2 | How many full passes through the training data. |
| `batch_size` | 32 (effective) | 8 per device × 4 gradient accumulation steps = 32 effective. |

---

## 5. How to Run

### Setup (run once on Colab)

```bash
pip install transformers peft datasets accelerate trl
huggingface-cli login   # paste your HF token — needed for Llama 3.2 gated access
```

> **Colab runtime setting:** Go to Runtime → Change runtime type → select **A100 GPU**.  
> All LoRA runs fit on a single A100 40GB. Do not use T4 — it is too small for this model.

### Main Run (rank 8, learning rate 1e-4)

```bash
python lora_base.py
```

### Hyperparameter Sweep

Edit `LORA_RANK` and `LEARNING_RATE` at the top of the script and re-run. The output folder name auto-includes rank and LR so runs do not overwrite each other.

```python
# Run 1: rank=8,  lr=1e-4  (default — no changes needed)
# Run 2: rank=16, lr=1e-4  → set LORA_RANK = 16
# Run 3: rank=8,  lr=5e-5  → set LEARNING_RATE = 5e-5
# Run 4: rank=8,  lr=1e-5  → set LEARNING_RATE = 1e-5
```

### Expected Training Time per Run (A100 40GB)

| Configuration | Time per epoch |
|---------------|----------------|
| LoRA rank=8 | ~2–3 hours |
| LoRA rank=16 | ~3–4 hours |
| Full fine-tuning (Mark's runs) | ~4–6 hours |

---

## 6. After Training — What to Report

After each run, record the following into the shared team spreadsheet:

- [ ] Final training loss and validation loss
- [ ] GPU memory peak (printed by the script before training starts)
- [ ] Total training time in hours
- [ ] CSR (Constraint Satisfaction Rate) per constraint category on the RECAST test split
- [ ] Hard CSR (fraction of responses satisfying ALL constraints simultaneously)

> **Sharing with Mark:**  
> After your best run, share the `lora_adapter/` folder. Mark loads it on top of the same Llama 3.2 1B base model and runs both your LoRA and his full fine-tuning through the same evaluation pipeline. That is how the LoRA vs. Full FT comparison stays apples-to-apples.

---

## 7. Glossary

| Term | Definition |
|------|------------|
| **RECAST-30K** | The 30,000-example dataset where every training instance has multiple simultaneous constraints, each with an automated validator. |
| **CSR** | Constraint Satisfaction Rate — the fraction of individual constraints satisfied in a model's response. E.g., 2 out of 3 constraints satisfied = CSR of 0.67. |
| **Hard CSR** | The fraction of responses that satisfy ALL constraints simultaneously. Harder to achieve than CSR. |
| **IFEval** | A benchmark for instruction-following. Used to check that fine-tuning on RECAST did not make the model worse at general instruction following. |
| **Adapter** | The small set of trained LoRA matrices (A and B). Saved separately from the frozen base model. |
| **bfloat16** | A 16-bit floating point format supported by A100 GPUs. Cuts memory roughly in half vs float32 with minimal accuracy loss. |
| **SFTTrainer** | Supervised Fine-Tuning Trainer from the `trl` library. Handles the training loop boilerplate automatically. |
| **q_proj / v_proj** | Query and value projection matrices inside each Transformer attention layer — the two most impactful targets for LoRA adaptation. |

---

## Repository Structure

```
recast-lora/
├── README.md                  ← this file
├── lora_base.py               ← Rinkle's LoRA training script
├── data/
│   └── recast_prepared/       ← Shreyas's formatted dataset goes here
└── outputs/
    └── lora_r8_lr1e-4/
        └── lora_adapter/      ← saved adapter weights (share with Mark)
```

---

*STAT 453 — Deep Learning | Spring 2026 | University of Wisconsin-Madison*
