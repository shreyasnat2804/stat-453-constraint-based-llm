# =============================================================================
# LoRA Fine-Tuning Base Script — STAT 453, Spring 2026
# Model: Llama 3.2 1B Instruct | Dataset: RECAST-30K
#I have made it as simple as it is - like I have added lot of comments to make it easier to understand for students who are new to this. I have also made it modular so that they can easily change the dataset, model, and hyperparameters without having to dig into the code.
# WHAT THIS SCRIPT DOES (plain English):
#   1. Takes the prepared RECAST dataset (from Shreyas)
#   2. Loads Llama 3.2 1B as a frozen base model
#   3. Injects small trainable LoRA adapter layers into the attention weights
#   4. Trains ONLY those adapter layers (not the full model)
#   5. Saves the tiny adapter file (~30-60 MB) for evaluation
#
# HOW TO RUN:
#   pip install transformers peft datasets accelerate trl
#   python lora_base.py
# =============================================================================

import torch
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset, DatasetDict
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

# --- Model ---
# This is the Hugging Face model ID for Llama 3.2 1B Instruct.
# "Instruct" means it's already been fine-tuned to follow instructions —
# we're building on top of that, not starting from scratch.
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# --- Dataset ---
# UPDATE THIS when Shreyas shares the prepared RECAST-30K dataset.
# Options:
#   - HF repo:    "shreyas-hf-username/recast-30k-prepared"
#   - Local path: "./data/recast_prepared"
DATASET_PATH = "./datalora.json"   # <-- update with Shreyas's output

# --- LoRA Hyperparameters ---
# RANK (r): How many "dimensions" the adapter uses. Higher = more expressive
# but more memory. We sweep r=8 and r=16 per the proposal.
# Think of rank like the resolution of the update — 8 is standard, 16 is richer.
LORA_RANK = 8

# ALPHA: A scaling factor applied to the LoRA update. Standard practice is
# alpha = 2 * rank. Do not change this unless you have a reason to.
LORA_ALPHA = LORA_RANK * 2   # = 16 when rank=8

# DROPOUT: Randomly zeroes some adapter weights during training to prevent
# overfitting. 0.05 (5%) is a safe default for fine-tuning.
LORA_DROPOUT = 0.05

# TARGET MODULES: Which weight matrices inside each Transformer layer to adapt.
# q_proj = query projection, v_proj = value projection.
# These are the two most impactful in attention — start here, expand if needed.
# Full set would be: ["q_proj", "k_proj", "v_proj", "o_proj"]
TARGET_MODULES = ["q_proj", "v_proj"]

# --- Training Hyperparameters ---
# LEARNING RATE: How fast the adapter weights update each step.
# We sweep three values: 1e-4 (fast), 5e-5 (medium), 1e-5 (slow/careful).
# Start with 1e-4, then tune based on validation loss curves.
LEARNING_RATE = 1e-4

# EPOCHS: How many full passes through the training data.
# 1 epoch is enough for the proposal's main runs. 2 if you want to push further.
NUM_EPOCHS = 1

# BATCH SIZE: How many training examples the model sees per update step.
# We use gradient accumulation to simulate a batch of 32 on a single GPU.
# per_device = 8, accumulation steps = 4 → effective batch = 8 * 4 = 32
PER_DEVICE_BATCH_SIZE = 8
GRAD_ACCUMULATION_STEPS = 4   # effective batch size = 32

# MAX SEQUENCE LENGTH: Inputs longer than this are truncated.
# 512 tokens covers most RECAST instructions + responses comfortably.
MAX_SEQ_LENGTH = 512

# --- Output ---
OUTPUT_DIR = f"./outputs/lora_r{LORA_RANK}_lr{LEARNING_RATE}"


# =============================================================================
# SECTION 2: PROMPT FORMATTING
# =============================================================================
# Llama 3 uses a specific chat format with special tokens. We must match this
# EXACTLY — if the format is wrong, the model doesn't know where instructions
# end and responses begin, and training quality drops sharply.
#
# Format:
#   <|begin_of_text|>
#   <|start_header_id|>user<|end_header_id|>
#   {the instruction / constraint-following prompt}
#   <|eot_id|>
#   <|start_header_id|>assistant<|end_header_id|>
#   {the correct response}
#   <|eot_id|>

PROMPT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{response}"
    "<|eot_id|>"
)


def format_example(example: dict) -> dict:
    """
    Converts one RECAST row into a formatted training string.

    RECAST rows look like:
        {"instruction": "Write a 3-sentence summary in a formal tone...",
         "response":    "The report outlines three key findings..."}

    This function wraps them in Llama's chat format so the model learns to
    produce the right response given the instruction.

    NOTE: If Shreyas's dataset uses different field names (e.g. "prompt"
    instead of "instruction"), update the .get() calls below.
    """
    #instruction = example.get("instruction", example.get("prompt", ""))
    #response = example.get("response", example.get("output", ""))
    #return {
    #    "text": PROMPT_TEMPLATE.format(instruction=instruction, response=response)
    #}
    # Your data uses "input" and "output" fields (not "instruction"/"response")
    instruction = example.get("input", "")
    response = example.get("output", "")
    return {
        "text": PROMPT_TEMPLATE.format(instruction=instruction, response=response)
    }


# =============================================================================
# SECTION 3: LOAD AND PREPARE THE DATASET
# =============================================================================
"""
def build_dataset() -> DatasetDict:
    """
"""
    Loads RECAST-30K, applies formatting, and splits into train/validation.

    Returns a DatasetDict with two keys:
        "train"      → 90% of the data, used for gradient updates
        "validation" → 10% of the data, used to monitor overfitting
    """
"""
    print(f"\n[1/4] Loading dataset from: {DATASET_PATH}")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    print(f"      Raw examples: {len(dataset)}")

    # Apply the prompt template to every example.
    # remove_columns drops the original fields so only "text" remains.
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # Shuffle before splitting so the val set is a random sample, not
    # just the last 10% (which could be biased toward certain constraint types).
    dataset = dataset.shuffle(seed=42)
    split = dataset.train_test_split(test_size=0.1, seed=42)

    print(f"      Train: {len(split['train'])} | Val: {len(split['test'])}")
    return DatasetDict({"train": split["train"], "validation": split["test"]})
"""
import json
from datasets import Dataset

def build_dataset() -> DatasetDict:
    print(f"\n[1/4] Loading dataset from: {DATASET_PATH}")

    # Read the JSON file manually and extract only "input" and "output"
    # This bypasses PyArrow's schema casting issue with deeply nested fields
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Extract only the two fields we need, skip everything else
    clean = []
    for row in raw:
        instruction = row.get("input", "")
        response    = row.get("output", "")
        if instruction and response:   # skip any rows missing either field
            clean.append({
                "text": PROMPT_TEMPLATE.format(
                    instruction=instruction,
                    response=response
                )
            })

    print(f"      Raw examples loaded: {len(clean)}")

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(clean)
    dataset = dataset.shuffle(seed=42)
    split   = dataset.train_test_split(test_size=0.1, seed=42)

    print(f"      Train: {len(split['train'])} | Val: {len(split['test'])}")
    return DatasetDict({"train": split["train"], "validation": split["test"]})
# =============================================================================
# SECTION 4: LOAD MODEL AND TOKENIZER
# =============================================================================

def load_model_and_tokenizer():
    """
    Loads the Llama 3.2 1B model and its tokenizer.

    Key decisions made here:
    - bfloat16: Half-precision format supported by A100 GPUs. Cuts memory in
      half vs float32 with minimal accuracy loss.
    - device_map="auto": Automatically places model layers on available GPUs.
    - use_cache=False: The KV-cache speeds up inference but breaks gradient
      flow during training — must disable it.
    """
    print(f"\n[2/4] Loading model: {MODEL_NAME}")

    # The tokenizer converts text → token IDs the model understands.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Llama 3 has no dedicated pad token — we reuse the end-of-sequence token.
    # padding_side="right" is required for causal language model training.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()  # Required when using PEFT + gradient checkpointing

    # How much GPU memory are we using just for the model weights?
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_reserved() / 1e9
        print(f"      Model loaded. GPU memory reserved: {mem_gb:.1f} GB")

    return model, tokenizer


# =============================================================================
# SECTION 5: INJECT LORA ADAPTERS
# =============================================================================

def apply_lora(model):
    """
    Freezes all original model weights and injects LoRA adapters.

    HOW LoRA WORKS (simple version):
        For each target weight matrix W (shape: d_in × d_out),
        LoRA adds two small matrices:
            A: shape (d_in × rank)
            B: shape (rank × d_out)
        The effective weight update is: ΔW = B × A × (alpha / rank)
        The original W is FROZEN — only A and B are trained.

    WHY THIS MATTERS FOR US:
        Llama 3.2 1B has ~1.24 billion parameters.
        With LoRA r=8 on q_proj + v_proj, we train ~4-8 million parameters.
        That's ~0.5% of the model — much cheaper, fits on a single A100 40GB.

    WHAT print_trainable_parameters() WILL SHOW:
        trainable params: ~4-8M || all params: ~1.24B || trainable%: ~0.5%
    """
    print(f"\n[3/4] Applying LoRA adapters (rank={LORA_RANK}, alpha={LORA_ALPHA})")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,   # we're fine-tuning a causal language model
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",                    # do not train bias terms (keeps things simple)
        inference_mode=False,           # we're training, not just doing inference
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # shows the ~0.5% stat described above
    return model


# =============================================================================
# SECTION 6: TRAIN
# =============================================================================

def train(model, tokenizer, dataset: DatasetDict):
    """
    Runs the supervised fine-tuning loop.

    SFTTrainer (from the trl library) wraps HuggingFace's Trainer and handles:
    - Tokenising the "text" field automatically
    - Masking the prompt portion (we only compute loss on the response)
    - Logging train/val loss every N steps

    WHAT TO WATCH:
    - Train loss should decrease steadily
    - Val loss should also decrease but more slowly
    - If val loss starts RISING while train loss falls → overfitting
      → try lowering LR or stopping early (EarlyStoppingCallback handles this)
    """
    print(f"\n[4/4] Starting training | LR={LEARNING_RATE} | Epochs={NUM_EPOCHS}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=10,              # replaced warmup_ratio     # gradually reduces LR over training
        #warmup_ratio=0.03,                # slowly ramp up LR for the first 3% of steps
        weight_decay=0.01,                # mild regularisation to prevent overfitting
        #bf16=True,                        # use bfloat16 on A100
        optim="adamw_torch",              # standard optimiser for LLM fine-tuning
        logging_steps=50,                 # print train loss every 50 steps
        eval_strategy="steps",
        eval_steps=200,                   # check val loss every 200 steps
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,      # keep the checkpoint with best val loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,          # lower loss = better
        save_total_limit=2,               # only keep 2 checkpoints to save disk space
        report_to="none",                 # set to "wandb" if you want experiment tracking
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()
    return trainer


# =============================================================================
# SECTION 7: SAVE ADAPTER WEIGHTS
# =============================================================================

def save_adapter(trainer):
    """
    Saves ONLY the LoRA adapter weights — not the full 2 GB base model.

    The adapter folder will be ~30-60 MB and contains:
        adapter_config.json   → LoRA settings (rank, alpha, target modules)
        adapter_model.bin     → the trained A and B matrices
        tokenizer files       → so the adapter can be loaded standalone

    To load later for evaluation:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        model = PeftModel.from_pretrained(base, "./outputs/.../lora_adapter")

    SHARE THIS FOLDER with Mark for the LoRA vs. Full FT comparison.
    """
    import os
    adapter_path = os.path.join(OUTPUT_DIR, "lora_adapter")
    trainer.model.save_pretrained(adapter_path)
    trainer.tokenizer.save_pretrained(adapter_path)
    print(f"\nAdapter saved to: {adapter_path}")
    print("Share this folder with Mark for the comparison runs.")
    return adapter_path


# =============================================================================
# MAIN
# =============================================================================
"""
if __name__ == "__main__":
    print("=" * 60)
    print("  LoRA Fine-Tuning — STAT 453 / RECAST Project")
    print(f"  Model:  {MODEL_NAME}")
    print(f"  Rank:   {LORA_RANK}  |  Alpha: {LORA_ALPHA}  |  LR: {LEARNING_RATE}")
    print("=" * 60)

    dataset           = build_dataset()
    model, tokenizer  = load_model_and_tokenizer()
    model             = apply_lora(model)
    trainer           = train(model, tokenizer, dataset)
    adapter_path      = save_adapter(trainer)

    print("\nDone! Next steps:")
    print(f"  1. Check val loss curve in {OUTPUT_DIR}/")
    print(f"  2. Run CSR evaluation on RECAST test split using saved adapter")
    print(f"  3. Share {adapter_path} with Mark")
 """

if __name__ == "__main__": #only to test
    print("=" * 60)
    print("  LoRA Fine-Tuning — STAT 453 / RECAST Project")
    print(f"  Model:  {MODEL_NAME}")
    print(f"  Rank:   {LORA_RANK}  |  Alpha: {LORA_ALPHA}  |  LR: {LEARNING_RATE}")
    print("=" * 60)

    # Step 1: Test dataset only
    dataset = build_dataset()
    print("\n✓ Dataset loaded successfully")
    print(f"  Sample text preview:\n  {dataset['train'][0]['text'][:200]}")

    # Step 2: Test model loads
    model, tokenizer = load_model_and_tokenizer()
    print("\n✓ Model loaded successfully")

    # Step 3: Test LoRA injection only
    model = apply_lora(model)
    print("\n✓ LoRA adapters injected successfully")

    print("\n" + "=" * 60)
    print("  Pipeline smoke test PASSED.")
    print("  All 3 stages work. Run full training on Colab A100.")
    print("=" * 60)
