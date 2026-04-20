# RECAST: Framework for Multi-Constraint Instruction Following in LLMs

## What Problem Does It Solve?

Existing instruction-following datasets cap out at ~10 constraints per instruction. Real-world prompts are often much more complex. RECAST pushes this boundary by creating a dataset (RECAST-30K) with an average of **13.4 constraints per instruction** across **19 constraint types**, and provides a framework to automatically verify whether a model's response satisfies each constraint.

---

## The Dataset: RECAST-30K

- ~30,000 instruction-response pairs
- Constraints are derived from real prompt-response pairs (Tülu 3 Persona IF dataset), not synthetically invented ,  this keeps them grounded and realistic
- Cost to create: ~$175 (scalable and reproducible)

---

## Two Types of Constraints

### 1. Rule-Based Constraints
Objective, deterministic. Can be checked with a simple Python script.

| Type | Example |
|------|---------|
| Length | "Provide exactly 5 sentences" |
| Format | "Answer in JSON format", "Use markdown syntax" |
| Case | "Use lowercase letters only" |
| Structure | "Format your answer as a list" |

**Verification:** Rule-based validators (pattern matching, counting, structural checks).

### 2. Model-Based Constraints
Subjective, semantic. Need an LLM to judge.

| Type | Example |
|------|---------|
| Tone | "Use a formal tone", "Be humorous" |
| Emotion | "Express sadness", "Show excitement" |
| Factuality | "Ensure the response contains accurate facts" |
| Helpfulness | "The reply should be helpful and relevant" |
| Role-playing | "Reply as an expert", "Simulate a customer service agent" |
| Style | "Avoid slang", "Use a conversational style" |

**Verification:** Specialized LLM validators prompted with criteria to assess compliance.

---

## How the Dataset Is Built (Pipeline)

1. **Seed data** :  Start with real instruction-response pairs from Tülu 3 Persona IF
2. **Constraint extraction** :  Pull both rule-based and model-based constraints from existing responses
3. **Constraint selection** :  Use LLMs to pick relevant constraints for each instruction
4. **Instruction augmentation** :  Integrate selected constraints into the original prompt naturally (not just appended as a checklist)
5. **Response generation** :  Multiple LLMs generate candidate responses; majority voting picks the best one
6. **Quality control** :  Multi-model voting at both instruction synthesis and response generation stages

---

## Evaluation: RECAST-Test

- Tiered difficulty levels based on **constraint nesting** (basic → comprehensive)
- Tests how well models hold up as the number and complexity of constraints increase
- Every constraint is automatically verifiable, so evaluation is fully automated

---

## Training Method: RLVC (Reinforcement Learning with Verifiable Constraints)

- Uses fine-grained, **per-constraint reward signals** (not just a single overall score)
- Each constraint in the response is individually checked and rewarded
- This lets the model learn which types of constraints it struggles with

---

## Key Results

- Small models (Llama 3.1-8B) trained on RECAST-30K **outperform instruction-tuned larger models**
- Mixing rule-based + model-based constraints during training gives better generalization than either alone
- Performance degrades as constraint count increases, but RECAST-trained models degrade more gracefully

---

## Why This Matters for Our Project

- We're using RECAST-30K as our evaluation/training dataset for constraint-following in small LLMs
- The verifiable constraint structure lets us compute fine-grained metrics (per-constraint pass rates, not just overall accuracy)
- The rule-based vs. model-based split maps directly to our evaluation framework design
- RLVC's per-constraint rewards are relevant if we explore LoRA/SFT fine-tuning for constraint adherence