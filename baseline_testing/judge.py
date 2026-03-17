"""
LLM Judge module using Gemma-2-9B-Instruct to evaluate qualitative constraints
that rule-based checkers cannot handle.
"""

import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

JUDGE_MODEL_ID = "google/gemma-2-9b-it"


class JudgeModel:
    """Uses Gemma-2-9B-Instruct as an LLM judge for qualitative constraint evaluation."""

    def __init__(self, use_4bit=True):
        """Load the judge model and tokenizer.

        Args:
            use_4bit: If True, load with 4-bit quantization via BitsAndBytes.
                      Otherwise, load in bfloat16 with device_map="auto".
        """
        torch.cuda.empty_cache()

        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                JUDGE_MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                JUDGE_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model.eval()

        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"Judge model loaded. GPU VRAM used: {allocated:.2f} GB")

    def judge_constraint(self, prompt: str, response: str, constraint: dict) -> bool | None:
        """Evaluate whether a response satisfies a single constraint.

        Args:
            prompt: The original instruction given to the model.
            response: The model's generated response.
            constraint: A dict describing the constraint (with keys like
                        "type"/"constraint_type" and "value"/"description"/"requirement").

        Returns:
            True if PASS, False if FAIL, None if the verdict could not be determined.
        """
        try:
            constraint_type = constraint.get("type", constraint.get("constraint_type", "unknown"))
            constraint_value = constraint.get(
                "value",
                constraint.get("description", constraint.get("requirement", str(constraint))),
            )

            judge_prompt = (
                "You are evaluating whether a language model response satisfies a specific "
                "constraint. Answer with exactly one word: PASS or FAIL.\n\n"
                f"Original instruction:\n{prompt}\n\n"
                f"Model response:\n{response}\n\n"
                "Constraint to evaluate:\n"
                f"Type: {constraint_type}\n"
                f"Requirement: {constraint_value}\n\n"
                "Does the response satisfy this constraint? Answer PASS or FAIL only:\n"
            )

            inputs = self.tokenizer(judge_prompt, return_tensors="pt").to(self.model.device)
            input_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=4,
                )

            new_tokens = outputs[0][input_length:]
            verdict = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()

            if verdict.startswith("PASS"):
                return True
            elif verdict.startswith("FAIL"):
                return False
            else:
                return None

        except Exception:
            return None

    def judge_all_skipped(self, record: dict) -> dict:
        """Run the LLM judge on all constraints that were skipped by rule-based checking.

        Args:
            record: A dict with keys "prompt", "response", "constraints", and "results".
                    "results" is a list of dicts with "type" and "passed" keys.
                    Entries where "passed" is None are treated as skipped.

        Returns:
            A new dict (deep copy) with updated results and recalculated aggregate metrics.
            The input record is not mutated.
        """
        result = copy.deepcopy(record)

        prompt = result["prompt"]
        response = result["response"]
        constraints = result["constraints"]
        results = result["results"]

        for i, res in enumerate(results):
            if res["passed"] is None:
                constraint = constraints[i]
                verdict = self.judge_constraint(prompt, response, constraint)
                res["passed"] = verdict
                res["judged_by"] = "llm"

        num_checked = sum(1 for r in results if r["passed"] is not None)
        num_passed = sum(1 for r in results if r["passed"] is True)
        per_constraint_csr = num_passed / num_checked if num_checked > 0 else 0.0
        hard_csr = num_checked > 0 and all(
            r["passed"] is True for r in results if r["passed"] is not None
        )

        result["num_checked"] = num_checked
        result["num_passed"] = num_passed
        result["per_constraint_csr"] = per_constraint_csr
        result["hard_csr"] = hard_csr

        return result

    def unload(self):
        """Delete the model and tokenizer and free GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        print("Judge model unloaded and GPU cache cleared.")
