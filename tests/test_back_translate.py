"""Unit tests for back_translate.py"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

from src.crllm.dataset.augmentation.back_translate import (
    detect_prompt_field,
    extract_constraint_tokens,
    verify_constraint_preservation,
    run_back_translation_pipeline,
)

try:
    from transformers import MarianMTModel, MarianTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

requires_transformers = pytest.mark.skipif(
    not HAS_TRANSFORMERS, reason="transformers not installed"
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_RECORD = {
    "winner_prompt": "Write a 100 word essay about 'python' using bullet points.",
    "response_of_winner_prompt": "Here is the essay...",
    "added_constraint": {
        "Length": ["at most 100 words"],
        "Keyword": ["include 'python'"],
    },
    "added_constraint_num": 2,
    "rule_evaluate_dict": {},
    "id": "personas_IF_abc123",
}

SAMPLE_RECORD_INSTRUCTION = {
    "instruction": "Explain machine learning in 50 words.",
    "response_of_winner_prompt": "ML is...",
    "added_constraint": {"Length": ["at most 50 words"]},
    "added_constraint_num": 1,
    "rule_evaluate_dict": {},
    "id": "test_instr_001",
}


def _mock_translate_batch(texts, model, tokenizer, max_length=512):
    """Simple mock: uppercases text to simulate a transformation."""
    return [t.upper() for t in texts]


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# TestDetectPromptField
# ---------------------------------------------------------------------------


class TestDetectPromptField:
    def test_finds_winner_prompt(self):
        assert detect_prompt_field(SAMPLE_RECORD) == "winner_prompt"

    def test_finds_instruction(self):
        assert detect_prompt_field(SAMPLE_RECORD_INSTRUCTION) == "instruction"

    def test_raises_on_missing(self):
        with pytest.raises(KeyError):
            detect_prompt_field({"text": "hello"})


# ---------------------------------------------------------------------------
# TestExtractConstraintTokens
# ---------------------------------------------------------------------------


class TestExtractConstraintTokens:
    def test_extracts_numbers(self):
        tokens = extract_constraint_tokens(SAMPLE_RECORD)
        assert "100" in tokens

    def test_extracts_quoted_strings(self):
        tokens = extract_constraint_tokens(SAMPLE_RECORD)
        assert "python" in tokens

    def test_extracts_from_added_constraint(self):
        tokens = extract_constraint_tokens(SAMPLE_RECORD)
        # Category names
        assert "length" in tokens
        assert "keyword" in tokens
        # Words from descriptions
        assert "words" in tokens

    def test_handles_empty_added_constraint(self):
        record = {
            "winner_prompt": "Hello world 42",
            "added_constraint": {},
            "id": "empty",
        }
        tokens = extract_constraint_tokens(record)
        assert "42" in tokens

    def test_handles_missing_added_constraint(self):
        record = {
            "winner_prompt": "Hello world",
            "id": "no_constraint",
        }
        tokens = extract_constraint_tokens(record)
        # Should not raise; just returns whatever it can find
        assert isinstance(tokens, set)

    def test_extracts_format_keywords(self):
        tokens = extract_constraint_tokens(SAMPLE_RECORD)
        assert "bullet" in tokens


# ---------------------------------------------------------------------------
# TestVerifyConstraintPreservation
# ---------------------------------------------------------------------------


class TestVerifyConstraintPreservation:
    def test_all_tokens_preserved(self):
        assert verify_constraint_preservation(
            "Write 100 words about python",
            "Please write 100 words about python programming",
            {"100", "python"},
        )

    def test_missing_number(self):
        assert not verify_constraint_preservation(
            "Write 100 words",
            "Write many words",
            {"100"},
        )

    def test_case_insensitive(self):
        assert verify_constraint_preservation(
            "Use JSON format",
            "use json format",
            {"JSON"},
        )

    def test_empty_tokens(self):
        assert verify_constraint_preservation(
            "anything", "anything else", set()
        )


# ---------------------------------------------------------------------------
# TestIdempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_skips_existing_output(self, tmp_path):
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        _write_jsonl(str(input_file), [SAMPLE_RECORD])
        output_file.write_text("existing content")

        with patch(
            "src.crllm.dataset.augmentation.back_translate.load_translation_models"
        ) as mock_load:
            stats = run_back_translation_pipeline(
                input_path=str(input_file),
                output_path=str(output_file),
            )
            mock_load.assert_not_called()

        assert stats["skipped"] is True
        assert output_file.read_text() == "existing content"

    def test_force_overwrites(self, tmp_path):
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        _write_jsonl(str(input_file), [SAMPLE_RECORD])
        output_file.write_text("old content")

        mock_fwd_model = MagicMock()
        mock_bwd_model = MagicMock()
        mock_fwd_tok = MagicMock()
        mock_bwd_tok = MagicMock()

        with patch(
            "src.crllm.dataset.augmentation.back_translate.load_translation_models",
            return_value=(mock_fwd_model, mock_fwd_tok, mock_bwd_model, mock_bwd_tok),
        ):
            with patch(
                "src.crllm.dataset.augmentation.back_translate.translate_batch",
                side_effect=_mock_translate_batch,
            ):
                stats = run_back_translation_pipeline(
                    input_path=str(input_file),
                    output_path=str(output_file),
                    force=True,
                )

        assert stats["skipped"] is False
        assert output_file.read_text() != "old content"


# ---------------------------------------------------------------------------
# TestOutputSchema
# ---------------------------------------------------------------------------


class TestOutputSchema:
    """Test output record structure using mocked translation."""

    def _run_pipeline(self, tmp_path, records):
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        _write_jsonl(str(input_file), records)

        mock_fwd_model = MagicMock()
        mock_bwd_model = MagicMock()
        mock_fwd_tok = MagicMock()
        mock_bwd_tok = MagicMock()

        with patch(
            "src.crllm.dataset.augmentation.back_translate.load_translation_models",
            return_value=(mock_fwd_model, mock_fwd_tok, mock_bwd_model, mock_bwd_tok),
        ):
            with patch(
                "src.crllm.dataset.augmentation.back_translate.translate_batch",
                side_effect=_mock_translate_batch,
            ):
                stats = run_back_translation_pipeline(
                    input_path=str(input_file),
                    output_path=str(output_file),
                )

        output_records = []
        with open(str(output_file), "r") as f:
            for line in f:
                if line.strip():
                    output_records.append(json.loads(line))

        return stats, output_records

    def test_record_count_matches(self, tmp_path):
        records = [SAMPLE_RECORD, SAMPLE_RECORD_INSTRUCTION]
        stats, output = self._run_pipeline(tmp_path, records)
        assert len(output) == len(records)
        assert stats["total"] == len(records)

    def test_id_has_bt_suffix(self, tmp_path):
        stats, output = self._run_pipeline(tmp_path, [SAMPLE_RECORD])
        assert output[0]["id"] == "personas_IF_abc123_bt"

    def test_augmentation_method_present(self, tmp_path):
        stats, output = self._run_pipeline(tmp_path, [SAMPLE_RECORD])
        assert output[0]["augmentation_method"] == "back_translation"

    def test_augmentation_intermediate_lang_present(self, tmp_path):
        stats, output = self._run_pipeline(tmp_path, [SAMPLE_RECORD])
        assert output[0]["augmentation_intermediate_lang"] == "de"

    def test_non_prompt_fields_preserved(self, tmp_path):
        stats, output = self._run_pipeline(tmp_path, [SAMPLE_RECORD])
        rec = output[0]
        assert rec["response_of_winner_prompt"] == SAMPLE_RECORD["response_of_winner_prompt"]
        assert rec["added_constraint"] == SAMPLE_RECORD["added_constraint"]
        assert rec["added_constraint_num"] == SAMPLE_RECORD["added_constraint_num"]
        assert rec["rule_evaluate_dict"] == SAMPLE_RECORD["rule_evaluate_dict"]


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_prompt_with_unicode(self):
        record = {
            "winner_prompt": "Explain the caf\u00e9 culture in 50 words.",
            "added_constraint": {"Length": ["at most 50 words"]},
            "id": "unicode_test",
        }
        tokens = extract_constraint_tokens(record)
        assert "50" in tokens

    def test_prompt_with_embedded_quotes(self):
        record = {
            "winner_prompt": 'Include the phrase "hello world" and \'goodbye\' in 30 words.',
            "added_constraint": {"Keyword": ["include 'hello world'"]},
            "id": "quotes_test",
        }
        tokens = extract_constraint_tokens(record)
        assert "hello world" in tokens
        assert "goodbye" in tokens
        assert "30" in tokens


# ---------------------------------------------------------------------------
# TestBackTranslatePrompt (requires models)
# ---------------------------------------------------------------------------


@requires_transformers
@pytest.mark.slow
class TestBackTranslatePrompt:
    @pytest.fixture(autouse=True, scope="class")
    def _load_models(self, request):
        from src.crllm.dataset.augmentation.back_translate import (
            load_translation_models,
            back_translate_prompt,
        )
        models = load_translation_models("de")
        request.cls.fwd_model = models[0]
        request.cls.fwd_tok = models[1]
        request.cls.bwd_model = models[2]
        request.cls.bwd_tok = models[3]

    def test_output_is_english(self):
        from src.crllm.dataset.augmentation.back_translate import back_translate_prompt
        result = back_translate_prompt(
            "Write a short paragraph about machine learning.",
            self.fwd_model, self.fwd_tok, self.bwd_model, self.bwd_tok,
        )
        # Basic heuristic: result should contain common English words
        assert any(w in result.lower() for w in ["the", "a", "an", "and", "of", "to", "in", "is"])

    def test_output_differs_from_input(self):
        from src.crllm.dataset.augmentation.back_translate import back_translate_prompt
        prompt = "Please write a detailed explanation of neural networks in exactly 100 words."
        result = back_translate_prompt(
            prompt,
            self.fwd_model, self.fwd_tok, self.bwd_model, self.bwd_tok,
        )
        # Back-translation should produce a paraphrase (not identical)
        assert result != prompt

    def test_short_prompt(self):
        from src.crllm.dataset.augmentation.back_translate import back_translate_prompt
        result = back_translate_prompt(
            "Hello",
            self.fwd_model, self.fwd_tok, self.bwd_model, self.bwd_tok,
        )
        assert len(result) > 0

    def test_empty_prompt_returns_empty(self):
        from src.crllm.dataset.augmentation.back_translate import back_translate_prompt
        result = back_translate_prompt(
            "",
            self.fwd_model, self.fwd_tok, self.bwd_model, self.bwd_tok,
        )
        assert result == ""
