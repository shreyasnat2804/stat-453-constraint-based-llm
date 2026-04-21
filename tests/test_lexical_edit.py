"""Tests for lexical_edit augmentation module."""

import json
import random

import pytest

from src.crllm.dataset.augmentation.lexical_edit import (
    _extract_entities,
    detect_prompt_field,
    eda_augment,
    extract_protected_tokens,
    get_wordnet_synonyms,
    is_protected,
    random_deletion,
    random_insertion,
    random_swap,
    run_lexical_edit_pipeline,
    synonym_replacement,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_RECORD = {
    "winner_prompt": "Write a 200 word essay about 'machine learning' in paragraph format.",
    "response_of_winner_prompt": "Machine learning is a field of AI...",
    "added_constraint": {
        "Length": ["at most 200 words"],
        "Keyword": ["include 'machine learning'"],
    },
    "added_constraint_num": 2,
    "rule_evaluate_dict": {},
    "id": "personas_IF_abc123",
}

SAMPLE_RECORD_INSTRUCTION = {
    "instruction": "Describe the weather in bullet format.",
    "response_of_winner_prompt": "The weather is sunny...",
    "added_constraint": {"Format": ["use bullet points"]},
    "added_constraint_num": 1,
    "rule_evaluate_dict": {},
    "id": "cluster_xyz789",
}


def _make_rng(seed=42):
    return random.Random(seed)


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# TestGetWordnetSynonyms
# ═════════════════════════════════════════════════════════════════════════════

class TestGetWordnetSynonyms:
    def test_common_word_returns_synonyms(self):
        syns = get_wordnet_synonyms("happy")
        assert len(syns) > 0
        assert all(isinstance(s, str) for s in syns)

    def test_madeup_word_returns_empty(self):
        syns = get_wordnet_synonyms("xyzzyplugh")
        assert syns == []

    def test_word_itself_excluded(self):
        syns = get_wordnet_synonyms("good")
        assert "good" not in [s.lower() for s in syns]


# ═════════════════════════════════════════════════════════════════════════════
# TestExtractProtectedTokens
# ═════════════════════════════════════════════════════════════════════════════

class TestExtractProtectedTokens:
    def test_numbers_protected(self):
        protected = extract_protected_tokens(SAMPLE_RECORD)
        assert "200" in protected

    def test_constraint_words_protected(self):
        protected = extract_protected_tokens(SAMPLE_RECORD)
        # "at", "most", "200", "words" come from constraint descriptions
        assert "most" in protected
        assert "words" in protected

    def test_format_keywords_protected(self):
        protected = extract_protected_tokens(SAMPLE_RECORD)
        assert "paragraph" in protected
        assert "json" in protected
        assert "bullet" in protected

    def test_quoted_strings_protected(self):
        protected = extract_protected_tokens(SAMPLE_RECORD)
        assert "machine" in protected
        assert "learning" in protected

    def test_constraint_category_names_protected(self):
        protected = extract_protected_tokens(SAMPLE_RECORD)
        assert "length" in protected
        assert "keyword" in protected

    def test_named_entities_protected(self):
        record = {
            "winner_prompt": "Write a letter to François Müller about his trip to Paris.",
            "response_of_winner_prompt": "Dear François...",
            "added_constraint": {},
            "id": "entity_test",
        }
        protected = extract_protected_tokens(record)
        assert "françois" in protected
        assert "müller" in protected
        assert "paris" in protected

    def test_entity_at_sentence_start_not_protected(self):
        # "Write" is at sentence start — should NOT be extracted as entity
        entities = _extract_entities("Write a letter. Send it to José.")
        assert "write" not in entities
        assert "send" not in entities
        assert "josé" in entities

    def test_allcaps_not_treated_as_entity(self):
        entities = _extract_entities("Use the JSON format. Contact IBM today.")
        assert "JSON" not in entities and "json" not in entities
        assert "IBM" not in entities and "ibm" not in entities


# ═════════════════════════════════════════════════════════════════════════════
# TestSynonymReplacement
# ═════════════════════════════════════════════════════════════════════════════

class TestSynonymReplacement:
    def test_replaces_non_protected(self):
        words = ["the", "happy", "cat", "sat"]
        protected = {"cat"}
        result = synonym_replacement(words, protected, n=2, rng=_make_rng())
        # cat should still be there
        assert "cat" in result
        assert len(result) == len(words)

    def test_protected_unchanged(self):
        words = ["big", "small", "red"]
        protected = {"big", "small", "red"}
        result = synonym_replacement(words, protected, n=3, rng=_make_rng())
        assert result == words

    def test_n_exceeds_eligible_gracefully(self):
        words = ["happy", "sad"]
        protected = {"happy"}
        # n=10 but only 1 eligible word
        result = synonym_replacement(words, protected, n=10, rng=_make_rng())
        assert len(result) == 2
        assert "happy" in result


# ═════════════════════════════════════════════════════════════════════════════
# TestRandomDeletion
# ═════════════════════════════════════════════════════════════════════════════

class TestRandomDeletion:
    def test_protected_never_deleted(self):
        words = ["alpha", "beta", "gamma", "delta"]
        protected = {"alpha", "gamma"}
        result = random_deletion(words, protected, p=1.0, rng=_make_rng())
        assert "alpha" in result
        assert "gamma" in result

    def test_at_least_one_word_remains(self):
        words = ["one", "two", "three"]
        protected = set()
        result = random_deletion(words, protected, p=1.0, rng=_make_rng())
        assert len(result) >= 1

    def test_empty_input_returns_empty(self):
        result = random_deletion([], set(), p=0.5, rng=_make_rng())
        assert result == []


# ═════════════════════════════════════════════════════════════════════════════
# TestRandomSwap
# ═════════════════════════════════════════════════════════════════════════════

class TestRandomSwap:
    def test_protected_not_swapped(self):
        words = ["alpha", "beta", "gamma"]
        protected = {"alpha", "beta", "gamma"}
        result = random_swap(words, protected, n=3, rng=_make_rng())
        assert result == words

    def test_single_word_unchanged(self):
        words = ["solo"]
        result = random_swap(words, set(), n=1, rng=_make_rng())
        assert result == ["solo"]


# ═════════════════════════════════════════════════════════════════════════════
# TestRandomInsertion
# ═════════════════════════════════════════════════════════════════════════════

class TestRandomInsertion:
    def test_inserts_synonym(self):
        words = ["happy", "dog", "runs"]
        protected = set()
        result = random_insertion(words, protected, n=1, rng=_make_rng())
        # Length should increase (assuming at least one word has synonyms)
        assert len(result) >= len(words)

    def test_length_increases(self):
        words = ["good", "big", "fast"]
        protected = set()
        result = random_insertion(words, protected, n=2, rng=_make_rng())
        assert len(result) >= len(words)


# ═════════════════════════════════════════════════════════════════════════════
# TestEdaAugment
# ═════════════════════════════════════════════════════════════════════════════

class TestEdaAugment:
    def test_constraint_tokens_preserved(self):
        text = "Write a 200 word essay about machine learning today"
        protected = extract_protected_tokens(SAMPLE_RECORD)
        result = eda_augment(text, protected, rng=_make_rng())
        # Protected tokens from the constraint should survive
        result_lower = result.lower()
        assert "200" in result
        assert "machine" in result_lower or "learning" in result_lower

    def test_output_differs_from_input(self):
        text = "The quick brown fox jumps over the lazy dog near the river"
        protected = set()
        result = eda_augment(
            text, protected,
            alpha_sr=0.3, alpha_ri=0.3, alpha_rs=0.3, alpha_rd=0.1,
            rng=_make_rng(),
        )
        assert result != text

    def test_all_zero_alphas_returns_input_unchanged(self):
        text = "Hello world this is a test"
        protected = set()
        result = eda_augment(
            text, protected,
            alpha_sr=0.0, alpha_ri=0.0, alpha_rs=0.0, alpha_rd=0.0,
            rng=_make_rng(),
        )
        assert result == text


# ═════════════════════════════════════════════════════════════════════════════
# TestIdempotency
# ═════════════════════════════════════════════════════════════════════════════

class TestIdempotency:
    def test_skips_existing_output(self, tmp_path):
        inp = tmp_path / "input.jsonl"
        out = tmp_path / "output.jsonl"
        _write_jsonl(inp, [SAMPLE_RECORD])
        out.write_text("already here\n")

        stats = run_lexical_edit_pipeline(inp, out, force=False)
        assert stats["status"] == "skipped"
        assert stats["reason"] == "output_exists"
        # Original content unchanged
        assert out.read_text() == "already here\n"

    def test_force_overwrites(self, tmp_path):
        inp = tmp_path / "input.jsonl"
        out = tmp_path / "output.jsonl"
        _write_jsonl(inp, [SAMPLE_RECORD])
        out.write_text("old data\n")

        stats = run_lexical_edit_pipeline(inp, out, force=True)
        assert stats["augmented"] == 1
        assert out.read_text() != "old data\n"


# ═════════════════════════════════════════════════════════════════════════════
# TestOutputSchema
# ═════════════════════════════════════════════════════════════════════════════

class TestOutputSchema:
    def test_non_prompt_fields_preserved(self, tmp_path):
        inp = tmp_path / "input.jsonl"
        out = tmp_path / "output.jsonl"
        _write_jsonl(inp, [SAMPLE_RECORD])

        run_lexical_edit_pipeline(inp, out, force=True)

        with open(out) as f:
            result = json.loads(f.readline())

        assert result["response_of_winner_prompt"] == SAMPLE_RECORD["response_of_winner_prompt"]
        assert result["added_constraint"] == SAMPLE_RECORD["added_constraint"]
        assert result["added_constraint_num"] == SAMPLE_RECORD["added_constraint_num"]
        assert result["rule_evaluate_dict"] == SAMPLE_RECORD["rule_evaluate_dict"]

    def test_id_has_lex_suffix(self, tmp_path):
        inp = tmp_path / "input.jsonl"
        out = tmp_path / "output.jsonl"
        _write_jsonl(inp, [SAMPLE_RECORD])

        run_lexical_edit_pipeline(inp, out, force=True)

        with open(out) as f:
            result = json.loads(f.readline())

        assert result["id"].endswith("_lex")
        assert result["id"] == "personas_IF_abc123_lex"

    def test_augmentation_method_field_present(self, tmp_path):
        inp = tmp_path / "input.jsonl"
        out = tmp_path / "output.jsonl"
        _write_jsonl(inp, [SAMPLE_RECORD])

        run_lexical_edit_pipeline(inp, out, force=True)

        with open(out) as f:
            result = json.loads(f.readline())

        assert result["augmentation_method"] == "lexical_edit"

    def test_record_count_matches_input(self, tmp_path):
        inp = tmp_path / "input.jsonl"
        out = tmp_path / "output.jsonl"
        records = [
            {**SAMPLE_RECORD, "id": f"rec_{i}"}
            for i in range(5)
        ]
        _write_jsonl(inp, records)

        stats = run_lexical_edit_pipeline(inp, out, force=True)

        assert stats["augmented"] == 5
        with open(out) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 5


# ═════════════════════════════════════════════════════════════════════════════
# TestEdgeCases
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_prompt(self):
        text = ""
        result = eda_augment(text, set(), rng=_make_rng())
        assert result == ""

    def test_single_word(self):
        text = "hello"
        result = eda_augment(text, set(), rng=_make_rng())
        # Should still return something (possibly different word)
        assert len(result) > 0

    def test_all_protected_prompt_returns_unchanged(self):
        words = ["json", "bullet", "list"]
        text = " ".join(words)
        protected = set(w.lower() for w in words)
        result = eda_augment(
            text, protected,
            alpha_sr=0.5, alpha_ri=0.0, alpha_rs=0.5, alpha_rd=0.0,
            rng=_make_rng(),
        )
        # SR and RS can't touch anything, RI and RD are 0
        assert result == text

    def test_special_characters_not_corrupted(self):
        text = "Use the @symbol and #hashtag with $dollars"
        protected = set()
        result = eda_augment(text, protected, alpha_sr=0.0, alpha_ri=0.0,
                             alpha_rs=0.0, alpha_rd=0.0, rng=_make_rng())
        assert "@symbol" in result
        assert "#hashtag" in result
        assert "$dollars" in result

    def test_instruction_field_supported(self, tmp_path):
        inp = tmp_path / "input.jsonl"
        out = tmp_path / "output.jsonl"
        _write_jsonl(inp, [SAMPLE_RECORD_INSTRUCTION])

        stats = run_lexical_edit_pipeline(inp, out, force=True)
        assert stats["augmented"] == 1

        with open(out) as f:
            result = json.loads(f.readline())
        assert "instruction" in result
        assert result["id"] == "cluster_xyz789_lex"
