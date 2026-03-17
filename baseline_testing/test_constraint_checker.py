"""Tests for ConstraintChecker."""

import json
import pytest
from constraint_checker import ConstraintChecker

checker = ConstraintChecker()


# ── Word count ──────────────────────────────────────────────────────
class TestWordCount:
    def test_at_least_pass(self):
        resp = "one two three four five"
        c = {"type": "length_constraint:word_count", "target": 5, "relation": "at_least"}
        assert checker.check_constraint(resp, c) is True

    def test_at_least_fail(self):
        resp = "one two"
        c = {"type": "length_constraint:word_count", "target": 5, "relation": "at_least"}
        assert checker.check_constraint(resp, c) is False

    def test_exactly(self):
        resp = "one two three"
        c = {"type": "length_constraint:word_count", "target": 3, "relation": "exactly"}
        assert checker.check_constraint(resp, c) is True

    def test_at_most(self):
        resp = "one two"
        c = {"type": "length_constraint:word_count", "target": 3, "relation": "at_most"}
        assert checker.check_constraint(resp, c) is True


# ── Sentence count ──────────────────────────────────────────────────
class TestSentenceCount:
    def test_basic(self):
        resp = "Hello. World. How are you?"
        c = {"type": "length_constraint:sentence_count", "target": 3, "relation": "exactly"}
        assert checker.check_constraint(resp, c) is True

    def test_fail(self):
        resp = "Just one sentence"
        c = {"type": "length_constraint:sentence_count", "target": 2, "relation": "at_least"}
        assert checker.check_constraint(resp, c) is False


# ── Paragraph count ─────────────────────────────────────────────────
class TestParagraphCount:
    def test_two_paragraphs(self):
        resp = "Paragraph one.\n\nParagraph two."
        c = {"type": "length_constraint:paragraph_count", "target": 2, "relation": "exactly"}
        assert checker.check_constraint(resp, c) is True


# ── Keyword existence ───────────────────────────────────────────────
class TestKeywordExistence:
    def test_present(self):
        resp = "The cat sat on the mat"
        c = {"type": "keywords:existence", "keywords": ["cat", "mat"]}
        assert checker.check_constraint(resp, c) is True

    def test_missing(self):
        resp = "The dog sat on the rug"
        c = {"type": "keywords:existence", "keywords": ["cat"]}
        assert checker.check_constraint(resp, c) is False

    def test_case_insensitive(self):
        resp = "The CAT is here"
        c = {"type": "keywords:existence", "keywords": ["cat"]}
        assert checker.check_constraint(resp, c) is True


# ── Keyword frequency ──────────────────────────────────────────────
class TestKeywordFrequency:
    def test_at_least(self):
        resp = "hello hello hello world"
        c = {"type": "keywords:frequency", "keyword": "hello", "target": 3, "relation": "at_least"}
        assert checker.check_constraint(resp, c) is True

    def test_not_enough(self):
        resp = "hello world"
        c = {"type": "keywords:frequency", "keyword": "hello", "target": 3, "relation": "at_least"}
        assert checker.check_constraint(resp, c) is False


# ── Forbidden keywords ─────────────────────────────────────────────
class TestForbiddenKeywords:
    def test_clean(self):
        resp = "A nice response"
        c = {"type": "keywords:forbidden", "keywords": ["bad", "ugly"]}
        assert checker.check_constraint(resp, c) is True

    def test_contains_forbidden(self):
        resp = "This is bad"
        c = {"type": "keywords:forbidden", "keywords": ["bad"]}
        assert checker.check_constraint(resp, c) is False


# ── Start / end with ───────────────────────────────────────────────
class TestStartEndWith:
    def test_start_with_pass(self):
        resp = "Dear friend, hello"
        c = {"type": "start_with", "target": "Dear"}
        assert checker.check_constraint(resp, c) is True

    def test_start_with_fail(self):
        resp = "Hello, friend"
        c = {"type": "start_with", "target": "Dear"}
        assert checker.check_constraint(resp, c) is False

    def test_end_with_pass(self):
        resp = "Sincerely yours"
        c = {"type": "end_with", "target": "yours"}
        assert checker.check_constraint(resp, c) is True


# ── All caps count (BUG: ignores relation) ─────────────────────────
class TestAllCapsCount:
    def test_at_most_pass(self):
        """Current behavior: always uses <= (at_most)."""
        resp = "HELLO WORLD this is normal"
        c = {"type": "capitalization:all_caps_count", "target": 3, "relation": "at_most"}
        assert checker.check_constraint(resp, c) is True

    def test_at_least_pass(self):
        resp = "HELLO WORLD this is normal"
        c = {"type": "capitalization:all_caps_count", "target": 1, "relation": "at_least"}
        assert checker.check_constraint(resp, c) is True  # 2 >= 1

    def test_exactly_pass(self):
        resp = "HELLO WORLD this is normal"
        c = {"type": "capitalization:all_caps_count", "target": 2, "relation": "exactly"}
        assert checker.check_constraint(resp, c) is True  # 2 == 2

    def test_exactly_fail(self):
        resp = "HELLO WORLD this is normal"
        c = {"type": "capitalization:all_caps_count", "target": 3, "relation": "exactly"}
        assert checker.check_constraint(resp, c) is False  # 2 != 3


# ── All lowercase ──────────────────────────────────────────────────
class TestAllLowercase:
    def test_pass(self):
        resp = "all lowercase text"
        c = {"type": "capitalization:all_lowercase"}
        assert checker.check_constraint(resp, c) is True

    def test_fail(self):
        resp = "Not all lowercase"
        c = {"type": "capitalization:all_lowercase"}
        assert checker.check_constraint(resp, c) is False


# ── Bullet points ──────────────────────────────────────────────────
class TestBulletPoints:
    def test_pass(self):
        resp = "- item one\n- item two\n- item three"
        c = {"type": "format:bullet_points", "target": 3, "relation": "at_least"}
        assert checker.check_constraint(resp, c) is True

    def test_asterisk_bullets(self):
        resp = "* item one\n* item two"
        c = {"type": "format:bullet_points", "target": 2, "relation": "exactly"}
        assert checker.check_constraint(resp, c) is True

    def test_fail(self):
        resp = "- item one"
        c = {"type": "format:bullet_points", "target": 3, "relation": "at_least"}
        assert checker.check_constraint(resp, c) is False


# ── Numbered list ──────────────────────────────────────────────────
class TestNumberedList:
    def test_present(self):
        resp = "1. First\n2. Second"
        c = {"type": "format:numbered_list"}
        assert checker.check_constraint(resp, c) is True

    def test_parenthesis_format(self):
        resp = "1) First\n2) Second"
        c = {"type": "format:numbered_list"}
        assert checker.check_constraint(resp, c) is True

    def test_absent(self):
        resp = "No numbered items here"
        c = {"type": "format:numbered_list"}
        assert checker.check_constraint(resp, c) is False


# ── Sections ────────────────────────────────────────────────────────
class TestSections:
    def test_pass(self):
        resp = "## Intro\nText\n## Body\nMore text\n## Conclusion\nEnd"
        c = {"type": "format:sections", "target": 3, "relation": "at_least"}
        assert checker.check_constraint(resp, c) is True

    def test_fail(self):
        resp = "## Only one section"
        c = {"type": "format:sections", "target": 3, "relation": "at_least"}
        assert checker.check_constraint(resp, c) is False


# ── JSON format ─────────────────────────────────────────────────────
class TestJsonFormat:
    def test_valid(self):
        resp = json.dumps({"key": "value"})
        c = {"type": "format:json"}
        assert checker.check_constraint(resp, c) is True

    def test_invalid(self):
        resp = "not json {{"
        c = {"type": "format:json"}
        assert checker.check_constraint(resp, c) is False


# ── English ─────────────────────────────────────────────────────────
class TestEnglish:
    def test_english(self):
        resp = "This is an English sentence with normal words"
        c = {"type": "language:english"}
        assert checker.check_constraint(resp, c) is True


# ── No comma ────────────────────────────────────────────────────────
class TestNoComma:
    def test_pass(self):
        resp = "No commas here"
        c = {"type": "punctuation:no_comma"}
        assert checker.check_constraint(resp, c) is True

    def test_fail(self):
        resp = "Has a comma, right here"
        c = {"type": "punctuation:no_comma"}
        assert checker.check_constraint(resp, c) is False


# ── Postscript ──────────────────────────────────────────────────────
class TestPostscript:
    def test_ps_dot(self):
        resp = "Main text.\nP.S. Don't forget!"
        c = {"type": "detectable_content:postscript"}
        assert checker.check_constraint(resp, c) is True

    def test_ps_colon(self):
        resp = "Main text.\nPS: Remember this"
        c = {"type": "detectable_content:postscript"}
        assert checker.check_constraint(resp, c) is True

    def test_absent(self):
        resp = "No postscript here"
        c = {"type": "detectable_content:postscript"}
        assert checker.check_constraint(resp, c) is False


# ── Highlight ───────────────────────────────────────────────────────
class TestHighlight:
    def test_bold(self):
        resp = "This is **bold** text"
        c = {"type": "detectable_format:highlight"}
        assert checker.check_constraint(resp, c) is True

    def test_italic(self):
        resp = "This is *italic* text"
        c = {"type": "detectable_format:highlight"}
        assert checker.check_constraint(resp, c) is True

    def test_none(self):
        resp = "Plain text only"
        c = {"type": "detectable_format:highlight"}
        assert checker.check_constraint(resp, c) is False


# ── Unknown type ────────────────────────────────────────────────────
class TestUnknownType:
    def test_returns_none(self):
        resp = "Any text"
        c = {"type": "some:unknown_type"}
        assert checker.check_constraint(resp, c) is None


# ── check_all integration ──────────────────────────────────────────
class TestCheckAll:
    def test_all_pass(self):
        resp = "hello hello hello"
        constraints = [
            {"type": "length_constraint:word_count", "target": 3, "relation": "exactly"},
            {"type": "keywords:existence", "keywords": ["hello"]},
        ]
        result = checker.check_all(resp, constraints)
        assert result["num_constraints"] == 2
        assert result["num_checked"] == 2
        assert result["num_passed"] == 2
        assert result["per_constraint_csr"] == 1.0
        assert result["hard_csr"] is True

    def test_partial_pass(self):
        resp = "hello world"
        constraints = [
            {"type": "keywords:existence", "keywords": ["hello"]},
            {"type": "keywords:existence", "keywords": ["missing"]},
        ]
        result = checker.check_all(resp, constraints)
        assert result["num_passed"] == 1
        assert result["num_checked"] == 2
        assert result["per_constraint_csr"] == 0.5
        assert result["hard_csr"] is False

    def test_with_unknown_type(self):
        resp = "hello"
        constraints = [
            {"type": "keywords:existence", "keywords": ["hello"]},
            {"type": "unknown:type"},
        ]
        result = checker.check_all(resp, constraints)
        assert result["num_constraints"] == 2
        assert result["num_checked"] == 1  # unknown skipped
        assert result["num_passed"] == 1
        assert result["per_constraint_csr"] == 1.0

    def test_empty_constraints(self):
        result = checker.check_all("any text", [])
        assert result["num_constraints"] == 0
        assert result["per_constraint_csr"] == 0.0
        assert result["hard_csr"] is False
