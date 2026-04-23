"""Unit tests for preprocess.py

Run from the repo root:
    pytest tests/test_preprocess.py -v
"""

import json
from pathlib import Path

import pytest

from crllm.dataset.preprocess.preprocess import (
    DedupIndex,
    FIELD_CONSTRAINT_N,
    FIELD_CONSTRAINTS,
    FIELD_PROMPT,
    FIELD_RESPONSE,
    HAS_DATASKETCH,
    HAS_LANGDETECT,
    _compute_minhash_for_text,
    _process_one_line,
    audit_distribution,
    clean_record,
    clean_text,
    detect_language,
    extract_categories,
    fingerprint,
    is_mostly_printable,
    normalize_symbols_to_ascii,
    normalize_whitespace,
    remove_html,
    remove_stopwords,
    replace_emojis,
    replace_unknown_chars,
    run_pipeline,
    shingles,
    validate_constraints,
)

from collections import Counter


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


DEFAULT_PROMPT = (
    "Please write a short and evocative poem about the moon shining "
    "gently over the ocean tonight with rippling waves reflecting the "
    "silver moonlight under a clear sky full of distant stars."
)  # 29 words — safely above the 15-word min_length gate


def _make_record(
    prompt: str = DEFAULT_PROMPT,
    response: str = "The silver moon dances on the waves, casting light across the deep.",
    constraints: dict | None = None,
    constraint_num: int | None = None,
    rec_id: str = "rec_0",
    **extra,
) -> dict:
    """Build a minimal RECAST-shaped record with sensible defaults."""
    if constraints is None:
        constraints = {
            "Length": ["Limit the response to 10-20 words."],
            "Style":  ["Use a poetic tone."],
        }
    rec = {
        FIELD_PROMPT:       prompt,
        FIELD_RESPONSE:     response,
        FIELD_CONSTRAINTS:  constraints,
        "id":               rec_id,
    }
    if constraint_num is not None:
        rec[FIELD_CONSTRAINT_N] = constraint_num
    else:
        rec[FIELD_CONSTRAINT_N] = sum(len(v) for v in constraints.values())
    rec.update(extra)
    return rec


@pytest.fixture
def valid_record():
    return _make_record()


@pytest.fixture
def sample_jsonl(tmp_path):
    """Write a JSONL fixture with a mix of keep/skip cases and return its path."""
    # All prompts below are phrased so langdetect reliably returns "en"
    # (it misclassifies short / technical-vocabulary strings — see the
    # `detect()` notes in preprocess.py).
    records = [
        # 0: valid, should be kept
        _make_record(rec_id="keep_0"),
        # 1: valid, different text, should be kept
        _make_record(
            prompt=(
                "The scientist carefully explained quantum entanglement to her "
                "excited group of curious undergraduate students this morning."
            ),
            response="When two particles share a state, measuring one instantly fixes the other.",
            constraints={"Style": ["Use plain, non-technical language."]},
            rec_id="keep_1",
        ),
        # 2: too short prompt (clearly English, but under min_length)
        _make_record(
            prompt="The quick brown fox jumped over.",
            rec_id="drop_short",
        ),
        # 3: empty response
        _make_record(response="", rec_id="drop_empty"),
        # 4: missing added_constraint
        _make_record(constraints={}, rec_id="drop_noconstraint"),
        # 5: near-duplicate of record 0 — same prompt with trivial edits so
        # it survives the length gate and exercises the fuzzy dedup path.
        _make_record(
            prompt=DEFAULT_PROMPT.replace("Please", "Kindly").replace(".", "!"),
            rec_id="drop_dup",
        ),
    ]
    path = tmp_path / "fixture.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        # intentionally add a blank line and a malformed line
        f.write("\n")
        f.write("{not valid json\n")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — text cleaning helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestRemoveHtml:
    def test_strips_tags(self):
        assert remove_html("<b>hello</b>") == " hello "

    def test_strips_entities(self):
        out = remove_html("a &amp; b &#39;c&#39;")
        assert "&amp;" not in out and "&#39;" not in out
        # Surrounding letters survive
        assert "a" in out and "b" in out and "c" in out

    def test_plain_text_unchanged(self):
        assert remove_html("no html here") == "no html here"

    def test_multiline_tags(self):
        # Tag spans a newline — DOTALL flag should handle it
        assert remove_html("<a\nhref='x'>link</a>") == " link "


class TestReplaceEmojis:
    def test_replaces_common_emoji(self):
        out = replace_emojis("hi 😀 there")
        assert "😀" not in out
        assert "hi" in out and "there" in out

    def test_no_emoji_passthrough(self):
        assert replace_emojis("just text") == "just text"


class TestReplaceUnknownChars:
    def test_strips_control_chars(self):
        assert "\x00" not in replace_unknown_chars("a\x00b")
        assert "\x01" not in replace_unknown_chars("a\x01b")

    def test_strips_zero_width(self):
        # U+200B zero-width space
        assert replace_unknown_chars("a\u200bb") == "ab"

    def test_collapses_symbol_runs(self):
        # Four or more non-word non-space chars collapse to a space
        out = replace_unknown_chars("good????bye")
        assert "????" not in out

    def test_preserves_normal_text(self):
        assert replace_unknown_chars("Hello, world.") == "Hello, world."


class TestRemoveStopwords:
    def test_strips_basic_stopwords(self):
        assert remove_stopwords("the cat is on the mat") == "cat mat"

    def test_case_insensitive(self):
        # "The" gets stripped even if capitalized
        assert remove_stopwords("The Cat") == "Cat"

    def test_handles_punctuation_attached(self):
        # "the," should match stopword "the" after stripping punctuation
        assert remove_stopwords("the, cat") == "cat"

    def test_preserves_non_stopwords(self):
        result = remove_stopwords("quantum mechanics matters")
        assert "quantum" in result
        assert "mechanics" in result


class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("a   b") == "a b"

    def test_collapses_tabs_newlines(self):
        assert normalize_whitespace("a\t\nb") == "a b"

    def test_strips_leading_trailing(self):
        assert normalize_whitespace("  x  ") == "x"


class TestIsMostlyPrintable:
    def test_plain_ascii(self):
        assert is_mostly_printable("Hello, world!") is True

    def test_empty_string(self):
        assert is_mostly_printable("") is False

    def test_mostly_garbage(self):
        # 20 control chars + 1 letter = ~5% printable
        assert is_mostly_printable("\x00" * 20 + "a") is False

    def test_threshold_boundary(self):
        # 9 printable + 1 non-printable = 90% printable, threshold 0.85 → True
        assert is_mostly_printable("abcdefghi\x00", threshold=0.85) is True


class TestNormalizeSymbolsToAscii:
    def test_plain_ascii_unchanged(self):
        assert normalize_symbols_to_ascii("Hello, world!") == "Hello, world!"

    def test_smart_quotes_mapped(self):
        # U+201C / U+201D → "    U+2018 / U+2019 → '
        out = normalize_symbols_to_ascii("\u201cHello\u201d he \u2018said\u2019.")
        assert out == '"Hello" he \'said\'.'

    def test_em_and_en_dash_mapped(self):
        out = normalize_symbols_to_ascii("range 10\u201320 \u2014 inclusive")
        assert "\u2013" not in out and "\u2014" not in out
        assert "10-20" in out and " - " in out

    def test_ellipsis_expanded(self):
        assert normalize_symbols_to_ascii("wait\u2026really?") == "wait...really?"

    def test_non_breaking_space_to_regular(self):
        # NBSP (U+00A0) → ASCII space; should NOT be stripped entirely
        out = normalize_symbols_to_ascii("a\u00a0b")
        assert out == "a b"

    def test_french_letters_preserved(self):
        # Accented Latin letters are kept — only symbols are targeted
        out = normalize_symbols_to_ascii("café naïve résumé façade")
        assert out == "café naïve résumé façade"

    def test_cyrillic_letters_preserved(self):
        # Russian letters must pass through untouched
        out = normalize_symbols_to_ascii("Привет мир")
        assert out == "Привет мир"

    def test_cjk_letters_preserved(self):
        out = normalize_symbols_to_ascii("你好世界")
        assert out == "你好世界"

    def test_arabic_letters_preserved(self):
        out = normalize_symbols_to_ascii("مرحبا بالعالم")
        assert "مرحبا" in out and "بالعالم" in out

    def test_non_ascii_symbols_stripped(self):
        # Arrows, math operators, currency with no ASCII map → dropped
        out = normalize_symbols_to_ascii("a → b ≠ c € d")
        assert "→" not in out and "≠" not in out and "€" not in out
        for letter in "abcd":
            assert letter in out

    def test_mixed_russian_with_symbols(self):
        # Russian letters preserved, smart quotes normalised, arrow dropped
        out = normalize_symbols_to_ascii("\u201cПривет\u201d → мир")
        assert '"Привет"' in out
        assert "→" not in out
        assert "мир" in out

    def test_digits_and_marks_preserved(self):
        # Arabic-Indic digit and combining mark should not be stripped
        out = normalize_symbols_to_ascii("year ٢٠٢٤ café")
        assert "٢٠٢٤" in out
        assert "café" in out


class TestCleanText:
    def test_integration(self):
        raw = "<b>Hello</b>   😀 \x00 world!"
        out = clean_text(raw)
        assert "<b>" not in out
        assert "\x00" not in out
        assert "😀" not in out
        # Whitespace collapsed
        assert "  " not in out
        assert "Hello" in out and "world" in out

    def test_integration_preserves_foreign_letters(self):
        # French + Russian + Chinese all survive the full clean_text chain,
        # but smart quotes and the arrow symbol around them do not.
        raw = "<i>\u201ccafé\u201d</i> Привет → 你好"
        out = clean_text(raw)
        assert "café" in out
        assert "Привет" in out
        assert "你好" in out
        assert "<i>" not in out
        assert "\u201c" not in out and "\u201d" not in out
        assert "→" not in out


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — constraint validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateConstraints:
    def test_valid(self, valid_record):
        ok, reason = validate_constraints(valid_record)
        assert ok is True
        assert reason == ""

    def test_missing_field(self):
        ok, reason = validate_constraints({})
        assert ok is False
        assert "missing" in reason or "empty" in reason

    def test_empty_dict(self):
        rec = _make_record(constraints={})
        ok, reason = validate_constraints(rec)
        assert ok is False

    def test_not_a_dict(self):
        # Build manually so _make_record's .values() sum doesn't explode
        rec = {
            FIELD_PROMPT:       "any prompt",
            FIELD_RESPONSE:     "any response",
            FIELD_CONSTRAINTS:  ["Length: 10 words"],
            FIELD_CONSTRAINT_N: 1,
        }
        ok, reason = validate_constraints(rec)
        assert ok is False
        assert "dict" in reason

    def test_empty_category_list(self):
        rec = _make_record(constraints={"Length": []})
        ok, reason = validate_constraints(rec)
        assert ok is False
        assert "empty" in reason or "non-list" in reason

    def test_blank_item(self):
        rec = _make_record(constraints={"Length": ["  "]})
        ok, reason = validate_constraints(rec)
        assert ok is False
        assert "blank" in reason

    def test_non_string_item(self):
        rec = _make_record(constraints={"Length": [42]})
        ok, reason = validate_constraints(rec)
        assert ok is False

    def test_count_mismatch_rejected(self):
        """Documents the current strict behaviour — the stated count must match.

        This is the check the team has flagged as over-strict; see
        preprocess.py `validate_constraints` for context.
        """
        rec = _make_record(
            constraints={"Length": ["a", "b"], "Style": ["c"]},
            constraint_num=5,  # stated 5 but actual 3
        )
        ok, reason = validate_constraints(rec)
        assert ok is False
        assert "does not match" in reason

    def test_count_match_ok(self):
        rec = _make_record(
            constraints={"Length": ["a", "b"], "Style": ["c"]},
            constraint_num=3,
        )
        ok, _ = validate_constraints(rec)
        assert ok is True

    def test_missing_count_ok(self):
        """When added_constraint_num is absent, no cross-check fires."""
        rec = _make_record(constraints={"Length": ["a"]}, constraint_num=None)
        rec.pop(FIELD_CONSTRAINT_N, None)
        ok, _ = validate_constraints(rec)
        assert ok is True


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — deduplication
# ═══════════════════════════════════════════════════════════════════════════════


class TestFingerprint:
    def test_deterministic(self):
        assert fingerprint("Hello") == fingerprint("Hello")

    def test_case_and_whitespace_insensitive(self):
        assert fingerprint("Hello World") == fingerprint("hello   world")

    def test_different_text(self):
        assert fingerprint("Hello") != fingerprint("World")


class TestShingles:
    def test_basic(self):
        s = shingles("abcdef", k=3)
        assert "abc" in s
        assert "def" in s
        assert len(s) == 4  # abc, bcd, cde, def

    def test_short_text(self):
        # Text shorter than k collapses into a single shingle
        s = shingles("ab", k=5)
        assert len(s) == 1


class TestDedupIndex:
    def test_exact_duplicate_detected(self):
        idx = DedupIndex(threshold=0.9)
        assert idx.is_duplicate("k1", "the quick brown fox") is False
        assert idx.is_duplicate("k2", "the quick brown fox") is True

    def test_case_normalized(self):
        idx = DedupIndex(threshold=0.9)
        assert idx.is_duplicate("k1", "The Quick Brown Fox") is False
        assert idx.is_duplicate("k2", "the quick brown fox") is True

    def test_unrelated_text_not_duplicate(self):
        idx = DedupIndex(threshold=0.9)
        assert idx.is_duplicate("k1", "the quick brown fox") is False
        assert idx.is_duplicate("k2", "completely different content here") is False

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_fuzzy_near_duplicate(self):
        """Tiny edits should still count as duplicates at a permissive threshold."""
        idx = DedupIndex(threshold=0.5)
        a = "Write a short poem about the moon over the ocean tonight."
        b = "Write a short poem about the moon over the ocean tonight!"
        assert idx.is_duplicate("k1", a) is False
        assert idx.is_duplicate("k2", b) is True

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_precomputed_minhash_accepted(self):
        idx = DedupIndex(threshold=0.9)
        text = "the quick brown fox jumps over the lazy dog"
        mh = _compute_minhash_for_text(text)
        assert idx.is_duplicate("k1", text, minhash=mh) is False
        # Second insert of the same text with a precomputed hash → duplicate
        mh2 = _compute_minhash_for_text(text)
        assert idx.is_duplicate("k2", text, minhash=mh2) is True


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — category extraction + distribution audit
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractCategories:
    def test_dict_categories(self, valid_record):
        cats = extract_categories(valid_record)
        assert set(cats) == {"Length", "Style"}

    def test_missing_returns_unknown(self):
        assert extract_categories({}) == ["unknown"]

    def test_non_dict_returns_unknown(self):
        assert extract_categories({FIELD_CONSTRAINTS: "not a dict"}) == ["unknown"]


class TestAuditDistribution:
    def test_balanced_no_warnings(self):
        counts = Counter({"A": 100, "B": 100, "C": 100})
        assert audit_distribution(counts, imbalance_threshold=0.5) == []

    def test_skewed_emits_warning(self):
        # "C" at 2% is well below 0.5 × (1/3) ≈ 16.6%
        counts = Counter({"A": 100, "B": 100, "C": 4})
        warnings = audit_distribution(counts, imbalance_threshold=0.5)
        assert any("C" in w for w in warnings)

    def test_empty_counter(self):
        assert audit_distribution(Counter()) == []


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — clean_record (per-record pipeline)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCleanRecord:
    def test_kept(self, valid_record):
        cleaned, reason = clean_record(valid_record)
        assert cleaned is not None
        assert reason == ""

    def test_stopwords_removed_from_prompt(self, valid_record):
        cleaned, _ = clean_record(valid_record)
        tokens = cleaned[FIELD_PROMPT].lower().split()
        # Stopwords present in both the NLTK set and the built-in fallback.
        # The default prompt contains all of these.
        for stopword in ("the", "a", "about", "with", "of"):
            assert stopword not in tokens, (
                f"stopword {stopword!r} still present in {tokens!r}"
            )

    def test_response_not_stopword_stripped(self, valid_record):
        cleaned, _ = clean_record(valid_record)
        # Response keeps its stopwords — must stay fluent
        assert "the" in cleaned[FIELD_RESPONSE].lower()

    def test_too_short_prompt(self):
        rec = _make_record(prompt="only five words here honestly")
        cleaned, reason = clean_record(rec, min_length=15)
        assert cleaned is None
        assert reason == "too_short"

    def test_empty_response(self):
        rec = _make_record(response="")
        cleaned, reason = clean_record(rec)
        assert cleaned is None
        assert reason == "empty_response"

    def test_bad_constraint(self):
        rec = _make_record(constraints={})
        cleaned, reason = clean_record(rec)
        assert cleaned is None
        assert reason.startswith("bad_constraint")

    @pytest.mark.skipif(not HAS_LANGDETECT, reason="langdetect not installed")
    def test_non_english_dropped(self):
        # Clearly French / non-English text, longer than min_length
        rec = _make_record(
            prompt=(
                "Je voudrais que vous rediger une petite histoire sur un "
                "chat qui voyage autour du monde avec son meilleur ami."
            )
        )
        cleaned, reason = clean_record(rec)
        # langdetect should flag this as non-English
        assert cleaned is None
        assert reason == "non_english"


# ═══════════════════════════════════════════════════════════════════════════════
# Parallel workers
# ═══════════════════════════════════════════════════════════════════════════════


class TestProcessOneLine:
    def test_kept(self, valid_record):
        line = json.dumps(valid_record)
        line_no, rec, reason = _process_one_line(1, line, min_length=15)
        assert line_no == 1
        assert rec is not None
        assert reason == ""

    def test_blank_line(self):
        line_no, rec, reason = _process_one_line(5, "   \n", min_length=15)
        assert rec is None
        assert reason == "empty_line"

    def test_bad_json(self):
        line_no, rec, reason = _process_one_line(3, "{not json", min_length=15)
        assert rec is None
        assert reason == "json_error"

    def test_propagates_skip_reason(self):
        # Long enough for langdetect to reliably say English, but still under
        # the 15-word min_length gate — isolates the too_short reason.
        rec = _make_record(
            prompt="The quick brown fox jumped over the lazy dog today.",
            rec_id="x",
        )
        line_no, out, reason = _process_one_line(1, json.dumps(rec), min_length=15)
        assert out is None
        assert reason == "too_short"


@pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
class TestComputeMinhash:
    def test_returns_minhash(self):
        mh = _compute_minhash_for_text("hello world this is a test")
        assert mh is not None

    def test_same_text_similar_hash(self):
        mh1 = _compute_minhash_for_text("hello world")
        mh2 = _compute_minhash_for_text("hello world")
        # MinHash.jaccard() returns 1.0 for identical shingle sets
        assert mh1.jaccard(mh2) == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunPipeline:
    def test_end_to_end_counts(self, sample_jsonl, tmp_path):
        output_path = tmp_path / "out.jsonl"
        stats = run_pipeline(
            input_path          = sample_jsonl,
            output_path         = output_path,
            min_length          = 15,
            dedup_threshold     = 0.5,
            imbalance_threshold = 0.5,
            n_jobs              = 1,   # serial for determinism in the test
        )

        # Structural assertions on the returned stats
        assert stats["total"] == 7  # 6 records + 1 malformed line; blank line skipped
        assert stats["skipped_other"] == 1             # malformed JSON
        assert stats["skipped_too_short"] >= 1
        assert stats["skipped_empty_response"] >= 1
        assert stats["skipped_bad_constraint"] >= 1
        # Dedup only fires when datasketch is installed AND threshold catches it;
        # with exact dedup only, the near-dup variant with "!" differs from "."
        if HAS_DATASKETCH:
            assert stats["skipped_duplicate"] >= 1

        # Something survived
        assert stats["kept"] >= 1

        # Output file has the right number of lines
        with open(output_path) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == stats["kept"]

        # Every kept record is valid JSON and still has the key fields
        for line in lines:
            rec = json.loads(line)
            assert rec[FIELD_PROMPT]
            assert rec[FIELD_RESPONSE]
            assert isinstance(rec[FIELD_CONSTRAINTS], dict)

    def test_dedup_across_pipeline(self, tmp_path):
        """Two identical prompts → second one is dropped as duplicate."""
        p = tmp_path / "dupes.jsonl"
        r = _make_record()
        with open(p, "w") as f:
            f.write(json.dumps(r) + "\n")
            f.write(json.dumps({**r, "id": "rec_1"}) + "\n")

        out = tmp_path / "out.jsonl"
        stats = run_pipeline(
            input_path=p, output_path=out,
            min_length=15, dedup_threshold=0.5,
            imbalance_threshold=0.5, n_jobs=1,
        )
        assert stats["kept"] == 1
        assert stats["skipped_duplicate"] == 1
