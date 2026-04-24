"""Tests for augmentation orchestrator."""

import json
import zipfile

import pytest
from unittest.mock import patch

from src.crllm.dataset.augmentation.augment import (
    _load_records,
    _write_jsonl,
    _read_jsonl,
    run_augmentation,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_RECORDS = [
    {
        "winner_prompt": "Write a 100 word essay about 'python' using bullet points.",
        "response_of_winner_prompt": "Here is the essay...",
        "added_constraint": {
            "Length": ["at most 100 words"],
            "Keyword": ["include 'python'"],
        },
        "added_constraint_num": 2,
        "rule_evaluate_dict": {},
        "id": "rec_001",
    },
    {
        "winner_prompt": "Describe the weather in 50 words.",
        "response_of_winner_prompt": "The weather is sunny...",
        "added_constraint": {"Length": ["at most 50 words"]},
        "added_constraint_num": 1,
        "rule_evaluate_dict": {},
        "id": "rec_002",
    },
]


# ── TestLoadRecords ──────────────────────────────────────────────────────────


class TestLoadRecords:
    def test_loads_json_array(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps(SAMPLE_RECORDS))
        records = _load_records(p)
        assert len(records) == 2
        assert records[0]["id"] == "rec_001"

    def test_loads_jsonl(self, tmp_path):
        p = tmp_path / "data.jsonl"
        with open(p, "w") as f:
            for rec in SAMPLE_RECORDS:
                f.write(json.dumps(rec) + "\n")
        records = _load_records(p)
        assert len(records) == 2

    def test_skips_blank_lines_jsonl(self, tmp_path):
        p = tmp_path / "data.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps(SAMPLE_RECORDS[0]) + "\n")
            f.write("\n")
            f.write(json.dumps(SAMPLE_RECORDS[1]) + "\n")
        records = _load_records(p)
        assert len(records) == 2

    def test_loads_zipped_jsonl(self, tmp_path):
        p = tmp_path / "data.jsonl.zip"
        with zipfile.ZipFile(p, "w") as z:
            content = "\n".join(json.dumps(r) for r in SAMPLE_RECORDS)
            z.writestr("data.jsonl", content)
        records = _load_records(p)
        assert len(records) == 2
        assert records[0]["id"] == "rec_001"

    def test_invalid_json_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json at all {{{")
        with pytest.raises(json.JSONDecodeError):
            _load_records(p)


# ── TestWriteAndReadJsonl ────────────────────────────────────────────────────


class TestWriteAndReadJsonl:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "out.jsonl"
        _write_jsonl(p, SAMPLE_RECORDS)
        result = _read_jsonl(p)
        assert len(result) == 2
        assert result[0]["id"] == "rec_001"
        assert result[1]["id"] == "rec_002"

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "nested" / "dir" / "out.jsonl"
        _write_jsonl(p, SAMPLE_RECORDS)
        assert p.exists()


# ── TestRunAugmentation ──────────────────────────────────────────────────────


def _mock_lex_pipeline(input_path, output_path, **kwargs):
    """Mock lexical edit: copy input with _lex suffix on ids."""
    records = _read_jsonl(input_path)
    for r in records:
        r["id"] = r["id"] + "_lex"
        r["augmentation_method"] = "lexical_edit"
    _write_jsonl(output_path, records)
    return {"total": len(records), "augmented": len(records), "errors": 0}


def _mock_bt_pipeline(input_path, output_path, **kwargs):
    """Mock back-translation: copy input with _bt suffix on ids."""
    records = _read_jsonl(input_path if isinstance(input_path, str) else str(input_path))
    for r in records:
        r["id"] = r["id"] + "_bt"
        r["augmentation_method"] = "back_translation"
    _write_jsonl(
        output_path if not isinstance(output_path, str) else __import__("pathlib").Path(output_path),
        records,
    )
    return {"total": len(records), "augmented": len(records), "fell_back": 0, "skipped": False}


class TestRunAugmentation:
    def _run_with_mocks(self, tmp_path, input_records, **kwargs):
        inp = tmp_path / "input.json"
        out = tmp_path / "output.jsonl"
        inp.write_text(json.dumps(input_records))

        with patch(
            "src.crllm.dataset.augmentation.augment.run_lexical_edit_pipeline",
            side_effect=_mock_lex_pipeline,
        ):
            with patch(
                "src.crllm.dataset.augmentation.augment.run_back_translation_pipeline",
                side_effect=_mock_bt_pipeline,
            ):
                stats = run_augmentation(str(inp), str(out), force=True, **kwargs)

        output_records = _read_jsonl(out)
        return stats, output_records

    def test_produces_3x_records(self, tmp_path):
        stats, output = self._run_with_mocks(tmp_path, SAMPLE_RECORDS)
        assert stats["status"] == "complete"
        assert stats["original"] == 2
        assert stats["lexical_edit"] == 2
        assert stats["back_translation"] == 2
        assert stats["total_output"] == 6
        assert len(output) == 6

    def test_original_records_first(self, tmp_path):
        _, output = self._run_with_mocks(tmp_path, SAMPLE_RECORDS)
        assert output[0]["id"] == "rec_001"
        assert output[1]["id"] == "rec_002"

    def test_lex_records_in_middle(self, tmp_path):
        _, output = self._run_with_mocks(tmp_path, SAMPLE_RECORDS)
        assert output[2]["id"] == "rec_001_lex"
        assert output[3]["id"] == "rec_002_lex"

    def test_bt_records_at_end(self, tmp_path):
        _, output = self._run_with_mocks(tmp_path, SAMPLE_RECORDS)
        assert output[4]["id"] == "rec_001_bt"
        assert output[5]["id"] == "rec_002_bt"

    def test_idempotency_skips(self, tmp_path):
        inp = tmp_path / "input.json"
        out = tmp_path / "output.jsonl"
        inp.write_text(json.dumps(SAMPLE_RECORDS))
        out.write_text("existing\n")

        stats = run_augmentation(str(inp), str(out), force=False)
        assert stats["status"] == "skipped"
        assert out.read_text() == "existing\n"

    def test_force_overwrites(self, tmp_path):
        stats, output = self._run_with_mocks(tmp_path, SAMPLE_RECORDS)
        assert stats["status"] == "complete"
        assert len(output) == 6

    def test_accepts_jsonl_input(self, tmp_path):
        inp = tmp_path / "input.jsonl"
        out = tmp_path / "output.jsonl"
        with open(inp, "w") as f:
            for rec in SAMPLE_RECORDS:
                f.write(json.dumps(rec) + "\n")

        with patch(
            "src.crllm.dataset.augmentation.augment.run_lexical_edit_pipeline",
            side_effect=_mock_lex_pipeline,
        ):
            with patch(
                "src.crllm.dataset.augmentation.augment.run_back_translation_pipeline",
                side_effect=_mock_bt_pipeline,
            ):
                stats = run_augmentation(str(inp), str(out), force=True)

        assert stats["status"] == "complete"
        assert stats["total_output"] == 6

    def test_single_record(self, tmp_path):
        stats, output = self._run_with_mocks(tmp_path, [SAMPLE_RECORDS[0]])
        assert stats["total_output"] == 3
        assert output[0]["id"] == "rec_001"
        assert output[1]["id"] == "rec_001_lex"
        assert output[2]["id"] == "rec_001_bt"
