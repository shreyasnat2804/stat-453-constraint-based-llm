"""Unit tests for cluster_dataset.py"""

import json
import os
import tempfile
import unittest

from src.crllm.dataset.clustering.cluster_dataset import (
    assign_clusters, build_summary, extract_constraint_categories,
    load_dataset, normalise_category)


class TestNormaliseCategory(unittest.TestCase):
    def test_aliases(self):
        self.assertEqual(normalise_category("strat_with"), "Start_With")
        self.assertEqual(normalise_category("Strat_With"), "Start_With")
        self.assertEqual(normalise_category("end_with"), "End_With")
        self.assertEqual(normalise_category("no_commas"), "No_Commas")
        self.assertEqual(normalise_category("all_lower"), "All_Lower")
        self.assertEqual(normalise_category("Background Info"), "Background_Info")
        self.assertEqual(normalise_category("Role Playing"), "Role_Playing")
        self.assertEqual(
            normalise_category("Numerical Constraints"), "Numerical_Constraints"
        )

    def test_passthrough(self):
        self.assertEqual(normalise_category("Length"), "Length")
        self.assertEqual(normalise_category("Format"), "Format")
        self.assertEqual(normalise_category("Tone"), "Tone")


class TestExtractConstraintCategories(unittest.TestCase):
    def test_dict_format(self):
        r = {
            "added_constraint": {
                "Length": ["Do not exceed 100 words"],
                "Keyword": ["Include 'python'", "Include 'code'"],
                "Format": ["Use bullet points"],
            }
        }
        cats = extract_constraint_categories(r)
        self.assertEqual(cats["Length"], 1)
        self.assertEqual(cats["Keyword"], 2)
        self.assertEqual(cats["Format"], 1)

    def test_list_format(self):
        r = {
            "constraints": [
                {"type": "keywords:existence"},
                {"type": "length_constraint:word_count"},
            ]
        }
        cats = extract_constraint_categories(r)
        self.assertEqual(len(cats), 2)

    def test_empty(self):
        cats = extract_constraint_categories({})
        self.assertEqual(len(cats), 0)

    def test_string_json(self):
        ac = json.dumps({"Tone": ["Be formal"]})
        cats = extract_constraint_categories({"added_constraint": ac})
        self.assertEqual(cats["Tone"], 1)


class TestAssignClusters(unittest.TestCase):
    def _r(self, ac):
        return {"added_constraint": ac}

    def test_single_category(self):
        r = self._r({"Length": ["word count >= 100"]})
        assign_clusters(r)
        self.assertEqual(r["constraint_categories"], ["Length"])
        self.assertEqual(r["primary_cluster"], "Length")
        self.assertEqual(r["cluster_label"], "Length")
        self.assertEqual(r["num_constraint_types"], 1)

    def test_dominant_category(self):
        r = self._r(
            {
                "Keyword": ["kw1", "kw2", "kw3"],
                "Length": ["wc"],
            }
        )
        assign_clusters(r)
        self.assertEqual(r["primary_cluster"], "Keyword")

    def test_tie_is_mixed(self):
        r = self._r(
            {
                "Length": ["wc"],
                "Keyword": ["kw"],
            }
        )
        assign_clusters(r)
        self.assertEqual(r["primary_cluster"], "mixed")

    def test_multi_category_label_sorted(self):
        r = self._r(
            {
                "Tone": ["formal"],
                "Format": ["bullets"],
                "Length": ["50 words"],
            }
        )
        assign_clusters(r)
        self.assertEqual(r["cluster_label"], "Format+Length+Tone")

    def test_empty_constraints(self):
        r = {}
        assign_clusters(r)
        self.assertEqual(r["constraint_categories"], ["unconstrained"])
        self.assertEqual(r["primary_cluster"], "unconstrained")

    def test_alias_normalisation(self):
        r = self._r({"Strat_With": ["Start with Hello"]})
        assign_clusters(r)
        self.assertIn("Start_With", r["constraint_categories"])


class TestBuildSummary(unittest.TestCase):
    def _tagged(self, cats, primary, label, level="L1"):
        return {
            "constraint_categories": cats,
            "primary_cluster": primary,
            "cluster_label": label,
            "num_constraint_types": len(cats),
            "added_constraint_num": level,
        }

    def test_totals(self):
        records = [
            self._tagged(["Keyword"], "Keyword", "Keyword"),
            self._tagged(["Keyword"], "Keyword", "Keyword"),
            self._tagged(["Format", "Length"], "mixed", "Format+Length"),
        ]
        summary = build_summary(records)
        self.assertEqual(summary["total_records"], 3)
        self.assertEqual(summary["by_primary_cluster"]["Keyword"]["count"], 2)
        self.assertEqual(summary["by_primary_cluster"]["mixed"]["count"], 1)

    def test_category_presence(self):
        records = [
            self._tagged(["Keyword", "Length"], "mixed", "Keyword+Length"),
            self._tagged(["Keyword"], "Keyword", "Keyword"),
        ]
        summary = build_summary(records)
        self.assertEqual(summary["category_presence"]["Keyword"]["count"], 2)
        self.assertEqual(summary["category_presence"]["Length"]["count"], 1)


class TestLoadDataset(unittest.TestCase):
    def test_json_array(self):
        data = [{"id": 1}, {"id": 2}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            self.assertEqual(len(load_dataset(path)), 2)
        finally:
            os.unlink(path)

    def test_jsonl(self):
        lines = "\n".join(json.dumps({"id": i}) for i in range(3))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(lines)
            path = f.name
        try:
            self.assertEqual(len(load_dataset(path)), 3)
        finally:
            os.unlink(path)

    def test_lfs_pointer_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(
                "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 999\n"
            )
            path = f.name
        try:
            with self.assertRaises(ValueError, msg="LFS pointer should raise"):
                load_dataset(path)
        finally:
            os.unlink(path)

    def test_empty_file_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_dataset(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
