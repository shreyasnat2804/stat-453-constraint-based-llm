"""
Standalone constraint checker for evaluating LLM constraint-following
on the RECAST-30K dataset.
"""

import json
import re


class ConstraintChecker:
    """Checks whether an LLM response satisfies a list of constraints."""

    def __init__(self):
        self._dispatch = {
            "length_constraint:word_count": self._check_word_count,
            "length_constraint:sentence_count": self._check_sentence_count,
            "length_constraint:paragraph_count": self._check_paragraph_count,
            "keywords:existence": self._check_keyword_existence,
            "keywords:frequency": self._check_keyword_frequency,
            "keywords:forbidden": self._check_keywords_forbidden,
            "start_with": self._check_start_with,
            "end_with": self._check_end_with,
            "capitalization:all_caps_count": self._check_all_caps_count,
            "capitalization:all_lowercase": self._check_all_lowercase,
            "format:bullet_points": self._check_bullet_points,
            "format:numbered_list": self._check_numbered_list,
            "format:sections": self._check_sections,
            "format:json": self._check_json,
            "language:english": self._check_english,
            "punctuation:no_comma": self._check_no_comma,
            "detectable_content:postscript": self._check_postscript,
            "detectable_format:highlight": self._check_highlight,
        }

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_target(constraint: dict):
        """Extract the numeric target from whichever key the constraint uses."""
        for key in ("target", "count", "value", "num"):
            if key in constraint:
                return int(constraint[key])
        return None

    @staticmethod
    def _compare(actual: int, target: int, relation: str) -> bool:
        """Compare *actual* against *target* using the given relation string."""
        relation = relation.strip().lower().replace(" ", "_")
        if relation in ("at_least", "atleast", "greater_than_or_equal", "gte", ">="):
            return actual >= target
        if relation in ("at_most", "atmost", "less_than_or_equal", "lte", "<="):
            return actual <= target
        if relation in ("exactly", "equal", "equals", "eq", "=="):
            return actual == target
        if relation in ("greater_than", "gt", ">"):
            return actual > target
        if relation in ("less_than", "lt", "<"):
            return actual < target
        # Fallback: treat as "at_least"
        return actual >= target

    # ------------------------------------------------------------------ #
    #  Length constraints
    # ------------------------------------------------------------------ #

    def _check_word_count(self, response: str, constraint: dict):
        try:
            actual = len(response.split())
            target = self._get_target(constraint)
            relation = constraint.get("relation", "at_least")
            return self._compare(actual, target, relation)
        except Exception:
            return None

    def _check_sentence_count(self, response: str, constraint: dict):
        try:
            sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
            actual = len(sentences)
            target = self._get_target(constraint)
            relation = constraint.get("relation", "at_least")
            return self._compare(actual, target, relation)
        except Exception:
            return None

    def _check_paragraph_count(self, response: str, constraint: dict):
        try:
            paragraphs = [p.strip() for p in re.split(r'\n\n+', response) if p.strip()]
            actual = len(paragraphs)
            target = self._get_target(constraint)
            relation = constraint.get("relation", "at_least")
            return self._compare(actual, target, relation)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Keyword constraints
    # ------------------------------------------------------------------ #

    def _check_keyword_existence(self, response: str, constraint: dict):
        try:
            keywords = constraint.get("keywords", constraint.get("keyword", []))
            if isinstance(keywords, str):
                keywords = [keywords]
            response_lower = response.lower()
            return all(kw.lower() in response_lower for kw in keywords)
        except Exception:
            return None

    def _check_keyword_frequency(self, response: str, constraint: dict):
        try:
            keyword = constraint.get("keyword", constraint.get("keywords", ""))
            if isinstance(keyword, list):
                keyword = keyword[0]
            actual = response.lower().count(keyword.lower())
            target = self._get_target(constraint)
            relation = constraint.get("relation", "at_least")
            return self._compare(actual, target, relation)
        except Exception:
            return None

    def _check_keywords_forbidden(self, response: str, constraint: dict):
        try:
            forbidden = constraint.get("keywords", constraint.get("forbidden", []))
            if isinstance(forbidden, str):
                forbidden = [forbidden]
            response_lower = response.lower()
            for word in forbidden:
                if word.lower() in response_lower:
                    return False
            return True
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Start / end constraints
    # ------------------------------------------------------------------ #

    def _check_start_with(self, response: str, constraint: dict):
        try:
            target = constraint.get("target", constraint.get("value", ""))
            return response.strip().lower().startswith(target.lower())
        except Exception:
            return None

    def _check_end_with(self, response: str, constraint: dict):
        try:
            target = constraint.get("target", constraint.get("value", ""))
            return response.strip().lower().endswith(target.lower())
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Capitalization constraints
    # ------------------------------------------------------------------ #

    def _check_all_caps_count(self, response: str, constraint: dict):
        try:
            caps_words = [w for w in response.split() if w == w.upper() and w.isalpha()]
            actual = len(caps_words)
            target = self._get_target(constraint)
            relation = constraint.get("relation", "at_most")
            return self._compare(actual, target, relation)
        except Exception:
            return None

    def _check_all_lowercase(self, response: str, constraint: dict):
        try:
            return response == response.lower()
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Format constraints
    # ------------------------------------------------------------------ #

    def _check_bullet_points(self, response: str, constraint: dict):
        try:
            lines = response.split("\n")
            bullet_count = sum(
                1 for line in lines
                if line.strip() and re.match(r'^[\u2022\-\*]\s', line.strip())
            )
            target = self._get_target(constraint)
            relation = constraint.get("relation", "at_least")
            return self._compare(bullet_count, target, relation)
        except Exception:
            return None

    def _check_numbered_list(self, response: str, constraint: dict):
        try:
            lines = response.split("\n")
            numbered = sum(
                1 for line in lines
                if re.match(r'^\d+[.\)]\s', line.strip())
            )
            return numbered > 0
        except Exception:
            return None

    def _check_sections(self, response: str, constraint: dict):
        try:
            section_count = len(re.findall(r'##', response))
            target = self._get_target(constraint)
            relation = constraint.get("relation", "at_least")
            return self._compare(section_count, target, relation)
        except Exception:
            return None

    def _check_json(self, response: str, constraint: dict):
        try:
            json.loads(response.strip())
            return True
        except (json.JSONDecodeError, ValueError):
            return False
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Language constraints
    # ------------------------------------------------------------------ #

    def _check_english(self, response: str, constraint: dict):
        try:
            words = response.split()
            if not words:
                return True
            ascii_words = sum(1 for w in words if w.isascii())
            return (ascii_words / len(words)) > 0.8
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Punctuation constraints
    # ------------------------------------------------------------------ #

    def _check_no_comma(self, response: str, constraint: dict):
        try:
            return "," not in response
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Detectable content / format constraints
    # ------------------------------------------------------------------ #

    def _check_postscript(self, response: str, constraint: dict):
        try:
            lower = response.lower()
            return "p.s." in lower or "ps:" in lower
        except Exception:
            return None

    def _check_highlight(self, response: str, constraint: dict):
        try:
            # Count **...** and *...* occurrences (bold first, then italic)
            bold = re.findall(r'\*\*[^*]+\*\*', response)
            italic = re.findall(r'(?<!\*)\*(?!\*)[^*]+\*(?!\*)', response)
            actual = len(bold) + len(italic)
            target = self._get_target(constraint)
            if target is not None:
                relation = constraint.get("relation", "at_least")
                return self._compare(actual, target, relation)
            return actual > 0
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def check_constraint(self, response: str, constraint: dict):
        """Check a single constraint. Returns True, False, or None."""
        ctype = constraint.get("type", constraint.get("constraint_type", ""))
        handler = self._dispatch.get(ctype)
        if handler is None:
            return None
        return handler(response, constraint)

    def check_all(self, response: str, constraints: list) -> dict:
        """
        Evaluate all constraints and return an aggregate report.

        Returns:
            dict with keys: results, num_constraints, num_checked,
            num_passed, per_constraint_csr, hard_csr
        """
        results = []
        for constraint in constraints:
            ctype = constraint.get("type", constraint.get("constraint_type", ""))
            passed = self.check_constraint(response, constraint)
            results.append({"type": ctype, "passed": passed})

        num_constraints = len(constraints)
        checked = [r for r in results if r["passed"] is not None]
        num_checked = len(checked)
        num_passed = sum(1 for r in checked if r["passed"] is True)

        per_constraint_csr = (num_passed / num_checked) if num_checked > 0 else 0.0
        hard_csr = num_checked > 0 and all(r["passed"] for r in checked)

        return {
            "results": results,
            "num_constraints": num_constraints,
            "num_checked": num_checked,
            "num_passed": num_passed,
            "per_constraint_csr": per_constraint_csr,
            "hard_csr": hard_csr,
        }
