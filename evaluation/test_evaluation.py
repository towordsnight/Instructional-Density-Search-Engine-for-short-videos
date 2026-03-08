"""Unit tests for evaluation metrics and ground truth validation."""

import json
import math
import os
import pytest

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    ndcg_at_k,
    dcg_at_k,
    mrr,
    average_precision,
    avg_density_at_k,
    avg_instructional_quality_at_k,
    _get_grade,
)


# ---- _get_grade helper tests ----

class TestGetGrade:
    def test_int_labels(self):
        labels = {"a": 3, "b": 0}
        assert _get_grade(labels, "a") == 3
        assert _get_grade(labels, "b") == 0
        assert _get_grade(labels, "c") == 0  # missing → default

    def test_dict_labels(self):
        labels = {"a": {"relevance": 3, "instructional": 2}}
        assert _get_grade(labels, "a", "relevance") == 3
        assert _get_grade(labels, "a", "instructional") == 2

    def test_missing_key_in_dict(self):
        labels = {"a": {"relevance": 3}}
        assert _get_grade(labels, "a", "instructional") == 0


# ---- Precision@K tests ----

class TestPrecisionAtK:
    def test_perfect_ranking(self):
        ranked = ["a", "b", "c", "d", "e"]
        labels = {"a": 3, "b": 3, "c": 2, "d": 2, "e": 2}
        assert precision_at_k(ranked, labels, k=5) == 1.0

    def test_no_relevant(self):
        ranked = ["a", "b", "c", "d", "e"]
        labels = {"a": 0, "b": 1, "c": 0, "d": 1, "e": 0}
        assert precision_at_k(ranked, labels, k=5) == 0.0

    def test_partial_relevant(self):
        ranked = ["a", "b", "c", "d", "e"]
        labels = {"a": 3, "b": 0, "c": 2, "d": 0, "e": 0}
        assert precision_at_k(ranked, labels, k=5) == 0.4

    def test_k_smaller_than_list(self):
        ranked = ["a", "b", "c", "d", "e"]
        labels = {"a": 3, "b": 3, "c": 0, "d": 0, "e": 0}
        assert precision_at_k(ranked, labels, k=2) == 1.0

    def test_empty_ranking(self):
        assert precision_at_k([], {"a": 3}, k=5) == 0.0

    def test_dict_labels(self):
        ranked = ["a", "b", "c"]
        labels = {"a": {"relevance": 3, "instructional": 1}, "b": {"relevance": 0, "instructional": 3}, "c": {"relevance": 2, "instructional": 0}}
        assert precision_at_k(ranked, labels, k=3) == pytest.approx(2/3)


# ---- Recall@K tests ----

class TestRecallAtK:
    def test_all_found(self):
        ranked = ["a", "b", "c", "d", "e"]
        labels = {"a": 3, "b": 2, "c": 0, "d": 0, "e": 0}
        assert recall_at_k(ranked, labels, k=5) == 1.0

    def test_partial_found(self):
        ranked = ["c", "d", "e", "a", "b"]
        labels = {"a": 3, "b": 2, "c": 0, "d": 0, "e": 0}
        assert recall_at_k(ranked, labels, k=3) == 0.0

    def test_no_relevant(self):
        ranked = ["a", "b", "c"]
        labels = {"a": 0, "b": 0, "c": 0}
        assert recall_at_k(ranked, labels, k=3) == 0.0

    def test_dict_labels(self):
        ranked = ["a", "b", "c"]
        labels = {"a": {"relevance": 3, "instructional": 0}, "b": {"relevance": 0, "instructional": 3}, "c": {"relevance": 2, "instructional": 0}}
        # 2 relevant (a=3, c=2), top-3 finds both
        assert recall_at_k(ranked, labels, k=3) == 1.0


# ---- F1@K tests ----

class TestF1AtK:
    def test_perfect(self):
        ranked = ["a", "b", "c", "d", "e"]
        labels = {"a": 3, "b": 3, "c": 2, "d": 2, "e": 2}
        assert f1_at_k(ranked, labels, k=5) == pytest.approx(1.0)

    def test_zero(self):
        ranked = ["a", "b", "c"]
        labels = {"a": 0, "b": 0, "c": 0}
        assert f1_at_k(ranked, labels, k=3) == 0.0

    def test_known_value(self):
        ranked = ["a", "b", "c", "d", "e"]
        labels = {"a": 3, "b": 0, "c": 2, "d": 0, "e": 0}
        # P@5 = 2/5 = 0.4, R@5 = 2/2 = 1.0, F1 = 2*0.4*1.0/1.4
        assert f1_at_k(ranked, labels, k=5) == pytest.approx(2 * 0.4 * 1.0 / 1.4)


# ---- nDCG@K tests ----

class TestNdcgAtK:
    def test_perfect_ranking(self):
        ranked = ["a", "b", "c"]
        labels = {"a": 3, "b": 2, "c": 1}
        assert ndcg_at_k(ranked, labels, k=3) == pytest.approx(1.0)

    def test_reversed_ranking(self):
        ranked = ["c", "b", "a"]
        labels = {"a": 3, "b": 2, "c": 1}
        result = ndcg_at_k(ranked, labels, k=3)
        assert result < 1.0
        assert result > 0.0

    def test_known_value(self):
        ranked = ["a", "b"]
        labels = {"a": 3, "b": 0}
        actual_dcg = dcg_at_k(ranked, labels, k=2)
        assert actual_dcg == pytest.approx(7.0)

    def test_all_irrelevant(self):
        ranked = ["a", "b", "c"]
        labels = {"a": 0, "b": 0, "c": 0}
        assert ndcg_at_k(ranked, labels, k=3) == 0.0

    def test_dict_labels(self):
        ranked = ["a", "b", "c"]
        labels = {"a": {"relevance": 3, "instructional": 0}, "b": {"relevance": 2, "instructional": 0}, "c": {"relevance": 1, "instructional": 0}}
        assert ndcg_at_k(ranked, labels, k=3) == pytest.approx(1.0)


# ---- MRR tests ----

class TestMRR:
    def test_relevant_at_position_1(self):
        ranked = ["a", "b", "c"]
        labels = {"a": 3, "b": 0, "c": 0}
        assert mrr(ranked, labels) == 1.0

    def test_relevant_at_position_3(self):
        ranked = ["a", "b", "c"]
        labels = {"a": 0, "b": 1, "c": 2}
        assert mrr(ranked, labels) == pytest.approx(1.0 / 3)

    def test_none_relevant(self):
        ranked = ["a", "b", "c"]
        labels = {"a": 0, "b": 1, "c": 1}
        assert mrr(ranked, labels) == 0.0

    def test_missing_labels(self):
        ranked = ["a", "b", "c"]
        labels = {"c": 3}
        assert mrr(ranked, labels) == pytest.approx(1.0 / 3)


# ---- MAP tests ----

class TestAveragePrecision:
    def test_perfect_ranking(self):
        ranked = ["a", "b", "c"]
        labels = {"a": 3, "b": 2, "c": 0}
        assert average_precision(ranked, labels) == pytest.approx(1.0)

    def test_worst_ranking(self):
        ranked = ["c", "a", "b"]
        labels = {"a": 3, "b": 2, "c": 0}
        assert average_precision(ranked, labels) == pytest.approx((1/2 + 2/3) / 2)

    def test_no_relevant(self):
        ranked = ["a", "b", "c"]
        labels = {"a": 0, "b": 0, "c": 0}
        assert average_precision(ranked, labels) == 0.0

    def test_single_relevant(self):
        ranked = ["a", "b", "c"]
        labels = {"a": 0, "b": 3, "c": 0}
        assert average_precision(ranked, labels) == pytest.approx(0.5)


# ---- avg_density_at_k tests ----

class TestAvgDensityAtK:
    def test_basic(self):
        results = [{"density": 0.8}, {"density": 0.6}, {"density": 0.4}]
        assert avg_density_at_k(results, k=3) == pytest.approx(0.6)

    def test_empty(self):
        assert avg_density_at_k([], k=5) == 0.0


# ---- avg_instructional_quality_at_k tests ----

class TestAvgInstructionalQualityAtK:
    def test_basic(self):
        ranked = ["a", "b", "c"]
        labels = {"a": {"relevance": 3, "instructional": 3}, "b": {"relevance": 0, "instructional": 1}, "c": {"relevance": 2, "instructional": 2}}
        assert avg_instructional_quality_at_k(ranked, labels, k=3) == pytest.approx(2.0)

    def test_int_labels_fallback(self):
        ranked = ["a", "b"]
        labels = {"a": 3, "b": 1}
        # int labels have no "instructional" key → defaults to 0
        assert avg_instructional_quality_at_k(ranked, labels, k=2) == 0.0

    def test_empty(self):
        assert avg_instructional_quality_at_k([], {}, k=5) == 0.0


# ---- Ground truth validation ----

class TestGroundTruth:
    @pytest.fixture
    def ground_truth(self):
        gt_path = os.path.join(os.path.dirname(__file__), "ground_truth.json")
        with open(gt_path, encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def metadata_ids(self):
        meta_path = os.path.join(os.path.dirname(__file__), "..", "metadata.json")
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        return {m["id"] for m in metadata}

    def test_has_at_least_10_queries(self, ground_truth):
        assert len(ground_truth["queries"]) >= 10

    def test_labeled_ids_subset_of_metadata(self, ground_truth, metadata_ids):
        for q_entry in ground_truth["queries"]:
            labeled_ids = set(q_entry["labels"].keys())
            extra = labeled_ids - metadata_ids
            assert not extra, (
                f"Query '{q_entry['query']}': {len(extra)} labeled IDs not in metadata: {extra}"
            )

    def test_consistent_label_sets(self, ground_truth):
        queries = ground_truth["queries"]
        first_ids = set(queries[0]["labels"].keys())
        for q_entry in queries[1:]:
            assert set(q_entry["labels"].keys()) == first_ids, (
                f"Query '{q_entry['query']}' has different labeled IDs than first query"
            )

    def test_valid_grades(self, ground_truth):
        for q_entry in ground_truth["queries"]:
            for vid, grade in q_entry["labels"].items():
                if isinstance(grade, dict):
                    assert grade.get("relevance", 0) in (0, 1, 2, 3), f"Invalid relevance for {vid}"
                    assert grade.get("instructional", 0) in (0, 1, 2, 3), f"Invalid instructional for {vid}"
                else:
                    assert grade in (0, 1, 2, 3), f"Invalid grade {grade} for {vid}"

    def test_each_query_has_relevant_docs(self, ground_truth):
        for q_entry in ground_truth["queries"]:
            relevant = [v for v, g in q_entry["labels"].items()
                        if (g.get("relevance", 0) if isinstance(g, dict) else g) >= 2]
            assert len(relevant) > 0, f"Query '{q_entry['query']}' has no relevant docs"

    def test_has_dual_labels(self, ground_truth):
        """All labels should have both relevance and instructional keys."""
        for q_entry in ground_truth["queries"]:
            for vid, grade in q_entry["labels"].items():
                assert isinstance(grade, dict), f"Expected dict label for {vid}, got {type(grade)}"
                assert "relevance" in grade, f"Missing 'relevance' for {vid}"
                assert "instructional" in grade, f"Missing 'instructional' for {vid}"
