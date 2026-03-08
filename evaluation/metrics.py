"""IR evaluation metrics: Precision@K, Recall@K, F1@K, nDCG@K, MRR, MAP, avg density, instructional quality."""

import math


def _get_grade(labels: dict, vid: str, key: str = "relevance", default: int = 0) -> int:
    """Extract a grade from labels that may be int or dict with {relevance, instructional}."""
    val = labels.get(vid, default)
    if isinstance(val, dict):
        return val.get(key, default)
    # Plain int labels: treat as relevance only
    if key == "relevance":
        return val
    return default


def _get_all_grades(labels: dict, key: str = "relevance") -> dict[str, int]:
    """Convert labels to flat {vid: grade} dict for a given key."""
    return {vid: _get_grade(labels, vid, key) for vid in labels}


def precision_at_k(ranked_ids: list[str], labels: dict, k: int = 5, threshold: int = 2) -> float:
    """Fraction of top-K results that are relevant (grade >= threshold)."""
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for vid in top_k if _get_grade(labels, vid) >= threshold)
    return relevant / len(top_k)


def recall_at_k(ranked_ids: list[str], labels: dict, k: int = 5, threshold: int = 2) -> float:
    """Fraction of all relevant documents found in top-K results (grade >= threshold)."""
    grades = _get_all_grades(labels)
    total_relevant = sum(1 for v in grades.values() if v >= threshold)
    if total_relevant == 0:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for vid in top_k if _get_grade(labels, vid) >= threshold)
    return hits / total_relevant


def f1_at_k(ranked_ids: list[str], labels: dict, k: int = 5, threshold: int = 2) -> float:
    """Harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(ranked_ids, labels, k, threshold)
    r = recall_at_k(ranked_ids, labels, k, threshold)
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def dcg_at_k(ranked_ids: list[str], labels: dict, k: int = 10) -> float:
    """Discounted cumulative gain at K using graded relevance."""
    score = 0.0
    for i, vid in enumerate(ranked_ids[:k]):
        rel = _get_grade(labels, vid)
        score += (2 ** rel - 1) / math.log2(i + 2)
    return score


def ndcg_at_k(ranked_ids: list[str], labels: dict, k: int = 10) -> float:
    """Normalized discounted cumulative gain at K."""
    actual_dcg = dcg_at_k(ranked_ids, labels, k)
    grades = _get_all_grades(labels)
    ideal_ids = sorted(grades.keys(), key=lambda x: grades[x], reverse=True)
    ideal_dcg = dcg_at_k(ideal_ids, labels, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def mrr(ranked_ids: list[str], labels: dict, threshold: int = 2) -> float:
    """Mean reciprocal rank: 1/position of first relevant result (grade >= threshold)."""
    for i, vid in enumerate(ranked_ids):
        if _get_grade(labels, vid) >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(ranked_ids: list[str], labels: dict, threshold: int = 2) -> float:
    """Average precision for a single query."""
    relevant_count = 0
    precision_sum = 0.0
    grades = _get_all_grades(labels)
    total_relevant = sum(1 for v in grades.values() if v >= threshold)
    if total_relevant == 0:
        return 0.0
    for i, vid in enumerate(ranked_ids):
        if _get_grade(labels, vid) >= threshold:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    return precision_sum / total_relevant


def mean_average_precision(all_ranked: list[list[str]], all_labels: list[dict], threshold: int = 2) -> float:
    """MAP across multiple queries."""
    if not all_ranked:
        return 0.0
    return sum(average_precision(r, l, threshold) for r, l in zip(all_ranked, all_labels)) / len(all_ranked)


def avg_density_at_k(results: list[dict], k: int = 5) -> float:
    """Average instructional density score of top-K results."""
    top_k = results[:k]
    if not top_k:
        return 0.0
    return sum(r.get("density", 0.0) for r in top_k) / len(top_k)


def avg_instructional_quality_at_k(ranked_ids: list[str], labels: dict, k: int = 5) -> float:
    """Average instructional quality grade of top-K results (from ground truth labels)."""
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    return sum(_get_grade(labels, vid, key="instructional") for vid in top_k) / len(top_k)
