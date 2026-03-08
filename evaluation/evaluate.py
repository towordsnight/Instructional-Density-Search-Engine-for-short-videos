#!/usr/bin/env python3
"""
Evaluate search engine against 3 baselines using ground-truth relevance labels.

Baselines:
  B1: Similarity only  — pure cosine similarity (density=1, no topical boost)
  B2: View-count        — sort by views descending
  B3: Our system        — similarity × density × topical_boost

Uses pooled evaluation: only videos with ground-truth labels are considered.
Supports dual labels: {relevance, instructional} per video per query.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from create_embeddings import load_model
from search import search, cosine_similarity
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    ndcg_at_k,
    mrr,
    average_precision,
    avg_density_at_k,
    avg_instructional_quality_at_k,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METRIC_NAMES = ["P@5", "R@5", "F1@5", "nDCG@10", "MRR", "MAP", "AvgDens@5", "AvgInstr@5"]
RELEVANCE_METRICS = ["P@5", "R@5", "F1@5", "nDCG@10", "MRR", "MAP"]


def load_ground_truth():
    gt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth.json")
    with open(gt_path, encoding="utf-8") as f:
        return json.load(f)


def load_data():
    model = load_model()
    embeddings = np.load(os.path.join(BASE_DIR, "embeddings.npy"))
    density_scores = np.load(os.path.join(BASE_DIR, "density_scores.npy"))
    with open(os.path.join(BASE_DIR, "metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    return model, embeddings, density_scores, metadata


def _filter_to_labeled(results, labels):
    """Keep only results whose ID appears in the label pool."""
    labeled_ids = set(labels.keys())
    return [r for r in results if r["id"] in labeled_ids]


def baseline_similarity_only(model, query, embeddings, density_scores, metadata, top_k=100):
    """B1: Pure cosine similarity — override density to 1.0, no topical boost."""
    ones = np.ones_like(density_scores)
    results = search(
        model, query, embeddings, ones, metadata,
        top_k=top_k, min_density=1.0, topical_boost=0.0, dedup_threshold=1.0,
    )
    id_to_density = {m["id"]: float(density_scores[i]) for i, m in enumerate(metadata)}
    for r in results:
        r["density"] = id_to_density.get(r["id"], 0.0)
    return results


def baseline_view_count(metadata, density_scores, top_k=100):
    """B2: Sort by view count descending."""
    indexed = list(enumerate(metadata))
    indexed.sort(key=lambda x: x[1].get("views", 0), reverse=True)
    results = []
    for rank, (i, meta) in enumerate(indexed[:top_k]):
        results.append({
            "rank": rank + 1,
            "id": meta["id"],
            "title": meta.get("title", ""),
            "views": meta.get("views", 0),
            "density": float(density_scores[i]),
            "score": meta.get("views", 0),
        })
    return results


def baseline_our_system(model, query, embeddings, density_scores, metadata, top_k=100):
    """B3: Our full system — similarity × density × topical_boost."""
    return search(
        model, query, embeddings, density_scores, metadata,
        top_k=top_k, min_density=0.1, topical_boost=0.5, dedup_threshold=1.0,
    )


def evaluate_baseline(results, labels, k_precision=5, k_ndcg=10):
    """Compute all metrics for one baseline on one query (pooled)."""
    filtered = _filter_to_labeled(results, labels)
    ranked_ids = [r["id"] for r in filtered]
    return {
        "P@5": precision_at_k(ranked_ids, labels, k=k_precision),
        "R@5": recall_at_k(ranked_ids, labels, k=k_precision),
        "F1@5": f1_at_k(ranked_ids, labels, k=k_precision),
        "nDCG@10": ndcg_at_k(ranked_ids, labels, k=k_ndcg),
        "MRR": mrr(ranked_ids, labels),
        "MAP": average_precision(ranked_ids, labels),
        "AvgDens@5": avg_density_at_k(filtered, k=k_precision),
        "AvgInstr@5": avg_instructional_quality_at_k(ranked_ids, labels, k=k_precision),
    }


def _performance_analysis(all_results, baselines, queries):
    """Print detailed performance analysis."""
    print(f"\n{'='*90}")
    print("=== Performance Analysis ===")
    print(f"{'='*90}")

    our = all_results["Our system"]
    sim = all_results["Similarity only"]
    n = len(our)

    def avg(metric_list, key):
        return sum(m[key] for m in metric_list) / len(metric_list)

    our_avgs = {m: avg(our, m) for m in METRIC_NAMES}
    sim_avgs = {m: avg(sim, m) for m in METRIC_NAMES}

    # Per-metric comparison
    print("\n  Our System vs Similarity-Only (per-query wins):")
    for metric in RELEVANCE_METRICS:
        wins = sum(1 for i in range(n) if our[i][metric] > sim[i][metric])
        ties = sum(1 for i in range(n) if our[i][metric] == sim[i][metric])
        losses = n - wins - ties
        delta = our_avgs[metric] - sim_avgs[metric]
        sign = "+" if delta >= 0 else ""
        print(f"    {metric:<10s}: {wins}W / {ties}T / {losses}L  (avg {sign}{delta:.3f})")

    # Density improvement
    density_delta = our_avgs["AvgDens@5"] - sim_avgs["AvgDens@5"]
    print(f"\n  Instructional Density (system-computed):")
    print(f"    Our system: {our_avgs['AvgDens@5']:.3f}   Sim-only: {sim_avgs['AvgDens@5']:.3f}   Delta: +{density_delta:.3f} ({density_delta/max(sim_avgs['AvgDens@5'],0.001)*100:.1f}%)")

    # Instructional quality improvement (from ground truth labels)
    instr_delta = our_avgs["AvgInstr@5"] - sim_avgs["AvgInstr@5"]
    print(f"\n  Instructional Quality (human-labeled):")
    print(f"    Our system: {our_avgs['AvgInstr@5']:.3f}   Sim-only: {sim_avgs['AvgInstr@5']:.3f}   Delta: {'+' if instr_delta >= 0 else ''}{instr_delta:.3f}")

    # Weakest queries
    print(f"\n  Weakest Queries (lowest nDCG@10 for Our System):")
    query_scores = [(queries[i]["query"], our[i]["nDCG@10"], sim[i]["nDCG@10"]) for i in range(n)]
    query_scores.sort(key=lambda x: x[1])
    for q, our_score, sim_score in query_scores[:3]:
        print(f"    \"{q}\": nDCG={our_score:.3f} (sim-only={sim_score:.3f})")

    # Strongest queries (Our > Sim)
    print(f"\n  Strongest Queries (Our System beats Sim-only on nDCG@10):")
    wins_list = [(queries[i]["query"], our[i]["nDCG@10"], sim[i]["nDCG@10"])
                 for i in range(n) if our[i]["nDCG@10"] > sim[i]["nDCG@10"]]
    if wins_list:
        for q, our_score, sim_score in sorted(wins_list, key=lambda x: x[1]-x[2], reverse=True):
            print(f"    \"{q}\": nDCG={our_score:.3f} vs {sim_score:.3f} (+{our_score-sim_score:.3f})")
    else:
        print("    (none)")

    # Verdict
    print(f"\n  {'='*70}")
    our_rel_wins = sum(1 for m in RELEVANCE_METRICS if our_avgs[m] > sim_avgs[m])
    sim_rel_wins = sum(1 for m in RELEVANCE_METRICS if sim_avgs[m] > our_avgs[m])

    print(f"  VERDICT:")
    if our_rel_wins > sim_rel_wins:
        print(f"    Our system WINS on {our_rel_wins}/{len(RELEVANCE_METRICS)} relevance metrics")
        print(f"    AND improves instructional density by +{density_delta:.3f}")
    elif our_rel_wins == sim_rel_wins:
        print(f"    Our system TIES on relevance ({our_rel_wins}/{len(RELEVANCE_METRICS)} metrics)")
        print(f"    BUT improves instructional density by +{density_delta:.3f}")
        print(f"    → Density weighting adds educational value without hurting relevance")
    else:
        rel_cost = sim_avgs["nDCG@10"] - our_avgs["nDCG@10"]
        print(f"    Our system loses on {sim_rel_wins}/{len(RELEVANCE_METRICS)} relevance metrics")
        print(f"    Relevance cost: -{rel_cost:.3f} nDCG@10")
        print(f"    Density gain:   +{density_delta:.3f} AvgDens@5")
        print(f"    Instr. quality: {'+' if instr_delta >= 0 else ''}{instr_delta:.3f} AvgInstr@5")
        if density_delta > rel_cost * 2:
            print(f"    → Density gain outweighs relevance cost — acceptable trade-off")
        else:
            print(f"    → Consider tuning density weight or topical boost parameters")
    print(f"  {'='*70}")

    return {
        "our_system": our_avgs,
        "similarity_only": sim_avgs,
        "verdict": "win" if our_rel_wins > sim_rel_wins else "tie" if our_rel_wins == sim_rel_wins else "loss",
        "density_improvement": density_delta,
        "instructional_improvement": instr_delta,
    }


def run_evaluation(model=None, embeddings=None, density_scores=None, metadata=None):
    """Run full evaluation. Returns structured results dict for API use."""
    if model is None:
        model, embeddings, density_scores, metadata = load_data()

    gt = load_ground_truth()
    queries = gt["queries"]

    meta_ids = {m["id"] for m in metadata}
    for q_entry in queries:
        labeled_ids = set(q_entry["labels"].keys())
        missing = labeled_ids - meta_ids
        if missing:
            print(f"  Warning: query \"{q_entry['query']}\" has {len(missing)} labeled IDs not in metadata")

    baselines = ["Similarity only", "View-count", "Our system"]
    all_results = {b: [] for b in baselines}
    per_query = []

    for qi, q_entry in enumerate(queries):
        query = q_entry["query"]
        labels = q_entry["labels"]

        r1 = baseline_similarity_only(model, query, embeddings, density_scores, metadata)
        r2 = baseline_view_count(metadata, density_scores)
        r3 = baseline_our_system(model, query, embeddings, density_scores, metadata)

        m1 = evaluate_baseline(r1, labels)
        m2 = evaluate_baseline(r2, labels)
        m3 = evaluate_baseline(r3, labels)

        all_results["Similarity only"].append(m1)
        all_results["View-count"].append(m2)
        all_results["Our system"].append(m3)

        per_query.append({
            "query": query,
            "Similarity only": m1,
            "View-count": m2,
            "Our system": m3,
        })

    # Compute aggregates
    aggregates = {}
    for name in baselines:
        metrics_list = all_results[name]
        n = len(metrics_list)
        aggregates[name] = {m: sum(x[m] for x in metrics_list) / n for m in METRIC_NAMES}

    return {
        "num_queries": len(queries),
        "num_videos": len(metadata),
        "num_labeled": len(queries[0]["labels"]) if queries else 0,
        "per_query": per_query,
        "aggregates": aggregates,
        "all_results": all_results,
    }


def _print_table_header():
    print(f"  {'Baseline':<20s} | {'P@5':>6s} | {'R@5':>6s} | {'F1@5':>6s} | {'nDCG@10':>7s} | {'MRR':>6s} | {'MAP':>6s} | {'Dens@5':>6s} | {'Instr@5':>7s}")
    print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")


def _print_table_row(name, m):
    print(f"  {name:<20s} | {m['P@5']:>6.3f} | {m['R@5']:>6.3f} | {m['F1@5']:>6.3f} | {m['nDCG@10']:>7.3f} | {m['MRR']:>6.3f} | {m['MAP']:>6.3f} | {m['AvgDens@5']:>6.3f} | {m['AvgInstr@5']:>7.3f}")


def main():
    print("Loading model and data...")
    model, embeddings, density_scores, metadata = load_data()
    gt = load_ground_truth()
    queries = gt["queries"]

    n_labeled = len(queries[0]["labels"]) if queries else 0
    print(f"Dataset: {len(metadata)} videos, {n_labeled} labeled, {len(queries)} queries")
    print(f"Evaluation mode: pooled (only labeled videos counted in metrics)")
    print(f"Labels: dual (relevance + instructional quality)\n")

    result = run_evaluation(model, embeddings, density_scores, metadata)
    baselines = ["Similarity only", "View-count", "Our system"]

    for qi, pq in enumerate(result["per_query"]):
        print(f"\n=== Query {qi+1}: \"{pq['query']}\" ===")
        _print_table_header()
        for name in baselines:
            _print_table_row(name, pq[name])

    # Aggregate table
    print(f"\n{'='*100}")
    print(f"=== Aggregate Results (averaged over {result['num_queries']} queries) ===")
    print(f"{'='*100}")
    _print_table_header()
    for name in baselines:
        _print_table_row(name, result["aggregates"][name])

    # Win summary
    print(f"\n{'='*100}")
    print("=== Win Summary ===")
    for metric in RELEVANCE_METRICS:
        scores = {name: result["aggregates"][name][metric] for name in baselines}
        winner = max(scores, key=scores.get)
        print(f"  {metric:<10s}: {winner} ({scores[winner]:.3f})")

    # Performance analysis
    _performance_analysis(result["all_results"], baselines, queries)


if __name__ == "__main__":
    main()
