#!/usr/bin/env python3
"""
Search pipeline for Short Compilation: query → similarity × density × topical boost → ranked results.
Ranks relevant shorts (e.g. Tiffany design, architecture, history) at the top.
"""

import json
import argparse
import re

import numpy as np
from numpy.linalg import norm

from create_embeddings import load_model
from sentence_transformers import SentenceTransformer

# Stopwords to exclude from query-term extraction
_STOPWORDS = {"a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "for", "with", "by", "is", "are", "was", "were"}


def _extract_query_terms(query: str) -> set[str]:
    """Extract meaningful terms from query for topical matching."""
    words = re.findall(r"\b[a-z]{2,}\b", query.lower())
    return {w for w in words if w not in _STOPWORDS}


def _query_term_boost(query: str, doc_text: str, boost_strength: float = 0.5) -> float:
    """
    Boost score when document contains query terms (e.g. design, Tiffany, history).
    Returns multiplier in [1.0, 1.0 + boost_strength].
    """
    if not doc_text:
        return 1.0
    terms = _extract_query_terms(query)
    if not terms:
        return 1.0
    doc_lower = doc_text.lower()
    matches = sum(1 for t in terms if t in doc_lower)
    ratio = matches / len(terms)  # 0 to 1
    return 1.0 + boost_strength * ratio


def cosine_similarity(query_emb: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and each document."""
    q_norm = norm(query_emb)
    if q_norm == 0:
        return np.zeros(len(doc_embeddings))
    # doc_embeddings: (n_docs, dim), query_emb: (dim,)
    sims = np.dot(doc_embeddings, query_emb) / (norm(doc_embeddings, axis=1) * q_norm)
    return np.clip(sims, -1, 1)  # cosine in [-1, 1]


def search(
    model: SentenceTransformer,
    query: str,
    embeddings: np.ndarray,
    density_scores: np.ndarray,
    metadata: list[dict],
    top_k: int = 10,
    min_density: float = 0.0,
    topical_boost: float = 0.5,
) -> list[dict]:
    """
    Search: similarity × density × topical_boost → ranked results.
    Boosts videos that contain query terms (e.g. design, Tiffany, history).

    Args:
        model: SentenceTransformer for encoding query
        query: User search query
        embeddings: (n_docs, dim) document embeddings
        density_scores: (n_docs,) instructional density scores
        metadata: List of video metadata dicts (must include 'title', 'transcript')
        top_k: Number of results to return
        min_density: Minimum density (0–1) to include
        topical_boost: Strength of query-term match boost (0.5 = up to 50% boost)

    Returns:
        List of {rank, score, density, similarity, ...metadata}
    """
    query_emb = model.encode([query], convert_to_numpy=True)[0]
    similarities = cosine_similarity(query_emb, embeddings)

    # Topical boost: videos with query terms (design, Tiffany, history, etc.) rank higher
    boosts = np.ones(len(metadata))
    for i, meta in enumerate(metadata):
        doc_text = f"{meta.get('title', '')} {meta.get('transcript', '')}"
        boosts[i] = _query_term_boost(query, doc_text, boost_strength=topical_boost)

    # Final score = similarity × density × topical_boost
    effective_density = np.maximum(density_scores, min_density)
    final_scores = similarities * effective_density * boosts

    indices = np.argsort(final_scores)[::-1]

    results = []
    for i, idx in enumerate(indices[:top_k]):
        results.append({
            "rank": i + 1,
            "score": float(final_scores[idx]),
            "similarity": float(similarities[idx]),
            "density": float(density_scores[idx]),
            "topical_boost": float(boosts[idx]),
            **metadata[idx],
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Search shorts: similarity × instructional density")
    parser.add_argument("query", type=str, nargs="?", help="Search query")
    parser.add_argument("--embeddings", "-e", type=str, default="embeddings.npy")
    parser.add_argument("--density", "-d", type=str, default="density_scores.npy")
    parser.add_argument("--metadata", "-m", type=str, default="metadata.json")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--top-k", "-k", type=int, default=10)
    parser.add_argument("--min-density", type=float, default=0.1,
                        help="Floor for density (avoid zeroing non-instructional videos)")
    parser.add_argument("--topical-boost", type=float, default=0.5,
                        help="Boost for query-term matches (0.5 = up to 50%% higher)")
    args = parser.parse_args()

    model = load_model(args.model)
    embeddings = np.load(args.embeddings)
    density_scores = np.load(args.density)
    with open(args.metadata, encoding="utf-8") as f:
        metadata = json.load(f)

    query = args.query
    if not query:
        query = input("Enter search query: ").strip()
    if not query:
        print("No query provided.")
        return

    results = search(
        model, query, embeddings, density_scores, metadata,
        top_k=args.top_k, min_density=args.min_density, topical_boost=args.topical_boost,
    )

    print(f"\nResults for: \"{query}\"\n")
    for r in results:
        print(f"  #{r['rank']} score={r['score']:.4f} (sim×density×boost={r['similarity']:.3f}×{r['density']:.3f}×{r['topical_boost']:.2f})")
        print(f"      {r.get('title', 'N/A')} [{r.get('platform', '')}]")
        print()


if __name__ == "__main__":
    main()
