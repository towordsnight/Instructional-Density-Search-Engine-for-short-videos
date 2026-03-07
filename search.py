#!/usr/bin/env python3
"""
Search pipeline for Short Compilation: query → similarity × density^intent × topical boost → ranked results.
Uses embedding-based intent detection to adapt density weighting per query.
Includes query expansion, result deduplication, and video URL construction.
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

# ---------------------------------------------------------------------------
# Intent detection — embedding-based prototype matching
# ---------------------------------------------------------------------------
# Instructional prototypes: queries where users want to learn something
_INSTRUCTIONAL_PROTOTYPES = [
    "how to do something step by step tutorial",
    "explain the history and meaning behind this",
    "teach me tips and techniques for beginners",
    "learn about the origin and story of a design",
    "guide to understanding why something was created",
]

# Browsing prototypes: queries where users want to explore/see items
_BROWSING_PROTOTYPES = [
    "show me items and collections to browse",
    "what does it look like when wearing jewelry",
    "unboxing haul showcase my collection tour",
    "sharing favorite pieces and accessories",
    "beautiful luxury items and fashion inspiration",
]

# Cached prototype embeddings (populated on first use)
_intent_cache: dict = {}


def _get_intent_embeddings(model: SentenceTransformer) -> tuple[np.ndarray, np.ndarray]:
    """Encode intent prototypes (cached after first call)."""
    if "inst" not in _intent_cache:
        _intent_cache["inst"] = model.encode(_INSTRUCTIONAL_PROTOTYPES, convert_to_numpy=True)
        _intent_cache["browse"] = model.encode(_BROWSING_PROTOTYPES, convert_to_numpy=True)
    return _intent_cache["inst"], _intent_cache["browse"]


def detect_intent(model: SentenceTransformer, query: str) -> float:
    """
    Detect query intent by comparing to instructional vs browsing prototypes.

    Returns intent_weight in [0.3, 1.0]:
        ~0.3 = pure browsing (density has minimal effect)
        ~1.0 = pure instructional (density has full effect)

    Uses cosine similarity between the query embedding and pre-defined
    prototype sentences for each intent category. The weight is computed as:
        ratio = max_inst_sim / (max_inst_sim + max_browse_sim)
        intent_weight = 0.3 + 0.7 * ratio
    """
    inst_embs, browse_embs = _get_intent_embeddings(model)
    query_emb = model.encode([query], convert_to_numpy=True)[0]
    q_norm = norm(query_emb)
    if q_norm == 0:
        return 0.65  # neutral default

    inst_sims = np.dot(inst_embs, query_emb) / (norm(inst_embs, axis=1) * q_norm)
    browse_sims = np.dot(browse_embs, query_emb) / (norm(browse_embs, axis=1) * q_norm)

    inst_score = float(np.max(inst_sims))
    browse_score = float(np.max(browse_sims))

    total = inst_score + browse_score
    if total == 0:
        return 0.65

    ratio = inst_score / total
    return 0.3 + 0.7 * ratio

# ---------------------------------------------------------------------------
# Query expansion synonyms
# ---------------------------------------------------------------------------
_SYNONYMS = {
    "meaning": ["significance", "symbolism", "interpretation"],
    "design": ["aesthetic", "style", "architecture"],
    "history": ["heritage", "legacy", "origin"],
    "tutorial": ["guide", "how-to", "lesson"],
    "learn": ["understand", "discover", "explore"],
    "tips": ["tricks", "advice", "hacks"],
    "recipe": ["cooking", "ingredient", "preparation"],
    "workout": ["exercise", "fitness", "training"],
    "craft": ["diy", "handmade", "build"],
    "review": ["comparison", "overview", "breakdown"],
    "explain": ["interpret", "describe", "clarify"],
    "inspire": ["inspiration", "creative", "motivate"],
}


def _expand_query_terms(terms: set[str]) -> set[str]:
    """Expand query terms with synonyms."""
    expanded = set(terms)
    for term in terms:
        if term in _SYNONYMS:
            expanded.update(_SYNONYMS[term])
    return expanded


def _extract_query_terms(query: str, expand: bool = False) -> set[str]:
    """Extract meaningful terms from query for topical matching."""
    words = re.findall(r"\b[a-z]{2,}\b", query.lower())
    terms = {w for w in words if w not in _STOPWORDS}
    if expand:
        terms = _expand_query_terms(terms)
    return terms


def _query_term_boost(query: str, doc_text: str, boost_strength: float = 0.5, expand: bool = True) -> float:
    """
    Boost score when document contains query terms (e.g. design, Tiffany, history).
    Uses expanded terms by default.
    Returns multiplier in [1.0, 1.0 + boost_strength].
    """
    if not doc_text:
        return 1.0
    terms = _extract_query_terms(query, expand=expand)
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


def _deduplicate_results(indices: np.ndarray, embeddings: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """
    Remove near-duplicate results. Walks sorted indices and skips any document
    with cosine similarity >= threshold to an already-selected result.
    """
    if threshold >= 1.0:
        return indices

    selected = []
    for idx in indices:
        is_dup = False
        for sel_idx in selected:
            emb_a = embeddings[idx]
            emb_b = embeddings[sel_idx]
            norm_a = norm(emb_a)
            norm_b = norm(emb_b)
            if norm_a == 0 or norm_b == 0:
                sim = 0.0
            else:
                sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            selected.append(idx)
    return np.array(selected, dtype=indices.dtype)


def _construct_video_url(meta: dict) -> str:
    """Construct video URL from platform and id when url field is missing."""
    platform = meta.get("platform", "").lower()
    video_id = meta.get("id", "")

    if platform == "youtube" and video_id:
        return f"https://www.youtube.com/shorts/{video_id}"
    if platform == "tiktok" and video_id:
        return f"https://www.tiktok.com/@/video/{video_id}"
    if platform == "instagram" and video_id:
        return f"https://www.instagram.com/reel/{video_id}"
    return ""


def search(
    model: SentenceTransformer,
    query: str,
    embeddings: np.ndarray,
    density_scores: np.ndarray,
    metadata: list[dict],
    top_k: int = 10,
    min_density: float = 0.0,
    topical_boost: float = 0.5,
    dedup_threshold: float = 0.95,
) -> list[dict]:
    """
    Search: similarity × density^intent_weight × topical_boost → ranked results.

    Uses embedding-based intent detection to adapt density weighting:
    - Instructional queries ("how to cook", "tiffany design history") → density^~0.9
      (educational content strongly preferred)
    - Browsing queries ("tiffany", "cute outfits") → density^~0.4
      (density gap flattened, similarity dominates)

    Args:
        model: SentenceTransformer for encoding query
        query: User search query
        embeddings: (n_docs, dim) document embeddings
        density_scores: (n_docs,) instructional density scores
        metadata: List of video metadata dicts (must include 'title', 'transcript')
        top_k: Number of results to return
        min_density: Minimum density (0–1) to include
        topical_boost: Strength of query-term match boost (0.5 = up to 50% boost)
        dedup_threshold: Cosine similarity threshold for dedup (1.0 = disabled)

    Returns:
        List of {rank, score, density, similarity, intent_weight, url, ...metadata}
    """
    query_emb = model.encode([query], convert_to_numpy=True)[0]
    similarities = cosine_similarity(query_emb, embeddings)

    # Intent detection: how much should density influence ranking?
    intent_weight = detect_intent(model, query)

    # Topical boost: videos with query terms (design, Tiffany, history, etc.) rank higher
    boosts = np.ones(len(metadata))
    for i, meta in enumerate(metadata):
        doc_text = f"{meta.get('title', '')} {meta.get('transcript', '')}"
        boosts[i] = _query_term_boost(query, doc_text, boost_strength=topical_boost)

    # Intent-aware density: density^intent_weight
    # High intent_weight (~1.0) → density has full effect (instructional queries)
    # Low intent_weight (~0.3) → density flattened (browsing queries)
    floored_density = np.maximum(density_scores, min_density)
    effective_density = np.power(floored_density, intent_weight)

    # Final score = similarity × density^intent_weight × topical_boost
    final_scores = similarities * effective_density * boosts

    indices = np.argsort(final_scores)[::-1]

    # Deduplication
    indices = _deduplicate_results(indices, embeddings, threshold=dedup_threshold)

    results = []
    for i, idx in enumerate(indices[:top_k]):
        meta = metadata[idx]
        url = meta.get("url", "") or _construct_video_url(meta)
        results.append({
            "rank": i + 1,
            "score": float(final_scores[idx]),
            "similarity": float(similarities[idx]),
            "density": float(density_scores[idx]),
            "effective_density": float(effective_density[idx]),
            "topical_boost": float(boosts[idx]),
            "intent_weight": float(intent_weight),
            "url": url,
            **meta,
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
    parser.add_argument("--dedup-threshold", type=float, default=0.95,
                        help="Cosine similarity threshold for dedup (1.0 = disabled)")
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
        dedup_threshold=args.dedup_threshold,
    )

    intent_weight = results[0]["intent_weight"] if results else 0.0
    print(f"\nResults for: \"{query}\" (intent_weight={intent_weight:.3f})\n")
    for r in results:
        print(f"  #{r['rank']} score={r['score']:.4f} (sim={r['similarity']:.3f} × density={r['density']:.3f}→{r['effective_density']:.3f} × boost={r['topical_boost']:.2f})")
        print(f"      {r.get('title', 'N/A')} [{r.get('platform', '')}]")
        if r.get("url"):
            print(f"      {r['url']}")
        print()


if __name__ == "__main__":
    main()
