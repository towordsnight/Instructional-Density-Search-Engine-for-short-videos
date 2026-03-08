import json
import numpy as np

from create_embeddings import load_model
from search import search


def precision_at_k(results, relevant_ids, k=5):
    top_k_ids = [r["id"] for r in results[:k]]
    hits = sum(1 for rid in top_k_ids if rid in relevant_ids)
    return hits / k


def recall_at_k(results, relevant_ids, k=5):
    top_k_ids = [r["id"] for r in results[:k]]
    hits = sum(1 for rid in top_k_ids if rid in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def f1_at_k(results, relevant_ids, k=5):
    p = precision_at_k(results, relevant_ids, k)
    r = recall_at_k(results, relevant_ids, k)
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def main():
    queries = [
        {
            "query": "meaning of Tiffany collections",
            "relevant_ids": {
                "yt_tft9uUJFrkE",
                "yt_sW5QGWCQgHY",
                "yt_VmRvzOniIdg",
            },
        },
        {
            "query": "how to cook steak",
            "relevant_ids": {
                "yt_f-twQeqQAZE",
                "yt_AmC9SmCBUj4",
            },
        },
        {
            "query": "quick beginner workout",
            "relevant_ids": {
                "yt_jx9I-1D6GLs",
                "yt_Qn1voplJI4I",
                "yt_B1KbolzpWD4",
                "yt_LFF7iCW5Y2E",
            },
        },
        {
            "query": "how to upgrade your graphics card",
            "relevant_ids": {
                "yt_Y2ZG9SrxNo8",
            },
        },
        {
            "query": "clean your monitor safely",
            "relevant_ids": {
                "yt_FPe3TAKOT9Y",
            },
        },
    ]

    model = load_model()
    embeddings = np.load("embeddings.npy")
    density_scores = np.load("density_scores.npy")

    with open("metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    p5_scores = []
    r5_scores = []
    f1_scores = []

    for q in queries:
        results = search(
            model,
            q["query"],
            embeddings,
            density_scores,
            metadata,
            top_k=5,
            min_density=0.1,
            topical_boost=0.5,
        )

        relevant_ids = q["relevant_ids"]

        p5 = precision_at_k(results, relevant_ids, k=5)
        r5 = recall_at_k(results, relevant_ids, k=5)
        f1 = f1_at_k(results, relevant_ids, k=5)

        p5_scores.append(p5)
        r5_scores.append(r5)
        f1_scores.append(f1)

        print(f"\nQuery: {q['query']}")
        print("Top 5 IDs:", [r["id"] for r in results[:5]])
        print(f"Precision@5: {p5:.3f}")
        print(f"Recall@5:    {r5:.3f}")
        print(f"F1@5:        {f1:.3f}")

    print("\n===== AVERAGES =====")
    print(f"Mean Precision@5: {np.mean(p5_scores):.3f}")
    print(f"Mean Recall@5:    {np.mean(r5_scores):.3f}")
    print(f"Mean F1@5:        {np.mean(f1_scores):.3f}")


if __name__ == "__main__":
    main()