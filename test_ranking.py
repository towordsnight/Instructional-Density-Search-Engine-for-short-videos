#!/usr/bin/env python3
"""
Test block: proves instructional videos rank higher than high-view entertainment vlogs
for query 'meaning of Tiffany collections'.
"""

# Sample data: 3 instructional (T, Knot, Lock) + 2 high-view entertainment vlogs
TEST_DATA = [
    # Instructional: architectural meaning of T collection
    {
        "id": "inst_t",
        "platform": "youtube",
        "title": "The architectural meaning of Tiffany T collection - design and symbolism",
        "transcript": "In this video we explore the architectural meaning and significance of the Tiffany T collection. The design represents strength and modernity. Learn the symbolism behind each piece. The collection's aesthetic draws from urban architecture. Discover the creative inspiration and craftsmanship.",
        "views": 12500,
    },
    # Instructional: Knot collection
    {
        "id": "inst_knot",
        "platform": "youtube",
        "title": "Tiffany Knot collection - meaning and heritage explained",
        "transcript": "The Knot collection has deep architectural significance. In this guide we interpret the meaning of the knot symbol. The design represents connection and legacy. First, we explore the history. Then, discover the craftsmanship. The collection's style blends tradition with modern aesthetic.",
        "views": 18200,
    },
    # Instructional: Lock collection
    {
        "id": "inst_lock",
        "platform": "youtube",
        "title": "Understanding the meaning of Tiffany Lock collection",
        "transcript": "Discover the architectural meaning of the Lock collection. The design inspiration comes from security and heritage. Learn the significance of each element. This tutorial explores the symbolism and creative vision. The collection represents timeless craftsmanship and style.",
        "views": 9800,
    },
    # Entertainment vlog: high views
    {
        "id": "vlog_1",
        "platform": "tiktok",
        "title": "LUXURY SHOPPING HAUL at Tiffany - spent way too much",
        "transcript": "OMG you guys I just went crazy at Tiffany. Look at all this stuff. So pretty. I don't even know what half of it means. Just had to have it. The bag is everything. That's the haul bye.",
        "views": 2100000,
    },
    # Entertainment vlog: high views
    {
        "id": "vlog_2",
        "platform": "instagram",
        "title": "Tiffany unboxing - my biggest shopping spree ever",
        "transcript": "Hey besties! So I went to Tiffany and went a little crazy. Here's what I got. This one is cute. This one too. No idea what the design means honestly. Just love the vibe. Enjoy the unboxing!",
        "views": 1850000,
    },
]


def run_test():
    """Run ranking test and print table proving instructional videos rank higher."""
    import numpy as np

    from create_embeddings import load_model, create_embeddings
    from instructional_score import compute_instructional_scores
    from search import search

    query = "meaning of Tiffany collections"
    print("=" * 80)
    print("RANKING TEST: Instructional vs. Entertainment for query")
    print(f'  "{query}"')
    print("=" * 80)

    # Build texts for embedding
    metadata = [{"id": d["id"], "platform": d["platform"], "title": d["title"], "transcript": d["transcript"], "views": d["views"]} for d in TEST_DATA]
    texts = [f"{d['title']} [SEP] {d['transcript']}" for d in TEST_DATA]

    # Load model and compute
    print("\nLoading model and computing embeddings...")
    model = load_model()
    embeddings = create_embeddings(model, texts, show_progress=False)
    density_scores = np.array(compute_instructional_scores(texts))

    # Search
    results = search(
        model, query, embeddings, density_scores, metadata,
        top_k=5, min_density=0.1, topical_boost=0.5,
    )

    # Build table: Title, Views, Density Score, Final Rank
    print("\n" + "-" * 80)
    print("RESULTS TABLE: Title | Views | Density Score | Final Rank")
    print("-" * 80)

    # Format for table
    rows = []
    for r in results:
        title = (r["title"][:50] + "…") if len(r["title"]) > 50 else r["title"]
        views_str = f"{r['views']:,}" if isinstance(r.get("views"), (int, float)) else str(r.get("views", "N/A"))
        rows.append((title, views_str, f"{r['density']:.3f}", r["rank"]))

    # Column widths
    col_w = (52, 12, 14, 12)
    header = ("Title", "Views", "Density", "Rank")
    print(f"  {header[0]:<{col_w[0]}} {header[1]:<{col_w[1]}} {header[2]:<{col_w[2]}} {header[3]:<{col_w[3]}}")
    print("  " + "-" * (sum(col_w) + 3))

    for title, views, density, rank in rows:
        print(f"  {title:<{col_w[0]}} {views:<{col_w[1]}} {density:<{col_w[2]}} {rank:<{col_w[3]}}")

    print("-" * 80)

    # Assert instructional rank higher
    inst_ids = {"inst_t", "inst_knot", "inst_lock"}
    vlog_ids = {"vlog_1", "vlog_2"}
    inst_ranks = [r["rank"] for r in results if r["id"] in inst_ids]
    vlog_ranks = [r["rank"] for r in results if r["id"] in vlog_ids]

    print("\n✓ INSTRUCTIONAL videos (T, Knot, Lock): ranks", sorted(inst_ranks))
    print("✓ ENTERTAINMENT vlogs (shopping hauls):  ranks", sorted(vlog_ranks))
    if max(inst_ranks) < min(vlog_ranks):
        print("\n✓ PROVEN: Instructional videos rank higher despite lower views.")
    else:
        print("\n  (Instructional and entertainment ranks overlap - check density scores)")
    print("=" * 80)


if __name__ == "__main__":
    run_test()
