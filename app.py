#!/usr/bin/env python3
"""
Flask web UI for Short Compilation search engine.
Serves a single-page interface for searching indexed short-form videos.
"""

import json
import os

import numpy as np
from flask import Flask, request, jsonify, send_from_directory

from create_embeddings import load_model
from search import search
from evaluation.evaluate import run_evaluation

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# Module globals — populated on startup
model = None
embeddings = None
density_scores = None
metadata = None


def _load_data():
    """Load model and data files into module globals."""
    global model, embeddings, density_scores, metadata
    model = load_model()
    embeddings = np.load("embeddings.npy")
    density_scores = np.load("density_scores.npy")
    with open("metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)


@app.route("/")
def index():
    """Serve the React frontend."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/search")
def api_search():
    """Search endpoint: GET /api/search?q=...&k=10"""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "Missing required parameter: q"}), 400

    k = request.args.get("k", 10, type=int)
    min_density = request.args.get("min_density", 0.1, type=float)
    results = search(
        model, q, embeddings, density_scores, metadata,
        top_k=k, min_density=min_density,
    )
    return jsonify(results)


@app.route("/api/evaluate")
def api_evaluate():
    """Run evaluation against ground-truth baselines. GET /api/evaluate"""
    result = run_evaluation(model, embeddings, density_scores, metadata)
    # Remove all_results (large, redundant with per_query)
    result.pop("all_results", None)
    return jsonify(result)


@app.route("/api/stats")
def api_stats():
    """Dataset statistics endpoint."""
    platforms = {}
    for m in metadata:
        p = m.get("platform", "unknown")
        platforms[p] = platforms.get(p, 0) + 1

    return jsonify({
        "total_videos": len(metadata),
        "avg_density": float(np.mean(density_scores)),
        "platforms": platforms,
    })


if __name__ == "__main__":
    _load_data()
    app.run(debug=True, port=5001)
