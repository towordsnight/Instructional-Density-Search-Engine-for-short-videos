#!/usr/bin/env python3
"""
Phase 4 tests — Flask web UI endpoints.
Uses Flask test client with mocked model/data globals (no actual model loading).
"""

import json
import numpy as np
import pytest

import app as app_module
from app import app

# ---------------------------------------------------------------------------
# Mock data: 3 videos, 384-dim embeddings (matches all-MiniLM-L6-v2 output)
# ---------------------------------------------------------------------------
MOCK_METADATA = [
    {"id": "yt_001", "platform": "youtube", "title": "5 Minute Ab Workout",
     "transcript": "Today we are doing a quick five minute ab workout. Start with planks."},
    {"id": "tt_001", "platform": "tiktok", "title": "POV: Cooking at 2am",
     "transcript": "It is 2am and you are hungry. Make instant noodles with an egg."},
    {"id": "ig_001", "platform": "instagram", "title": "Sunset at the beach",
     "transcript": "Golden hour hits different. Watching the sun go down. Pure peace."},
]

MOCK_EMBEDDINGS = np.random.RandomState(42).rand(3, 384).astype(np.float32)
MOCK_DENSITY = np.array([0.6, 0.3, 0.1], dtype=np.float32)


class _FakeModel:
    """Minimal stand-in for SentenceTransformer; returns a random 384-dim vector."""
    def encode(self, texts, convert_to_numpy=True, **kwargs):
        rng = np.random.RandomState(hash(texts[0]) % 2**31)
        return rng.rand(len(texts), 384).astype(np.float32)


@pytest.fixture(autouse=True)
def _patch_globals():
    """Inject mock data into app module globals before every test."""
    app_module.model = _FakeModel()
    app_module.embeddings = MOCK_EMBEDDINGS.copy()
    app_module.density_scores = MOCK_DENSITY.copy()
    app_module.metadata = [dict(m) for m in MOCK_METADATA]
    yield


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── TestSearchEndpoint ────────────────────────────────────────────────────

class TestSearchEndpoint:
    def test_returns_json_list(self, client):
        rv = client.get("/api/search?q=workout")
        assert rv.status_code == 200
        data = rv.get_json()
        assert isinstance(data, list)

    def test_requires_query(self, client):
        rv = client.get("/api/search")
        assert rv.status_code == 400
        assert "error" in rv.get_json()

    def test_respects_top_k(self, client):
        rv = client.get("/api/search?q=workout&k=2")
        data = rv.get_json()
        assert len(data) <= 2

    def test_result_has_expected_keys(self, client):
        rv = client.get("/api/search?q=cooking")
        data = rv.get_json()
        assert len(data) > 0
        first = data[0]
        for key in ("rank", "score", "similarity", "density", "url", "title", "platform"):
            assert key in first, f"Missing key: {key}"

    def test_results_ordered_by_score(self, client):
        rv = client.get("/api/search?q=fitness")
        data = rv.get_json()
        scores = [r["score"] for r in data]
        assert scores == sorted(scores, reverse=True)


# ── TestStatsEndpoint ─────────────────────────────────────────────────────

class TestStatsEndpoint:
    def test_returns_json(self, client):
        rv = client.get("/api/stats")
        assert rv.status_code == 200
        assert rv.content_type.startswith("application/json")

    def test_has_required_fields(self, client):
        data = client.get("/api/stats").get_json()
        for key in ("total_videos", "avg_density", "platforms"):
            assert key in data, f"Missing key: {key}"

    def test_correct_video_count(self, client):
        data = client.get("/api/stats").get_json()
        assert data["total_videos"] == len(MOCK_METADATA)


# ── TestIndexRoute ────────────────────────────────────────────────────────

class TestIndexRoute:
    def test_returns_200(self, client):
        rv = client.get("/")
        assert rv.status_code == 200

    def test_contains_search_form(self, client):
        html = client.get("/").data.decode()
        assert "searchForm" in html or "<form" in html


# ── TestAppStartup ────────────────────────────────────────────────────────

class TestAppStartup:
    def test_globals_populated(self):
        assert app_module.model is not None
        assert app_module.embeddings is not None
        assert app_module.density_scores is not None
        assert app_module.metadata is not None

    def test_metadata_length_matches_embeddings(self):
        assert len(app_module.metadata) == app_module.embeddings.shape[0]

    def test_model_callable(self):
        out = app_module.model.encode(["hello"], convert_to_numpy=True)
        assert out.shape == (1, 384)


# ── TestErrorHandling ─────────────────────────────────────────────────────

class TestErrorHandling:
    def test_missing_query_returns_400(self, client):
        rv = client.get("/api/search")
        assert rv.status_code == 400

    def test_empty_query_returns_400(self, client):
        rv = client.get("/api/search?q=")
        assert rv.status_code == 400
        data = rv.get_json()
        assert "error" in data
