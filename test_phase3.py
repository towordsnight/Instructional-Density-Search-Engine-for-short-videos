#!/usr/bin/env python3
"""
Phase 3 test suite: weighted scoring, entertainment penalties, length normalization,
query expansion, deduplication, and video URL construction.
"""

import math
import numpy as np
import pytest

from instructional_score import (
    SIGNAL_CATEGORIES,
    INSTRUCTIONAL_SIGNALS,
    ENTERTAINMENT_SIGNALS,
    ENTERTAINMENT_PENALTY,
    compute_instructional_score,
    compute_instructional_scores,
)
from search import (
    _SYNONYMS,
    _expand_query_terms,
    _extract_query_terms,
    _query_term_boost,
    _deduplicate_results,
    _construct_video_url,
)


# ============================================================
# TestWeightedCategories (5 tests)
# ============================================================
class TestWeightedCategories:
    """Verify weighted category structure and scoring behavior."""

    def test_how_to_scores_higher_than_action_verbs(self):
        """how-to cues (weight 3.0) should score higher than action verbs (weight 1.0)."""
        # Pad with filler so density doesn't saturate for both
        filler = " the quick brown fox jumps over the lazy dog" * 5
        how_to_text = "how to tutorial guide" + filler
        action_text = "make add mix cut" + filler
        assert compute_instructional_score(how_to_text) > compute_instructional_score(action_text)

    def test_category_structure(self):
        """Each category must have weight and patterns."""
        for name, cat in SIGNAL_CATEGORIES.items():
            assert "weight" in cat, f"Missing weight in {name}"
            assert "patterns" in cat, f"Missing patterns in {name}"
            assert isinstance(cat["weight"], (int, float))
            assert isinstance(cat["patterns"], dict)
            assert len(cat["patterns"]) > 0, f"Empty patterns in {name}"

    def test_all_62_patterns_preserved(self):
        """All original 62 patterns must still exist in the flat dict."""
        total = sum(len(cat["patterns"]) for cat in SIGNAL_CATEGORIES.values())
        assert total == len(INSTRUCTIONAL_SIGNALS)
        # Spot-check a few known patterns
        assert "how_to" in INSTRUCTIONAL_SIGNALS
        assert "design" in INSTRUCTIONAL_SIGNALS
        assert "history" in INSTRUCTIONAL_SIGNALS
        assert "make" in INSTRUCTIONAL_SIGNALS

    def test_backward_compat_flat_dict(self):
        """INSTRUCTIONAL_SIGNALS flat dict is derived from categories."""
        for cat in SIGNAL_CATEGORIES.values():
            for name, pattern in cat["patterns"].items():
                assert name in INSTRUCTIONAL_SIGNALS
                assert INSTRUCTIONAL_SIGNALS[name] == pattern

    def test_tutorial_scores_higher_than_generic(self):
        """A tutorial text should score higher than a text with only generic verbs."""
        filler = " the quick brown fox jumps over the lazy dog" * 5
        tutorial = "In this video we learn how to build a guide step by step tutorial" + filler
        generic = "try do use make start begin take put" + filler
        assert compute_instructional_score(tutorial) > compute_instructional_score(generic)


# ============================================================
# TestEntertainmentSignals (5 tests)
# ============================================================
class TestEntertainmentSignals:
    """Verify entertainment signal detection and penalty."""

    def test_patterns_match(self):
        """Each entertainment pattern should match its target word."""
        import re
        test_strings = {
            "omg": "omg this is amazing",
            "bestie": "hey besties",
            "haul": "shopping haul",
            "unboxing": "unboxing video",
            "slay": "you slay",
            "vibe": "love the vibe",
            "went_crazy": "i went crazy",
            "you_guys": "you guys look",
            "its_giving": "it's giving luxury",
        }
        for name, text in test_strings.items():
            pattern = ENTERTAINMENT_SIGNALS[name]
            assert re.search(pattern, text, re.IGNORECASE), f"Pattern {name} didn't match '{text}'"

    def test_reduces_score(self):
        """Adding entertainment words should reduce the score."""
        filler = " the quick brown fox jumps over the lazy dog" * 5
        base = "In this video we learn how to design step by step tutorial" + filler
        with_ent = "In this video we learn how to design step by step tutorial omg bestie haul slay vibe" + filler
        assert compute_instructional_score(base) > compute_instructional_score(with_ent)

    def test_pure_entertainment_zero(self):
        """Text with only entertainment signals and no instructional content → 0.0."""
        text = "omg bestie haul unboxing slay vibe obsessed cute bro spree lit fire"
        assert compute_instructional_score(text) == 0.0

    def test_mixed_content(self):
        """Mixed content scores between pure instructional and pure entertainment."""
        filler = " the quick brown fox jumps over the lazy dog" * 8
        instructional = "how to learn design step by step tutorial guide meaning" + filler
        entertainment = "omg bestie haul slay vibe cute bro lit fire" + filler
        # More instructional than entertainment so net > 0
        mixed = "how to learn design step by step tutorial guide meaning omg bestie" + filler
        score_inst = compute_instructional_score(instructional)
        score_ent = compute_instructional_score(entertainment)
        score_mixed = compute_instructional_score(mixed)
        assert score_inst > score_mixed
        assert score_mixed > score_ent

    def test_no_negative_scores(self):
        """Score should never go below 0.0 even with many entertainment signals."""
        text = "omg omg omg bestie bestie haul haul slay slay vibe vibe obsessed cute bro lit fire"
        score = compute_instructional_score(text)
        assert score >= 0.0


# ============================================================
# TestLengthNormalization (4 tests)
# ============================================================
class TestLengthNormalization:
    """Verify length normalization produces density-based scores."""

    def test_short_text_scores(self):
        """Short text with high signal density should score well."""
        text = "how to learn design step by step"
        score = compute_instructional_score(text)
        assert score > 0.5

    def test_diluted_text_lower(self):
        """Same signals diluted with filler words should score lower."""
        concentrated = "how to learn design step by step tutorial"
        diluted = "how to " + "word " * 200 + "learn design step by step tutorial"
        assert compute_instructional_score(concentrated) > compute_instructional_score(diluted)

    def test_empty_whitespace_zero(self):
        """Empty and whitespace-only texts score 0.0."""
        assert compute_instructional_score("") == 0.0
        assert compute_instructional_score("   ") == 0.0
        assert compute_instructional_score(None) == 0.0

    def test_same_density_similar_scores(self):
        """Texts with same signal density should have similar scores regardless of length."""
        short = "how to learn design step by step tutorial guide"
        # Double the text — same density
        long = short + " " + short
        score_short = compute_instructional_score(short)
        score_long = compute_instructional_score(long)
        # Should be similar (within 0.15 tolerance due to normalization curve)
        assert abs(score_short - score_long) < 0.15


# ============================================================
# TestScorerBackwardCompat (3 tests)
# ============================================================
class TestScorerBackwardCompat:
    """Verify backward-compatible API."""

    def test_api_signature(self):
        """compute_instructional_score takes str, returns float."""
        result = compute_instructional_score("test text")
        assert isinstance(result, float)

    def test_batch_api(self):
        """compute_instructional_scores takes list, returns list of floats."""
        results = compute_instructional_scores(["text one", "text two", "text three"])
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, float) for r in results)

    def test_score_range(self):
        """All scores must be in [0.0, 1.0]."""
        texts = [
            "",
            "random words",
            "how to learn design step by step tutorial guide meaning symbolism history heritage",
            "omg bestie haul slay vibe",
            "a " * 1000,
        ]
        for text in texts:
            score = compute_instructional_score(text)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for: {text[:50]}"


# ============================================================
# TestQueryExpansion (4 tests)
# ============================================================
class TestQueryExpansion:
    """Verify query expansion with synonyms."""

    def test_known_term_expands(self):
        """'meaning' should expand to include synonyms."""
        terms = {"meaning"}
        expanded = _expand_query_terms(terms)
        assert "significance" in expanded
        assert "symbolism" in expanded
        assert "interpretation" in expanded
        assert "meaning" in expanded  # original preserved

    def test_unknown_passes_through(self):
        """Unknown terms are kept as-is, no expansion."""
        terms = {"tiffany", "blue"}
        expanded = _expand_query_terms(terms)
        assert expanded == {"tiffany", "blue"}

    def test_expand_flag_works(self):
        """_extract_query_terms with expand=True returns expanded terms."""
        terms_no_expand = _extract_query_terms("meaning of design", expand=False)
        terms_expanded = _extract_query_terms("meaning of design", expand=True)
        assert len(terms_expanded) > len(terms_no_expand)
        assert "symbolism" in terms_expanded
        assert "aesthetic" in terms_expanded

    def test_boost_with_synonyms(self):
        """Boost should be higher when doc matches expanded synonyms."""
        query = "meaning of design"
        doc_with_synonyms = "the significance and symbolism of the aesthetic style"
        doc_without = "the blue color of the product"
        boost_syn = _query_term_boost(query, doc_with_synonyms, expand=True)
        boost_none = _query_term_boost(query, doc_without, expand=True)
        assert boost_syn > boost_none


# ============================================================
# TestDeduplication (4 tests)
# ============================================================
class TestDeduplication:
    """Verify near-duplicate removal logic."""

    def test_identical_removed(self):
        """Identical embeddings should be deduplicated to one."""
        emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        indices = np.array([0, 1, 2])
        result = _deduplicate_results(indices, emb, threshold=0.95)
        assert len(result) == 2
        assert 0 in result
        assert 2 in result

    def test_different_preserved(self):
        """Distinct embeddings should all be preserved."""
        emb = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        indices = np.array([0, 1, 2])
        result = _deduplicate_results(indices, emb, threshold=0.95)
        assert len(result) == 3

    def test_threshold_boundary(self):
        """Documents right at threshold boundary should be removed."""
        # Two vectors with cosine similarity = 1.0 (identical direction)
        emb = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
        indices = np.array([0, 1, 2])
        result = _deduplicate_results(indices, emb, threshold=0.99)
        assert len(result) == 2  # [1,0,0] and [2,0,0] are parallel → deduped

    def test_disabled_at_one(self):
        """threshold=1.0 should disable deduplication."""
        emb = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        indices = np.array([0, 1, 2])
        result = _deduplicate_results(indices, emb, threshold=1.0)
        assert len(result) == 3


# ============================================================
# TestVideoUrls (3 tests)
# ============================================================
class TestVideoUrls:
    """Verify video URL construction."""

    def test_url_from_metadata(self):
        """Existing url field should be preferred (tested via search integration)."""
        meta = {"platform": "youtube", "id": "abc123", "url": "https://example.com/video"}
        # _construct_video_url is only called when url is missing, but check it works
        url = _construct_video_url(meta)
        assert url == "https://www.youtube.com/shorts/abc123"

    def test_url_constructed_for_youtube(self):
        """YouTube URL constructed from platform + id."""
        meta = {"platform": "youtube", "id": "abc123"}
        url = _construct_video_url(meta)
        assert url == "https://www.youtube.com/shorts/abc123"

        meta_tiktok = {"platform": "tiktok", "id": "12345"}
        url_tt = _construct_video_url(meta_tiktok)
        assert "tiktok.com" in url_tt
        assert "12345" in url_tt

        meta_ig = {"platform": "instagram", "id": "reel99"}
        url_ig = _construct_video_url(meta_ig)
        assert "instagram.com" in url_ig

    def test_empty_when_unknown(self):
        """Unknown platform returns empty string."""
        meta = {"platform": "unknown", "id": "xyz"}
        assert _construct_video_url(meta) == ""

        meta_empty = {}
        assert _construct_video_url(meta_empty) == ""


# ============================================================
# TestRankingPreservation (2 tests)
# ============================================================
class TestRankingPreservation:
    """Verify that Phase 3 changes preserve and improve ranking quality."""

    # Same test data as test_ranking.py
    TEST_DATA = [
        {
            "id": "inst_t", "platform": "youtube",
            "title": "The architectural meaning of Tiffany T collection - design and symbolism",
            "transcript": "In this video we explore the architectural meaning and significance of the Tiffany T collection. The design represents strength and modernity. Learn the symbolism behind each piece. The collection's aesthetic draws from urban architecture. Discover the creative inspiration and craftsmanship.",
        },
        {
            "id": "inst_knot", "platform": "youtube",
            "title": "Tiffany Knot collection - meaning and heritage explained",
            "transcript": "The Knot collection has deep architectural significance. In this guide we interpret the meaning of the knot symbol. The design represents connection and legacy. First, we explore the history. Then, discover the craftsmanship. The collection's style blends tradition with modern aesthetic.",
        },
        {
            "id": "inst_lock", "platform": "youtube",
            "title": "Understanding the meaning of Tiffany Lock collection",
            "transcript": "Discover the architectural meaning of the Lock collection. The design inspiration comes from security and heritage. Learn the significance of each element. This tutorial explores the symbolism and creative vision. The collection represents timeless craftsmanship and style.",
        },
        {
            "id": "vlog_1", "platform": "tiktok",
            "title": "LUXURY SHOPPING HAUL at Tiffany - spent way too much",
            "transcript": "OMG you guys I just went crazy at Tiffany. Look at all this stuff. So pretty. I don't even know what half of it means. Just had to have it. The bag is everything. That's the haul bye.",
        },
        {
            "id": "vlog_2", "platform": "instagram",
            "title": "Tiffany unboxing - my biggest shopping spree ever",
            "transcript": "Hey besties! So I went to Tiffany and went a little crazy. Here's what I got. This one is cute. This one too. No idea what the design means honestly. Just love the vibe. Enjoy the unboxing!",
        },
    ]

    def test_instructional_ranks_top_3(self):
        """Instructional texts should have higher density than vlogs."""
        texts = [f"{d['title']} [SEP] {d['transcript']}" for d in self.TEST_DATA]
        scores = compute_instructional_scores(texts)

        inst_scores = scores[:3]  # inst_t, inst_knot, inst_lock
        vlog_scores = scores[3:]  # vlog_1, vlog_2

        # Every instructional score should be higher than every vlog score
        assert min(inst_scores) > max(vlog_scores), (
            f"Instructional scores {inst_scores} should all exceed vlog scores {vlog_scores}"
        )

    def test_vlog_density_low(self):
        """Vlog density should be very low due to entertainment penalties."""
        texts = [f"{d['title']} [SEP] {d['transcript']}" for d in self.TEST_DATA]
        scores = compute_instructional_scores(texts)

        vlog_scores = scores[3:]
        for score in vlog_scores:
            assert score < 0.3, f"Vlog score {score} should be < 0.3 with entertainment penalties"
