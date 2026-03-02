#!/usr/bin/env python3
"""
Instructional density scoring for short videos.
Scores content based on weighted instructional signals and entertainment penalties.
Higher score = more instructional/educational content.
"""

import math
import re
from typing import Sequence


# ---------------------------------------------------------------------------
# Weighted signal categories — higher weight = stronger instructional signal
# ---------------------------------------------------------------------------
SIGNAL_CATEGORIES = {
    "how_to_cues": {
        "weight": 3.0,
        "patterns": {
            "how_to": r"\bhow\s+to\b",
            "tutorial": r"\btutorial\b",
            "guide": r"\bguide\b",
            "step": r"\bstep\s+\d+\b|\bstep\s+by\s+step\b",
            "let_me_show": r"\blet\s+me\s+show\b",
            "in_this_video": r"\bin\s+this\s+video\b",
            "in_order_to": r"\bin\s+order\s+to\b",
            "you_need": r"\byou\s+need\b",
            "to_make": r"\bto\s+make\b",
        },
    },
    "educational_concepts": {
        "weight": 2.0,
        "patterns": {
            "learn": r"\blearn\b",
            "teach": r"\bteach(?:ing|es)?\b",
            "tips": r"\btips?\b",
            "tricks": r"\btricks?\b",
            "recipe": r"\brecipe\b",
            "workout": r"\bworkout\b",
            "exercise": r"\bexercise\b",
        },
    },
    "domain_knowledge": {
        "weight": 2.0,
        "patterns": {
            "design": r"\bdesign\b",
            "inspiration": r"\binspiration\b",
            "inspire": r"\binspire[d]?\b",
            "aesthetic": r"\baesthetic\b",
            "style": r"\bstyle\b",
            "collection": r"\bcollection\b",
            "creative": r"\bcreative\b",
            "art": r"\bart\b",
            "craftsmanship": r"\bcraftsmanship\b",
            "architecture": r"\barchitect(?:ure|ural)\b",
            "meaning": r"\bmeaning\b",
            "significance": r"\bsignificance\b",
            "symbolism": r"\bsymbolism\b",
            "symbol": r"\bsymbol\b",
            "interpret": r"\binterpret(?:ation|ed)?\b",
            "represent": r"\brepresent(?:s|ed)?\b",
        },
    },
    "historical_context": {
        "weight": 1.5,
        "patterns": {
            "history": r"\bhistory\b",
            "historical": r"\bhistorical\b",
            "heritage": r"\bheritage\b",
            "legacy": r"\blegacy\b",
            "origin": r"\borigin\b",
            "story": r"\bstory\b",
            "founded": r"\bfounded\b",
            "since": r"\bsince\s+\d{4}\b",
            "discover": r"\bdiscover(?:y|ed)?\b",
            "explore": r"\bexplore(?:d|s)?\b",
        },
    },
    "sequential_markers": {
        "weight": 1.5,
        "patterns": {
            "first": r"\bfirst\b",
            "then": r"\bthen\b",
            "next": r"\bnext\b",
            "after": r"\bafter\s+(that|this)\b",
            "finally": r"\bfinally\b",
            "before": r"\bbefore\b",
            "now": r"\bnow\b",
        },
    },
    "action_verbs": {
        "weight": 1.0,
        "patterns": {
            "make": r"\bmake\b",
            "add": r"\badd\b",
            "mix": r"\bmix\b",
            "cut": r"\bcut\b",
            "put": r"\bput\b",
            "take": r"\btake\b",
            "start": r"\bstart\b",
            "begin": r"\bbegin\b",
            "try": r"\btry\b",
            "use": r"\buse\b",
            "follow": r"\bfollow\b",
            "watch": r"\bwatch\b",
            "do": r"\bdo\b",
            "create": r"\bcreate\b",
            "build": r"\bbuild\b",
        },
    },
}

# Backward-compatible flat dict: all 62 patterns (name → regex)
INSTRUCTIONAL_SIGNALS = {}
for _cat in SIGNAL_CATEGORIES.values():
    INSTRUCTIONAL_SIGNALS.update(_cat["patterns"])


# ---------------------------------------------------------------------------
# Entertainment signals — penalty for non-instructional content
# ---------------------------------------------------------------------------
ENTERTAINMENT_SIGNALS = {
    "omg": r"\bomg\b",
    "bestie": r"\bbesties?\b",
    "haul": r"\bhaul\b",
    "unboxing": r"\bunboxing\b",
    "slay": r"\bslay\b",
    "vibe": r"\bvibes?\b",
    "obsessed": r"\bobsessed\b",
    "no_cap": r"\bno\s+cap\b",
    "cute": r"\bcute\b",
    "bro": r"\bbro\b",
    "spree": r"\bspree\b",
    "went_crazy": r"\bwent\s+crazy\b",
    "so_pretty": r"\bso\s+pretty\b",
    "is_everything": r"\bis\s+everything\b",
    "you_guys": r"\byou\s+guys\b",
    "im_dead": r"\bi'?m\s+dead\b",
    "im_crying": r"\bi'?m\s+crying\b",
    "its_giving": r"\bit'?s\s+giving\b",
    "lit": r"\blit\b",
    "fire": r"\bfire\b",
}

ENTERTAINMENT_PENALTY = 2.0


def compute_instructional_score(text: str) -> float:
    """
    Compute instructional density score for a single text (0.0 to 1.0).
    Higher = more instructional/educational content.

    Uses weighted categories, entertainment penalty, and length normalization.
    """
    if not text or not str(text).strip():
        return 0.0

    text = str(text).lower()
    word_count = len(text.split())
    if word_count == 0:
        return 0.0

    # Weighted instructional signal sum
    weighted_sum = 0.0
    for cat in SIGNAL_CATEGORIES.values():
        weight = cat["weight"]
        for pattern in cat["patterns"].values():
            matches = re.findall(pattern, text, re.IGNORECASE)
            weighted_sum += len(matches) * weight

    # Entertainment penalty
    entertainment_count = 0
    for pattern in ENTERTAINMENT_SIGNALS.values():
        entertainment_count += len(re.findall(pattern, text, re.IGNORECASE))

    penalty = entertainment_count * ENTERTAINMENT_PENALTY
    net = max(0.0, weighted_sum - penalty)

    # Length-normalized density (signals per 100 words)
    density = (net / word_count) * 100
    raw = min(density, 20.0)
    score = min(1.0, math.sqrt(raw) / 4.5)
    return round(score, 4)


def compute_instructional_scores(texts: Sequence[str]) -> list[float]:
    """Compute instructional scores for a list of texts."""
    return [compute_instructional_score(t) for t in texts]
