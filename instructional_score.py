#!/usr/bin/env python3
"""
Instructional density scoring for short videos.
Scores content based on instructional signals (how-to cues, sequential markers, etc.).
Higher score = more instructional/educational content.
"""

import math
import re
from typing import Sequence


# Instructional & informational signal patterns
# Covers: how-to, design inspiration, architecture, history, educational content
INSTRUCTIONAL_SIGNALS = {
    # How-to / tutorial cues
    "how_to": r"\bhow\s+to\b",
    "tutorial": r"\btutorial\b",
    "guide": r"\bguide\b",
    "learn": r"\blearn\b",
    "teach": r"\bteach(?:ing|es)?\b",
    "step": r"\bstep\s+\d+\b|\bstep\s+by\s+step\b",
    "tips": r"\btips?\b",
    "tricks": r"\btricks?\b",
    # Design / inspiration / aesthetic (for "Design inspiration of Tiffany collections")
    "design": r"\bdesign\b",
    "inspiration": r"\binspiration\b",
    "inspire": r"\binspire[d]?\b",
    "aesthetic": r"\baesthetic\b",
    "style": r"\bstyle\b",
    "collection": r"\bcollection\b",
    "creative": r"\bcreative\b",
    "art": r"\bart\b",
    "craftsmanship": r"\bcraftsmanship\b",
    # Architecture / meaning (for "architectural meaning of Tiffany collections")
    "architecture": r"\barchitect(?:ure|ural)\b",
    "meaning": r"\bmeaning\b",
    "significance": r"\bsignificance\b",
    "symbolism": r"\bsymbolism\b",
    "symbol": r"\bsymbol\b",
    "interpret": r"\binterpret(?:ation|ed)?\b",
    "represent": r"\brepresent(?:s|ed)?\b",
    # History / heritage (for "history of Tiffany")
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
    # Sequential / procedural markers
    "first": r"\bfirst\b",
    "then": r"\bthen\b",
    "next": r"\bnext\b",
    "after": r"\bafter\s+(that|this)\b",
    "finally": r"\bfinally\b",
    "before": r"\bbefore\b",
    "now": r"\bnow\b",
    # Action verbs (imperative / instructional)
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
    # Instructional phrases
    "in_this_video": r"\bin\s+this\s+video\b",
    "let_me_show": r"\blet\s+me\s+show\b",
    "in_order_to": r"\bin\s+order\s+to\b",
    "you_need": r"\byou\s+need\b",
    "to_make": r"\bto\s+make\b",
    "recipe": r"\brecipe\b",
    "workout": r"\bworkout\b",
    "exercise": r"\bexercise\b",
}


def compute_instructional_score(text: str) -> float:
    """
    Compute instructional density score for a single text (0.0 to 1.0).
    Higher = more instructional/educational content.
    """
    if not text or not str(text).strip():
        return 0.0

    text = str(text).lower()
    total_signals = 0

    for name, pattern in INSTRUCTIONAL_SIGNALS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        total_signals += len(matches)

    # Normalize: cap at ~15 signals for score 1.0, use sqrt for diminishing returns
    raw = min(total_signals, 20)
    score = min(1.0, math.sqrt(raw) / 4.5)  # sqrt(20)/4.5 ≈ 0.99
    return round(score, 4)


def compute_instructional_scores(texts: Sequence[str]) -> list[float]:
    """Compute instructional scores for a list of texts."""
    return [compute_instructional_score(t) for t in texts]
