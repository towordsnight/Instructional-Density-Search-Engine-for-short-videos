#!/usr/bin/env python3
"""
Transcript cleaning pipeline for short-form video transcripts.

Raw transcripts from YouTube captions and Whisper STT contain noise (filler
words, [Music] tags, repeated words, timestamps, etc.) that degrades embedding
quality and search accuracy. This module provides a composable cleaning
pipeline that runs before embeddings are generated.

No external dependencies — stdlib only (re, unicodedata, json, argparse).
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# Step 1: Unicode normalization
# ---------------------------------------------------------------------------

def _normalize_unicode(text: str) -> str:
    """NFC normalization, remove zero-width chars, replace non-breaking spaces."""
    text = unicodedata.normalize("NFC", text)
    # Remove zero-width characters (U+200B..U+200F, U+FEFF)
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    # Replace non-breaking spaces with regular spaces
    text = text.replace("\u00a0", " ")
    return text


# ---------------------------------------------------------------------------
# Step 2: Remove caption artifacts
# ---------------------------------------------------------------------------

# Tags commonly inserted by YouTube auto-captions and Whisper
_CAPTION_TAG_RE = re.compile(
    r"\[(?:Music|Applause|Laughter|Inaudible|Foreign|"
    r"music|applause|laughter|inaudible|foreign|"
    r"MUSIC|APPLAUSE|LAUGHTER|INAUDIBLE|FOREIGN)\]",
    re.IGNORECASE,
)


def _remove_caption_artifacts(text: str) -> str:
    """Strip [Music], [Applause], [Laughter], [Inaudible], [Foreign], etc."""
    return _CAPTION_TAG_RE.sub("", text)


# ---------------------------------------------------------------------------
# Step 3: Remove timestamps
# ---------------------------------------------------------------------------

# Matches patterns like: 0:15, 1:23:45, (00:15), [1:23]
_TIMESTAMP_RE = re.compile(
    r"[\(\[]?\d{1,2}:\d{2}(?::\d{2})?[\)\]]?"
)


def _remove_timestamps(text: str) -> str:
    """Strip timestamp patterns like 0:15, 1:23:45, (00:15)."""
    return _TIMESTAMP_RE.sub("", text)


# ---------------------------------------------------------------------------
# Step 4: Remove URLs
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+")


def _remove_urls(text: str) -> str:
    """Strip http:// and https:// URLs."""
    return _URL_RE.sub("", text)


# ---------------------------------------------------------------------------
# Step 5: Remove @mentions and #hashtags
# ---------------------------------------------------------------------------

_MENTION_HASHTAG_RE = re.compile(r"[@#]\w+")


def _remove_mentions_hashtags(text: str) -> str:
    """Strip @user mentions and #hashtags."""
    return _MENTION_HASHTAG_RE.sub("", text)


# ---------------------------------------------------------------------------
# Step 6: Remove filler words
# ---------------------------------------------------------------------------

# Conservative filler list — NOT including "like", "so", "okay" (too ambiguous)
_FILLER_SINGLE_RE = re.compile(
    r"\b(?:uh|um|hmm|er|ah)\b",
    re.IGNORECASE,
)
_FILLER_PHRASE_RE = re.compile(
    r"\b(?:you know|i mean)\b",
    re.IGNORECASE,
)


def _remove_fillers(text: str) -> str:
    """Remove filler words: uh, um, hmm, er, ah, you know, i mean."""
    text = _FILLER_PHRASE_RE.sub("", text)
    text = _FILLER_SINGLE_RE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Step 7: Collapse repeated words
# ---------------------------------------------------------------------------

_REPEATED_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1)+\b", re.IGNORECASE)


def _collapse_repeated_words(text: str) -> str:
    """Collapse repeated words: 'the the the' -> 'the'."""
    return _REPEATED_WORD_RE.sub(r"\1", text)


# ---------------------------------------------------------------------------
# Step 8: Normalize whitespace
# ---------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    """Collapse multi-spaces, fix space around punctuation."""
    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)
    # Remove space before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    # Ensure space after punctuation (if followed by a word character)
    text = re.sub(r"([,.!?;:])(\w)", r"\1 \2", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Step 9: Sentence segmentation for run-on text
# ---------------------------------------------------------------------------

# Discourse markers that often begin a new sentence in spoken transcripts
_DISCOURSE_MARKERS = (
    "So", "First", "Then", "Next", "After that", "Finally",
    "Now", "Also", "But", "And then", "However", "Actually",
    "Basically", "Meanwhile", "Instead", "Anyway",
)

_DISCOURSE_RE = re.compile(
    r"(?<=[a-z]) (" + "|".join(re.escape(m) for m in _DISCOURSE_MARKERS) + r")\b"
)


def _segment_sentences(text: str) -> str:
    """Insert periods before discourse markers in run-on text that lacks punctuation."""
    return _DISCOURSE_RE.sub(r". \1", text)


# ---------------------------------------------------------------------------
# Pipeline registry and public API
# ---------------------------------------------------------------------------

STEPS = {
    "normalize_unicode": _normalize_unicode,
    "remove_caption_artifacts": _remove_caption_artifacts,
    "remove_timestamps": _remove_timestamps,
    "remove_urls": _remove_urls,
    "remove_mentions_hashtags": _remove_mentions_hashtags,
    "remove_fillers": _remove_fillers,
    "collapse_repeated_words": _collapse_repeated_words,
    "normalize_whitespace": _normalize_whitespace,
    "segment_sentences": _segment_sentences,
}

# Default step order
DEFAULT_STEP_ORDER = [
    "normalize_unicode",
    "remove_caption_artifacts",
    "remove_timestamps",
    "remove_urls",
    "remove_mentions_hashtags",
    "remove_fillers",
    "collapse_repeated_words",
    "normalize_whitespace",
    "segment_sentences",
]


def clean_transcript(text: str, steps: list[str] | None = None) -> str:
    """
    Clean a single transcript through the processing pipeline.

    Args:
        text: Raw transcript text.
        steps: Optional list of step names to apply (in order).
               If None, all steps are applied in default order.

    Returns:
        Cleaned transcript string.
    """
    if not text or not text.strip():
        return ""

    if steps is None:
        steps = DEFAULT_STEP_ORDER

    for step_name in steps:
        fn = STEPS.get(step_name)
        if fn is None:
            raise ValueError(f"Unknown cleaning step: {step_name!r}")
        text = fn(text)

    return text


def clean_dataset(input_path: str, output_path: str) -> dict:
    """
    Batch-clean all transcripts in a JSON dataset file.

    Args:
        input_path: Path to input JSON file (list of video dicts).
        output_path: Path to write cleaned JSON file.

    Returns:
        Summary dict with keys: total, cleaned, skipped, empty_after_clean.
    """
    with open(input_path) as f:
        dataset = json.load(f)

    stats = {"total": len(dataset), "cleaned": 0, "skipped": 0, "empty_after_clean": 0}

    for item in dataset:
        transcript = item.get("transcript")
        if not transcript or not str(transcript).strip():
            stats["skipped"] += 1
            continue

        cleaned = clean_transcript(str(transcript))
        item["transcript"] = cleaned
        stats["cleaned"] += 1

        if not cleaned:
            stats["empty_after_clean"] += 1

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    return stats


# ---------------------------------------------------------------------------
# CLI entry point: python -m text_processing.clean_transcript
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clean transcripts in a short-form video dataset."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input JSON dataset file",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to write cleaned JSON dataset file",
    )
    args = parser.parse_args()

    print(f"Cleaning transcripts: {args.input} -> {args.output}")
    stats = clean_dataset(args.input, args.output)

    print(f"\n=== Cleaning Summary ===")
    print(f"  Total entries:      {stats['total']}")
    print(f"  Cleaned:            {stats['cleaned']}")
    print(f"  Skipped (no text):  {stats['skipped']}")
    print(f"  Empty after clean:  {stats['empty_after_clean']}")
    print(f"  Output saved to:    {args.output}")


if __name__ == "__main__":
    main()
