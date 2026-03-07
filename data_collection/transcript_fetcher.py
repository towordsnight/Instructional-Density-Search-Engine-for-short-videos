"""Transcript extraction via YouTube captions API and Whisper speech-to-text."""

import json
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi

_yt_api = YouTubeTranscriptApi()

# Cache directory for storing fetched transcripts
_CACHE_DIR = Path(__file__).resolve().parent.parent / "dataset" / "transcript_cache"
_CACHE_FILE = _CACHE_DIR / "transcripts.json"


def _load_cache() -> dict[str, str]:
    """Load transcript cache from disk."""
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    """Save transcript cache to disk."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def get_cached_transcript(video_id: str) -> str | None:
    """Return cached transcript if available."""
    cache = _load_cache()
    return cache.get(video_id)


def cache_transcript(video_id: str, transcript: str) -> None:
    """Store a transcript in the cache."""
    cache = _load_cache()
    cache[video_id] = transcript
    _save_cache(cache)


def fetch_youtube_transcript(video_id: str) -> str | None:
    """Fetch the transcript for a YouTube video using its captions.

    Checks the local cache first. On success, caches the result.
    Returns the full transcript as a single string, or None if unavailable.
    """
    cached = get_cached_transcript(video_id)
    if cached:
        return cached

    try:
        result = _yt_api.fetch(video_id)
        transcript = " ".join(snippet.text for snippet in result)
        transcript = transcript.strip() if transcript.strip() else None
        if transcript:
            cache_transcript(video_id, transcript)
        return transcript
    except Exception:
        return None


def transcribe_with_whisper(audio_path: str, model_name: str = "base") -> str | None:
    """Transcribe an audio file using OpenAI Whisper.

    Args:
        audio_path: Path to the audio file.
        model_name: Whisper model size ("tiny", "base", "small", "medium", "large").

    Returns the transcript text, or None on failure.
    """
    try:
        import whisper

        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path)
        text = result.get("text", "").strip()
        return text if text else None
    except Exception:
        return None
