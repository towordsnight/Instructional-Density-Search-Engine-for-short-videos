"""Transcript extraction via YouTube captions API and Whisper speech-to-text."""

from youtube_transcript_api import YouTubeTranscriptApi

_yt_api = YouTubeTranscriptApi()


def fetch_youtube_transcript(video_id: str) -> str | None:
    """Fetch the transcript for a YouTube video using its captions.

    Returns the full transcript as a single string, or None if unavailable.
    """
    try:
        result = _yt_api.fetch(video_id)
        transcript = " ".join(snippet.text for snippet in result)
        return transcript.strip() if transcript.strip() else None
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
