"""Manual URL processor for TikTok and Instagram videos using yt-dlp + Whisper."""

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

from data_collection.transcript_fetcher import transcribe_with_whisper


def _generate_id(url: str, platform: str) -> str:
    """Generate a stable short ID from a URL."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    prefix = platform[:2]  # "ti" for tiktok, "ig" for instagram
    return f"{prefix}_{url_hash}"


def _download_metadata_and_audio(url: str, audio_dir: str) -> dict | None:
    """Use yt-dlp to extract metadata and download audio for a single URL.

    Returns a dict with title, uploader, view_count, audio_path, or None on failure.
    """
    try:
        import yt_dlp

        # First, extract metadata without downloading
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(url, download=False)

        title = info.get("title", "Untitled")
        uploader = info.get("uploader", "Unknown")
        view_count = info.get("view_count", 0)

        # Download audio only
        audio_path = os.path.join(audio_dir, f"{info.get('id', 'audio')}.mp3")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_path,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # yt-dlp may append .mp3 to the path
        if not os.path.exists(audio_path) and os.path.exists(audio_path + ".mp3"):
            audio_path = audio_path + ".mp3"

        return {
            "title": title,
            "uploader": uploader,
            "view_count": view_count or 0,
            "audio_path": audio_path,
        }
    except Exception as e:
        print(f"  [WARNING] Failed to process {url}: {e}")
        return None


def load_manual_urls(filepath: str) -> list[dict]:
    """Load manual URLs from a CSV file.

    Expected columns: url, platform
    Returns a list of dicts with url and platform keys.
    """
    filepath = Path(filepath)
    if filepath.suffix == ".json":
        with open(filepath) as f:
            return json.load(f)

    urls = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("url", "").strip()
            platform = row.get("platform", "").strip().lower()
            if url and platform:
                urls.append({"url": url, "platform": platform})
    return urls


def process_manual_urls(filepath: str, whisper_model: str = "base") -> list[dict]:
    """Process all manual URLs: download audio, transcribe with Whisper.

    Returns a list of dicts matching the standard data format:
    {id, platform, title, transcript, url, channel, views, likes, published_at}
    """
    urls = load_manual_urls(filepath)
    if not urls:
        print("  No manual URLs found.")
        return []

    results = []
    with tempfile.TemporaryDirectory(prefix="shortcomp_audio_") as audio_dir:
        for entry in urls:
            url = entry["url"]
            platform = entry["platform"]
            print(f"  Processing [{platform}]: {url}")

            meta = _download_metadata_and_audio(url, audio_dir)
            if meta is None:
                continue

            # Transcribe audio with Whisper
            transcript = transcribe_with_whisper(meta["audio_path"], whisper_model)
            if transcript is None:
                print(f"  [WARNING] No transcript generated for {url}, skipping.")
                continue

            results.append({
                "id": _generate_id(url, platform),
                "platform": platform,
                "title": meta["title"],
                "transcript": transcript,
                "url": url,
                "channel": meta["uploader"],
                "views": meta["view_count"],
                "likes": 0,
                "published_at": "",
            })

    return results
