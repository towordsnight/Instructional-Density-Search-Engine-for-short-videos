"""Orchestrator script: collects videos from all sources and builds the final dataset."""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Add parent directory to path so we can import data_collection as a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_collection.youtube_api import SEARCH_QUERIES, get_video_details, search_shorts
from data_collection.transcript_fetcher import (
    fetch_youtube_transcript,
    transcribe_with_whisper,
    cache_transcript,
)
from data_collection.tiktok_instagram_collector import process_manual_urls
from text_processing.clean_transcript import clean_transcript


def _whisper_fallback(video_id: str, whisper_model: str) -> str | None:
    """Download YouTube audio via yt-dlp and transcribe with Whisper."""
    try:
        import yt_dlp

        url = f"https://www.youtube.com/shorts/{video_id}"
        with tempfile.TemporaryDirectory(prefix="yt_whisper_") as tmp_dir:
            audio_path = os.path.join(tmp_dir, f"{video_id}.mp3")
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

            # yt-dlp may append .mp3
            if not os.path.exists(audio_path) and os.path.exists(audio_path + ".mp3"):
                audio_path = audio_path + ".mp3"

            if not os.path.exists(audio_path):
                return None

            transcript = transcribe_with_whisper(audio_path, whisper_model)
            if transcript:
                cache_transcript(video_id, transcript)
            return transcript
    except Exception as e:
        print(f"    [Whisper fallback failed for {video_id}: {e}]")
        return None


def collect_youtube(
    api_key: str,
    queries: list[str],
    results_per_query: int = 25,
    whisper_model: str = "base",
) -> list[dict]:
    """Search YouTube across multiple queries and fetch transcripts."""
    print(f"\n=== YouTube Collection ({len(queries)} queries, ~{results_per_query} each) ===")

    # Step 1: Search for video IDs across all queries
    all_video_ids = {}
    for query in queries:
        print(f"  Searching: '{query}'")
        results = search_shorts(api_key, query, max_results=results_per_query)
        for r in results:
            all_video_ids[r["video_id"]] = r
        print(f"    Found {len(results)} videos")

    print(f"  Total unique videos found: {len(all_video_ids)}")

    # Step 2: Fetch detailed metadata
    print("  Fetching video details...")
    video_ids = list(all_video_ids.keys())
    details = get_video_details(api_key, video_ids)
    details_map = {d["video_id"]: d for d in details}

    # Step 3: Fetch transcripts (caption API first, Whisper fallback)
    print("  Fetching transcripts...")
    dataset = []
    skipped = 0
    caption_count = 0
    whisper_count = 0
    for i, vid_id in enumerate(video_ids):
        detail = details_map.get(vid_id)
        if not detail:
            skipped += 1
            continue

        # Try caption API first (uses cache internally)
        transcript = fetch_youtube_transcript(vid_id)
        if transcript:
            caption_count += 1
        else:
            # Fallback: download audio and transcribe with Whisper
            print(f"    [{i+1}/{len(video_ids)}] No captions for {vid_id}, trying Whisper...")
            transcript = _whisper_fallback(vid_id, whisper_model)
            if transcript:
                whisper_count += 1

        if not transcript:
            skipped += 1
            continue

        dataset.append({
            "id": f"yt_{vid_id}",
            "platform": "youtube",
            "title": detail["title"],
            "transcript": transcript,
            "url": detail["url"],
            "channel": detail["channel"],
            "views": detail["views"],
            "likes": detail["likes"],
            "published_at": detail["published_at"],
            "thumbnail": detail.get("thumbnail", ""),
        })

        # Brief delay every 5 requests to avoid rate limiting
        if (i + 1) % 5 == 0:
            time.sleep(1)

    print(f"  YouTube: {len(dataset)} collected (captions: {caption_count}, whisper: {whisper_count}), {skipped} skipped")
    return dataset


def collect_manual(manual_urls_path: str, whisper_model: str = "base") -> list[dict]:
    """Process manual TikTok/Instagram URLs."""
    if not os.path.exists(manual_urls_path):
        print(f"\n=== Manual URLs ===")
        print(f"  File not found: {manual_urls_path} — skipping.")
        return []

    print(f"\n=== Manual URLs ({manual_urls_path}) ===")
    return process_manual_urls(manual_urls_path, whisper_model)


def deduplicate(dataset: list[dict]) -> list[dict]:
    """Remove duplicate entries by video ID."""
    seen = set()
    unique = []
    for item in dataset:
        if item["id"] not in seen:
            seen.add(item["id"])
            unique.append(item)
    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Build short-form video dataset from YouTube, TikTok, and Instagram."
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("YOUTUBE_API_KEY"),
        help="YouTube Data API key (or set YOUTUBE_API_KEY env var)",
    )
    parser.add_argument(
        "--manual-urls",
        default=str(Path(__file__).resolve().parent / "manual_urls.csv"),
        help="Path to CSV/JSON file with manual TikTok/Instagram URLs",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent.parent / "dataset" / "shorts_data.json"),
        help="Output path for the final dataset JSON",
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=25,
        help="Number of YouTube results per search query (default: 25)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for TikTok/Instagram transcription (default: base)",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: YouTube API key required. Use --api-key or set YOUTUBE_API_KEY env var.")
        sys.exit(1)

    output_path = Path(args.output)

    # Load existing data to merge with (don't lose previous results)
    existing = []
    if output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
            if existing:
                print(f"\n  Loaded {len(existing)} existing videos from {output_path}")
        except (json.JSONDecodeError, IOError):
            existing = []

    dataset = list(existing)

    # Collect from YouTube
    yt_data = collect_youtube(args.api_key, SEARCH_QUERIES, args.results_per_query, args.whisper_model)
    dataset.extend(yt_data)

    # Collect from manual URLs (TikTok/Instagram)
    manual_data = collect_manual(args.manual_urls, args.whisper_model)
    dataset.extend(manual_data)

    # Deduplicate
    dataset = deduplicate(dataset)

    # Clean transcripts
    cleaned_count = 0
    for item in dataset:
        if item.get("transcript"):
            item["transcript"] = clean_transcript(item["transcript"])
            cleaned_count += 1
    print(f"\n  Cleaned {cleaned_count} transcripts")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # Print summary
    platform_counts = {}
    for item in dataset:
        p = item["platform"]
        platform_counts[p] = platform_counts.get(p, 0) + 1

    print(f"\n=== Summary ===")
    print(f"  Total videos: {len(dataset)}")
    for platform, count in sorted(platform_counts.items()):
        print(f"    {platform}: {count}")
    print(f"  Output saved to: {output_path}")


if __name__ == "__main__":
    main()
