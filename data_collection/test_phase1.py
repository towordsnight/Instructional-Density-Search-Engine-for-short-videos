"""Comprehensive tests for Phase 1: Data Collection Module.

Tests all four modules with mocked external APIs:
- youtube_api.py (Task 1.1)
- transcript_fetcher.py (Task 1.2)
- tiktok_instagram_collector.py (Task 1.3)
- build_dataset.py (Task 1.4)
"""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_collection.youtube_api import search_shorts, get_video_details, SEARCH_QUERIES
from data_collection.transcript_fetcher import fetch_youtube_transcript, transcribe_with_whisper
from data_collection.tiktok_instagram_collector import (
    _generate_id,
    load_manual_urls,
    process_manual_urls,
)
from data_collection.build_dataset import collect_youtube, collect_manual, deduplicate


# ============================================================================
# Task 1.1 — YouTube API Client (youtube_api.py)
# ============================================================================

class TestSearchQueries:
    """Test that the predefined search queries are well-formed."""

    def test_queries_exist(self):
        assert len(SEARCH_QUERIES) == 10

    def test_queries_are_nonempty_strings(self):
        for q in SEARCH_QUERIES:
            assert isinstance(q, str)
            assert len(q.strip()) > 0

    def test_queries_cover_diverse_topics(self):
        topics = {"cook", "workout", "diy", "tech", "history", "fashion", "science", "language", "photography", "music"}
        combined = " ".join(SEARCH_QUERIES).lower()
        for topic in topics:
            assert topic in combined, f"Missing topic: {topic}"


class TestSearchShorts:
    """Test search_shorts() with mocked YouTube API."""

    MOCK_SEARCH_RESPONSE = {
        "items": [
            {
                "id": {"videoId": "abc123"},
                "snippet": {
                    "title": "Quick Pasta Recipe",
                    "channelTitle": "CookChannel",
                    "publishedAt": "2025-12-01T00:00:00Z",
                },
            },
            {
                "id": {"videoId": "def456"},
                "snippet": {
                    "title": "5 Min Workout",
                    "channelTitle": "FitChannel",
                    "publishedAt": "2025-11-15T00:00:00Z",
                },
            },
        ]
    }

    @patch("data_collection.youtube_api.build")
    def test_returns_correct_structure(self, mock_build):
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_service.search().list().execute.return_value = self.MOCK_SEARCH_RESPONSE

        results = search_shorts("FAKE_KEY", "how to cook", max_results=10)

        assert len(results) == 2
        assert results[0]["video_id"] == "abc123"
        assert results[0]["title"] == "Quick Pasta Recipe"
        assert results[0]["channel"] == "CookChannel"
        assert "published_at" in results[0]

    @patch("data_collection.youtube_api.build")
    def test_empty_response(self, mock_build):
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_service.search().list().execute.return_value = {"items": []}

        results = search_shorts("FAKE_KEY", "nonexistent query")
        assert results == []

    @patch("data_collection.youtube_api.build")
    def test_api_called_with_correct_params(self, mock_build):
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_service.search().list().execute.return_value = {"items": []}

        search_shorts("FAKE_KEY", "test query", max_results=5)

        mock_build.assert_called_once_with("youtube", "v3", developerKey="FAKE_KEY")
        mock_service.search().list.assert_called_with(
            q="test query",
            part="snippet",
            type="video",
            videoDuration="short",
            order="relevance",
            maxResults=5,
        )


class TestGetVideoDetails:
    """Test get_video_details() with mocked YouTube API."""

    MOCK_DETAILS_RESPONSE = {
        "items": [
            {
                "id": "abc123",
                "snippet": {
                    "title": "Quick Pasta Recipe",
                    "channelTitle": "CookChannel",
                    "publishedAt": "2025-12-01T00:00:00Z",
                    "description": "A quick pasta recipe tutorial.",
                },
                "statistics": {
                    "viewCount": "150000",
                    "likeCount": "5000",
                },
            }
        ]
    }

    @patch("data_collection.youtube_api.build")
    def test_returns_correct_structure(self, mock_build):
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_service.videos().list().execute.return_value = self.MOCK_DETAILS_RESPONSE

        results = get_video_details("FAKE_KEY", ["abc123"])

        assert len(results) == 1
        detail = results[0]
        assert detail["video_id"] == "abc123"
        assert detail["title"] == "Quick Pasta Recipe"
        assert detail["views"] == 150000
        assert detail["likes"] == 5000
        assert detail["url"] == "https://www.youtube.com/shorts/abc123"
        assert detail["description"] == "A quick pasta recipe tutorial."

    @patch("data_collection.youtube_api.build")
    def test_handles_missing_statistics(self, mock_build):
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_service.videos().list().execute.return_value = {
            "items": [
                {
                    "id": "xyz789",
                    "snippet": {
                        "title": "No Stats Video",
                        "channelTitle": "SomeChannel",
                        "publishedAt": "2025-10-01T00:00:00Z",
                    },
                }
            ]
        }

        results = get_video_details("FAKE_KEY", ["xyz789"])
        assert results[0]["views"] == 0
        assert results[0]["likes"] == 0

    @patch("data_collection.youtube_api.build")
    def test_batches_over_50(self, mock_build):
        """Should call the API multiple times for >50 video IDs."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_service.videos().list().execute.return_value = {"items": []}

        video_ids = [f"vid_{i}" for i in range(75)]
        get_video_details("FAKE_KEY", video_ids)

        # Should have been called twice: once for 50, once for 25
        assert mock_service.videos().list().execute.call_count == 2


# ============================================================================
# Task 1.2 — Transcript Fetcher (transcript_fetcher.py)
# ============================================================================

class TestFetchYouTubeTranscript:
    """Test fetch_youtube_transcript() with mocked youtube-transcript-api."""

    @patch("data_collection.transcript_fetcher._yt_api")
    def test_returns_transcript_text(self, mock_api):
        # Mock transcript snippets
        snippet1 = MagicMock()
        snippet1.text = "Hello everyone"
        snippet2 = MagicMock()
        snippet2.text = "today we learn cooking"
        mock_api.fetch.return_value = [snippet1, snippet2]

        result = fetch_youtube_transcript("abc123")
        assert result == "Hello everyone today we learn cooking"

    @patch("data_collection.transcript_fetcher._yt_api")
    def test_returns_none_on_exception(self, mock_api):
        mock_api.fetch.side_effect = Exception("No captions available")

        result = fetch_youtube_transcript("no_captions")
        assert result is None

    @patch("data_collection.transcript_fetcher._yt_api")
    def test_returns_none_for_empty_transcript(self, mock_api):
        mock_api.fetch.return_value = []

        result = fetch_youtube_transcript("empty_vid")
        assert result is None

    @patch("data_collection.transcript_fetcher._yt_api")
    def test_strips_whitespace(self, mock_api):
        snippet = MagicMock()
        snippet.text = "  some text  "
        mock_api.fetch.return_value = [snippet]

        result = fetch_youtube_transcript("vid1")
        assert result == "some text"


class TestTranscribeWithWhisper:
    """Test transcribe_with_whisper() with mocked whisper module."""

    @patch("data_collection.transcript_fetcher.whisper", create=True)
    def test_returns_transcription(self, mock_whisper_module):
        # We need to mock the import inside the function
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "This is the transcribed audio"}

        with patch.dict("sys.modules", {"whisper": MagicMock()}):
            with patch("data_collection.transcript_fetcher.whisper", create=True) as mock_w:
                # Since whisper is imported inside the function, we need to patch sys.modules
                import importlib
                mock_whisper = MagicMock()
                mock_whisper.load_model.return_value = mock_model
                with patch.dict("sys.modules", {"whisper": mock_whisper}):
                    result = transcribe_with_whisper("/fake/audio.mp3", "base")
                    assert result == "This is the transcribed audio"

    def test_returns_none_when_whisper_not_installed(self):
        """If whisper is not installed, should return None gracefully."""
        with patch.dict("sys.modules", {"whisper": None}):
            # This will cause ImportError which is caught by the except block
            result = transcribe_with_whisper("/nonexistent/audio.mp3")
            assert result is None


# ============================================================================
# Task 1.3 — TikTok/Instagram Collector (tiktok_instagram_collector.py)
# ============================================================================

class TestGenerateId:
    """Test the _generate_id helper."""

    def test_returns_string(self):
        result = _generate_id("https://tiktok.com/video/123", "tiktok")
        assert isinstance(result, str)

    def test_prefix_for_tiktok(self):
        result = _generate_id("https://tiktok.com/video/123", "tiktok")
        assert result.startswith("ti_")

    def test_prefix_for_instagram(self):
        result = _generate_id("https://instagram.com/reel/ABC", "instagram")
        assert result.startswith("in_")

    def test_deterministic(self):
        """Same URL + platform should always produce the same ID."""
        id1 = _generate_id("https://tiktok.com/video/123", "tiktok")
        id2 = _generate_id("https://tiktok.com/video/123", "tiktok")
        assert id1 == id2

    def test_different_urls_different_ids(self):
        id1 = _generate_id("https://tiktok.com/video/123", "tiktok")
        id2 = _generate_id("https://tiktok.com/video/456", "tiktok")
        assert id1 != id2


class TestLoadManualUrls:
    """Test load_manual_urls() for CSV and JSON input."""

    def test_loads_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["url", "platform"])
            writer.writerow(["https://tiktok.com/video/123", "tiktok"])
            writer.writerow(["https://instagram.com/reel/ABC", "instagram"])
            f.flush()
            path = f.name

        try:
            result = load_manual_urls(path)
            assert len(result) == 2
            assert result[0]["url"] == "https://tiktok.com/video/123"
            assert result[0]["platform"] == "tiktok"
            assert result[1]["platform"] == "instagram"
        finally:
            os.unlink(path)

    def test_loads_json(self):
        data = [
            {"url": "https://tiktok.com/video/1", "platform": "tiktok"},
            {"url": "https://instagram.com/reel/2", "platform": "instagram"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = f.name

        try:
            result = load_manual_urls(path)
            assert len(result) == 2
            assert result[0]["url"] == "https://tiktok.com/video/1"
        finally:
            os.unlink(path)

    def test_skips_empty_rows_in_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["url", "platform"])
            writer.writerow(["https://tiktok.com/video/123", "tiktok"])
            writer.writerow(["", ""])  # empty row
            writer.writerow(["https://tiktok.com/video/456", "tiktok"])
            f.flush()
            path = f.name

        try:
            result = load_manual_urls(path)
            assert len(result) == 2
        finally:
            os.unlink(path)

    def test_loads_existing_manual_urls_csv(self):
        """Test loading the actual manual_urls.csv template."""
        csv_path = str(Path(__file__).resolve().parent / "manual_urls.csv")
        result = load_manual_urls(csv_path)
        assert isinstance(result, list)
        # Template has 2 example URLs
        assert len(result) == 2
        assert all("url" in r and "platform" in r for r in result)


class TestProcessManualUrls:
    """Test process_manual_urls() with mocked yt-dlp and Whisper."""

    @patch("data_collection.tiktok_instagram_collector.transcribe_with_whisper")
    @patch("data_collection.tiktok_instagram_collector._download_metadata_and_audio")
    def test_processes_urls_successfully(self, mock_download, mock_whisper):
        mock_download.return_value = {
            "title": "Cool TikTok",
            "uploader": "Creator1",
            "view_count": 50000,
            "audio_path": "/tmp/fake_audio.mp3",
        }
        mock_whisper.return_value = "This is a tutorial about cooking pasta quickly"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["url", "platform"])
            writer.writerow(["https://tiktok.com/video/123", "tiktok"])
            f.flush()
            path = f.name

        try:
            results = process_manual_urls(path)
            assert len(results) == 1
            assert results[0]["platform"] == "tiktok"
            assert results[0]["title"] == "Cool TikTok"
            assert results[0]["transcript"] == "This is a tutorial about cooking pasta quickly"
            assert results[0]["channel"] == "Creator1"
            assert results[0]["views"] == 50000
            assert results[0]["url"] == "https://tiktok.com/video/123"
            assert results[0]["id"].startswith("ti_")
        finally:
            os.unlink(path)

    @patch("data_collection.tiktok_instagram_collector.transcribe_with_whisper")
    @patch("data_collection.tiktok_instagram_collector._download_metadata_and_audio")
    def test_skips_failed_downloads(self, mock_download, mock_whisper):
        mock_download.return_value = None  # download failed

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["url", "platform"])
            writer.writerow(["https://tiktok.com/video/123", "tiktok"])
            f.flush()
            path = f.name

        try:
            results = process_manual_urls(path)
            assert results == []
        finally:
            os.unlink(path)

    @patch("data_collection.tiktok_instagram_collector.transcribe_with_whisper")
    @patch("data_collection.tiktok_instagram_collector._download_metadata_and_audio")
    def test_skips_failed_transcription(self, mock_download, mock_whisper):
        mock_download.return_value = {
            "title": "Silent Video",
            "uploader": "Creator",
            "view_count": 100,
            "audio_path": "/tmp/audio.mp3",
        }
        mock_whisper.return_value = None  # transcription failed

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["url", "platform"])
            writer.writerow(["https://tiktok.com/video/123", "tiktok"])
            f.flush()
            path = f.name

        try:
            results = process_manual_urls(path)
            assert results == []
        finally:
            os.unlink(path)

    def test_empty_file_returns_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["url", "platform"])  # header only
            f.flush()
            path = f.name

        try:
            results = process_manual_urls(path)
            assert results == []
        finally:
            os.unlink(path)


# ============================================================================
# Task 1.4 — Build Dataset (build_dataset.py)
# ============================================================================

class TestDeduplicate:
    """Test the deduplicate() function."""

    def test_removes_duplicates(self):
        data = [
            {"id": "yt_abc", "title": "Video 1"},
            {"id": "yt_def", "title": "Video 2"},
            {"id": "yt_abc", "title": "Video 1 duplicate"},
        ]
        result = deduplicate(data)
        assert len(result) == 2
        assert result[0]["title"] == "Video 1"  # keeps first occurrence

    def test_no_duplicates_passthrough(self):
        data = [
            {"id": "yt_1", "title": "A"},
            {"id": "yt_2", "title": "B"},
            {"id": "yt_3", "title": "C"},
        ]
        result = deduplicate(data)
        assert len(result) == 3

    def test_empty_input(self):
        assert deduplicate([]) == []

    def test_all_same_id(self):
        data = [
            {"id": "same", "title": "First"},
            {"id": "same", "title": "Second"},
            {"id": "same", "title": "Third"},
        ]
        result = deduplicate(data)
        assert len(result) == 1
        assert result[0]["title"] == "First"


class TestCollectYouTube:
    """Test collect_youtube() with mocked API calls."""

    @patch("data_collection.build_dataset.fetch_youtube_transcript")
    @patch("data_collection.build_dataset.get_video_details")
    @patch("data_collection.build_dataset.search_shorts")
    def test_collects_videos_with_transcripts(self, mock_search, mock_details, mock_transcript):
        mock_search.return_value = [
            {"video_id": "abc123", "title": "Test Video", "channel": "TestCh", "published_at": "2025-01-01T00:00:00Z"},
        ]
        mock_details.return_value = [
            {
                "video_id": "abc123",
                "title": "Test Video",
                "channel": "TestCh",
                "published_at": "2025-01-01T00:00:00Z",
                "views": 10000,
                "likes": 500,
                "url": "https://www.youtube.com/shorts/abc123",
            }
        ]
        mock_transcript.return_value = "This is a tutorial about cooking"

        result = collect_youtube("FAKE_KEY", ["cooking tutorial"], results_per_query=5)

        assert len(result) == 1
        assert result[0]["id"] == "yt_abc123"
        assert result[0]["platform"] == "youtube"
        assert result[0]["transcript"] == "This is a tutorial about cooking"
        assert result[0]["url"] == "https://www.youtube.com/shorts/abc123"

    @patch("data_collection.build_dataset.fetch_youtube_transcript")
    @patch("data_collection.build_dataset.get_video_details")
    @patch("data_collection.build_dataset.search_shorts")
    def test_skips_videos_without_transcripts(self, mock_search, mock_details, mock_transcript):
        mock_search.return_value = [
            {"video_id": "abc123", "title": "V1", "channel": "C1", "published_at": "2025-01-01T00:00:00Z"},
            {"video_id": "def456", "title": "V2", "channel": "C2", "published_at": "2025-01-01T00:00:00Z"},
        ]
        mock_details.return_value = [
            {"video_id": "abc123", "title": "V1", "channel": "C1", "published_at": "2025-01-01T00:00:00Z", "views": 100, "likes": 10, "url": "https://www.youtube.com/shorts/abc123"},
            {"video_id": "def456", "title": "V2", "channel": "C2", "published_at": "2025-01-01T00:00:00Z", "views": 200, "likes": 20, "url": "https://www.youtube.com/shorts/def456"},
        ]
        # First video has transcript, second doesn't
        mock_transcript.side_effect = ["Transcript here", None]

        result = collect_youtube("FAKE_KEY", ["test"], results_per_query=5)

        assert len(result) == 1
        assert result[0]["id"] == "yt_abc123"

    @patch("data_collection.build_dataset.fetch_youtube_transcript")
    @patch("data_collection.build_dataset.get_video_details")
    @patch("data_collection.build_dataset.search_shorts")
    def test_deduplicates_across_queries(self, mock_search, mock_details, mock_transcript):
        """Same video appearing in two different query results should only appear once."""
        mock_search.return_value = [
            {"video_id": "abc123", "title": "V1", "channel": "C1", "published_at": "2025-01-01T00:00:00Z"},
        ]
        mock_details.return_value = [
            {"video_id": "abc123", "title": "V1", "channel": "C1", "published_at": "2025-01-01T00:00:00Z", "views": 100, "likes": 10, "url": "https://www.youtube.com/shorts/abc123"},
        ]
        mock_transcript.return_value = "Some transcript"

        # Two queries returning the same video
        result = collect_youtube("FAKE_KEY", ["query1", "query2"], results_per_query=5)

        # Should only have 1 video despite 2 queries
        assert len(result) == 1


class TestCollectManual:
    """Test collect_manual() wrapper."""

    def test_returns_empty_for_missing_file(self):
        result = collect_manual("/nonexistent/path/manual_urls.csv")
        assert result == []

    @patch("data_collection.build_dataset.process_manual_urls")
    def test_calls_process_manual_urls(self, mock_process):
        mock_process.return_value = [{"id": "ti_test", "platform": "tiktok"}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("url,platform\n")
            f.flush()
            path = f.name

        try:
            result = collect_manual(path)
            assert len(result) == 1
            mock_process.assert_called_once_with(path, "base")
        finally:
            os.unlink(path)


class TestBuildDatasetCLI:
    """Test the main() CLI argument parsing."""

    def test_cli_requires_api_key(self):
        """main() should exit with error if no API key is provided."""
        from data_collection.build_dataset import main

        with patch("sys.argv", ["build_dataset.py"]):
            with patch.dict(os.environ, {}, clear=True):
                # Remove YOUTUBE_API_KEY if set
                env = os.environ.copy()
                env.pop("YOUTUBE_API_KEY", None)
                with patch.dict(os.environ, env, clear=True):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 1


class TestEndToEndDataFormat:
    """Test that the overall data pipeline produces correctly formatted output."""

    @patch("data_collection.build_dataset.fetch_youtube_transcript")
    @patch("data_collection.build_dataset.get_video_details")
    @patch("data_collection.build_dataset.search_shorts")
    def test_output_has_required_fields(self, mock_search, mock_details, mock_transcript):
        mock_search.return_value = [
            {"video_id": "abc123", "title": "V1", "channel": "C1", "published_at": "2025-01-01T00:00:00Z"},
        ]
        mock_details.return_value = [
            {
                "video_id": "abc123",
                "title": "Test Video",
                "channel": "TestChannel",
                "published_at": "2025-01-01T00:00:00Z",
                "description": "A description",
                "views": 5000,
                "likes": 200,
                "url": "https://www.youtube.com/shorts/abc123",
            }
        ]
        mock_transcript.return_value = "This is a great tutorial about cooking"

        result = collect_youtube("FAKE_KEY", ["cooking"], results_per_query=5)

        required_fields = {"id", "platform", "title", "transcript", "url", "channel", "views", "likes", "published_at"}
        for item in result:
            assert required_fields.issubset(item.keys()), f"Missing fields: {required_fields - item.keys()}"
            assert isinstance(item["id"], str)
            assert isinstance(item["transcript"], str)
            assert len(item["transcript"]) > 0
            assert item["platform"] == "youtube"

    def test_dataset_json_is_valid(self):
        """Test that we can write and read back a dataset JSON file."""
        sample_data = [
            {
                "id": "yt_test1",
                "platform": "youtube",
                "title": "Test Video 1",
                "transcript": "Hello world this is a test",
                "url": "https://www.youtube.com/shorts/test1",
                "channel": "TestChannel",
                "views": 1000,
                "likes": 50,
                "published_at": "2025-01-01T00:00:00Z",
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_data, f, indent=2)
            path = f.name

        try:
            with open(path) as f:
                loaded = json.load(f)
            assert len(loaded) == 1
            assert loaded[0]["id"] == "yt_test1"
            assert loaded[0]["transcript"] == "Hello world this is a test"
        finally:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
