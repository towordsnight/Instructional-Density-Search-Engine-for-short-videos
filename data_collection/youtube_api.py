"""YouTube Data API client for searching and fetching short-form video metadata."""

from googleapiclient.discovery import build

# Diverse topic queries to get a broad dataset of short-form videos
SEARCH_QUERIES = [
    "how to cook",
    "quick workout tutorial",
    "DIY craft tutorial",
    "tech tips tutorial",
    "history explained shorts",
    "fashion design inspiration",
    "science experiment tutorial",
    "language learning tips",
    "photography tips",
    "music production tutorial",
]


def search_shorts(api_key: str, query: str, max_results: int = 25) -> list[dict]:
    """Search YouTube for short-form videos matching a query.

    Returns a list of dicts with video IDs and basic metadata.
    """
    youtube = build("youtube", "v3", developerKey=api_key)

    response = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        videoDuration="short",
        order="relevance",
        maxResults=max_results,
    ).execute()

    results = []
    for item in response.get("items", []):
        video_id = item["id"]["videoId"]
        snippet = item["snippet"]
        results.append({
            "video_id": video_id,
            "title": snippet["title"],
            "channel": snippet["channelTitle"],
            "published_at": snippet["publishedAt"],
        })

    return results


def get_video_details(api_key: str, video_ids: list[str]) -> list[dict]:
    """Fetch detailed metadata for a list of video IDs.

    Returns a list of dicts with title, description, view/like counts, etc.
    Processes in batches of 50 (API limit).
    """
    youtube = build("youtube", "v3", developerKey=api_key)
    all_details = []

    # Process in batches of 50 (YouTube API limit per request)
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        response = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(batch),
        ).execute()

        for item in response.get("items", []):
            snippet = item["snippet"]
            stats = item.get("statistics", {})
            all_details.append({
                "video_id": item["id"],
                "title": snippet["title"],
                "channel": snippet["channelTitle"],
                "published_at": snippet["publishedAt"],
                "description": snippet.get("description", ""),
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "url": f"https://www.youtube.com/shorts/{item['id']}",
            })

    return all_details
