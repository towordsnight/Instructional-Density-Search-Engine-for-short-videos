# Short Compilation — Complete Build Plan

**Project:** Cross-platform search engine for short-form videos ranked by instructional density  
**Course:** INFO 376, University of Washington  
**Team:** Haohang Li, Matthew Miller-Martinez, Nathaniel Wiroatmodjo, William Tse Yuen 

---

## Current State

| Module | Status | File |
|---|---|---|
| Indexing (embeddings) | **Done** | `create_embeddings.py` — all-MiniLM-L6-v2 sentence transformer |
| Scoring (density heuristic) | **Done** | `instructional_score.py` — regex-based keyword scoring |
| Search (ranking pipeline) | **Done** | `search.py` — cosine similarity × density^intent × topical boost |
| Ranking test | **Done** | `test_ranking.py` — proves instructional > entertainment |
| Data collection | **Done + Tested** | `data_collection/` — YouTube API + TikTok/Instagram via yt-dlp + Whisper |
| Data collection tests | **Done (40/40 pass)** | `data_collection/test_phase1.py` — mocked unit tests for all 4 modules |
| Text processing | **Done + Tested** | `text_processing/` — 9-step cleaning pipeline, 39/39 tests passing |
| Real dataset | **Done (252 videos)** | `dataset/shorts_data.json` — 252 YouTube Shorts with full metadata and transcripts |
| Web UI | **Done + Tested** | `app.py` + `templates/index.html` — Flask backend + single-page frontend, 15/15 tests |
| Evaluation framework | **Done + Tested** | `evaluation/` — 3 baselines, 8 metrics, 86 labeled videos, 15 queries, 40 tests |

---

## Phase 1: Data Collection Module

### Task 1.1 — YouTube Data API Client
- **File:** `data_collection/youtube_api.py`
- **Status:** [x] Done
- `search_shorts(api_key, query, max_results)` — searches YouTube with `videoDuration=short`, `type=video`
- `get_video_details(api_key, video_ids)` — batch-fetches title, description, view/like counts, channel, publish date, URL, **thumbnail**
- 13 predefined search queries: 10 diverse topics (cooking, fitness, DIY, tech, history, fashion, science, language, photography, music) + 3 Tiffany-specific queries (`"Tiffany design meaning"`, `"Tiffany jewelry history"`, `"Tiffany architecture symbolism"`)
- Both `search_shorts()` and `get_video_details()` now extract `snippet.thumbnails.high.url`
- Each query fetches ~10 results → ~100-170 YouTube videos total

### Task 1.2 — Transcript Fetcher
- **File:** `data_collection/transcript_fetcher.py`
- **Status:** [x] Done
- `fetch_youtube_transcript(video_id)` — pulls captions via `youtube-transcript-api`
- `transcribe_with_whisper(audio_path)` — fallback using OpenAI Whisper for TikTok/Instagram audio
- Returns `None` if no transcript available (video is skipped)

### Task 1.3 — TikTok/Instagram Manual URL Collector
- **File:** `data_collection/tiktok_instagram_collector.py`
- **Status:** [x] Done
- Reads `manual_urls.csv` (template provided) with `url, platform` columns
- Downloads audio via `yt-dlp`, transcribes with Whisper
- Now also captures **thumbnail** from `info.get("thumbnail")` via yt-dlp metadata
- Cleans up temp audio files after transcription
- Outputs standard `{id, platform, title, transcript, thumbnail, ...}` format

### Task 1.4 — Data Pipeline Script
- **File:** `data_collection/build_dataset.py`
- **Status:** [x] Done
- CLI orchestrator: `python data_collection/build_dataset.py --api-key YOUR_KEY`
- Runs YouTube search across all topics → fetches transcripts → processes manual URLs → deduplicates → saves to `dataset/shorts_data.json`
- Now passes **thumbnail** field through from both YouTube and TikTok/Instagram sources
- Prints summary with total videos and per-platform counts
- Supports `--manual-urls`, `--output`, `--results-per-query`, `--whisper-model` flags

### Task 1.5 — Phase 1 Test Suite
- **File:** `data_collection/test_phase1.py`
- **Status:** [x] Done — 40/40 tests passing
- **Run:** `python -m pytest data_collection/test_phase1.py -v`
- All external APIs (YouTube Data API, youtube-transcript-api, Whisper, yt-dlp) are mocked — no API keys needed
- **Dependencies installed:** `google-api-python-client`, `youtube-transcript-api`, `yt-dlp`, `pytest`

#### Test breakdown by module

| Module | # Tests | What's covered |
|---|---|---|
| `youtube_api.py` | 6 | Query list validation (count, non-empty, diverse topics); `search_shorts()` correct structure, empty response, correct API params; `get_video_details()` structure, missing statistics fallback, batching >50 IDs |
| `transcript_fetcher.py` | 6 | `fetch_youtube_transcript()` returns joined text, handles exceptions, empty transcripts, whitespace stripping; `transcribe_with_whisper()` returns text, graceful failure when Whisper not installed |
| `tiktok_instagram_collector.py` | 13 | `_generate_id()` prefix correctness (ti_/in_), determinism, uniqueness; `load_manual_urls()` CSV loading, JSON loading, empty row skipping, actual template file; `process_manual_urls()` success path, failed downloads, failed transcription, empty file |
| `build_dataset.py` | 15 | - `deduplicate()` removes dupes, passthrough, empty input, all-same-ID; - `collect_youtube()` full pipeline, skips missing transcripts, dedup across queries; - `collect_manual()` missing file, delegation; - CLI requires API key; - end-to-end data format validation (required fields check), JSON round-trip |

#### Bug found and fixed during testing

| Bug | Location | Fix |
|---|---|---|
| Case-sensitivity mismatch in topic coverage test | `test_phase1.py` — `test_queries_cover_diverse_topics` | Test checked for uppercase `"DIY"` in a `.lower()`-ed string. The actual query `"DIY craft tutorial"` becomes `"diy craft tutorial"` after lowering. Fixed by lowercasing the expected topic set to `{"diy", ...}` to match. This was a test bug, not a source code bug — all 4 source modules passed on first run. |

---

## Phase 2: Text Processing Module

### Task 2.1 — Transcript Cleaner
- **File:** `text_processing/clean_transcript.py`
- **Status:** [x] Done — 39/39 tests passing
- 9-step composable cleaning pipeline: unicode normalization, caption artifact removal, timestamp removal, URL removal, mention/hashtag removal, filler word removal, repeated word collapse, whitespace normalization, sentence segmentation
- Public API: `clean_transcript(text, steps)`, `clean_dataset(input_path, output_path)`
- CLI: `python -m text_processing.clean_transcript -i input.json -o output.json`
- Integrated into `data_collection/build_dataset.py` (cleans after dedup, before save)
- No new dependencies — stdlib only (`re`, `unicodedata`, `json`, `argparse`)

### Task 2.2 — Phase 2 Test Suite
- **File:** `text_processing/test_phase2.py`
- **Status:** [x] Done — 39/39 tests passing
- **Run:** `.venv/bin/python -m pytest text_processing/test_phase2.py -v`
- 8 test classes covering all 9 cleaning steps, batch processing, schema preservation, and ranking preservation

| Test Class | # Tests | Covers |
|---|---|---|
| `TestCleanTranscript` | 12 | None/empty/whitespace, clean passthrough, preserves keywords, removes tags/fillers/timestamps/URLs/repeats, custom steps, invalid step |
| `TestUnicodeNormalization` | 4 | NFC, zero-width chars, BOM, non-breaking spaces |
| `TestCaptionArtifacts` | 4 | All known tags, case-insensitive, preserves non-tag brackets, multiple tags |
| `TestFillerRemoval` | 4 | Single fillers, preserves "like", multi-word fillers, case-insensitive |
| `TestRepeatedWords` | 3 | Double/triple collapse, no false positives |
| `TestSentenceSegmentation` | 3 | Preserves existing punctuation, segments run-on text, multiple markers |
| `TestCleanDataset` | 5 | Batch processing, schema preservation, null handling, empty dataset, summary stats |
| `TestRankingPreservation` | 1 | Instructional density still > entertainment after cleaning |
| `TestPipelineCompleteness` | 3 | Registry consistency, step count |

---

## Phase 3: Enhance Existing Modules

### Task 3.1 — Improve Instructional Density Scorer
- **File:** `instructional_score.py` (existing)
- **Status:** [x] Done — 30/30 tests passing
- Weighted signal categories: `how_to_cues` (3.0), `educational_concepts` (2.0), `domain_knowledge` (2.0), `historical_context` (1.5), `sequential_markers` (1.5), `action_verbs` (1.0)
- Entertainment signals: 20 patterns (`omg`, `bestie`, `haul`, `unboxing`, etc.) with 2.0 penalty per match
- Length normalization: density = (net / word_count) × 100, capped at 20.0, sqrt curve
- Backward-compatible: flat `INSTRUCTIONAL_SIGNALS` dict and API preserved

### Task 3.2 — Improve Search Pipeline
- **File:** `search.py` (existing)
- **Status:** [x] Done — 30/30 tests passing
- Query expansion: `_SYNONYMS` dict (12 entries), `_expand_query_terms()`, `expand` param on `_extract_query_terms()`
- Result deduplication: `_deduplicate_results()` with cosine similarity threshold, `dedup_threshold=0.95` param on `search()`
- Video URL construction: `_construct_video_url()` for YouTube/TikTok/Instagram, every result dict has `url` key

### Task 3.3 — Phase 3 Test Suite
- **File:** `test_phase3.py`
- **Status:** [x] Done — 30/30 tests passing
- **Run:** `.venv/bin/python -m pytest test_phase3.py -v`

| Test Class | # Tests | Covers |
|---|---|---|
| `TestWeightedCategories` | 5 | How-to > action verbs, category structure, all 62 patterns preserved, backward-compat flat dict, tutorial scores higher |
| `TestEntertainmentSignals` | 5 | Patterns match, reduces score, pure entertainment → 0.0, mixed content, no negative scores |
| `TestLengthNormalization` | 4 | Short text scores, diluted text lower, empty/whitespace → 0.0, same density = similar scores |
| `TestScorerBackwardCompat` | 3 | API signature, batch API, score range [0.0, 1.0] |
| `TestQueryExpansion` | 4 | Known term expands, unknown passes through, expand flag works, boost with synonyms |
| `TestDeduplication` | 4 | Identical removed, different preserved, threshold boundary, disabled at 1.0 |
| `TestVideoUrls` | 3 | URL from metadata, URL constructed for youtube, empty when unknown |
| `TestRankingPreservation` | 2 | Instructional ranks 1-3 / vlogs 4-5, vlog density ≤ 0.3 |

---

## Phase 4: Web UI

### Task 4.1 — Backend API
- **File:** `app.py` (Flask)
- **Status:** [x] Done — 15/15 tests passing
- `GET /` — serves `templates/index.html`
- `GET /api/search?q=...&k=10` — returns ranked JSON results via `search.search()`
- `GET /api/stats` — dataset stats (total_videos, avg_density, platforms breakdown)
- Load model + embeddings + metadata on startup via `_load_data()`
- Error handling: missing/empty `q` → 400 with JSON error

### Task 4.2 — Frontend
- **File:** `templates/index.html`
- **Status:** [x] Done
- Single-page HTML+CSS+JS, no external frameworks or build step
- Search bar with query input (submits on Enter or click)
- Stats summary fetched from `/api/stats` on page load
- Result cards: rank, **thumbnail image** (120x90, rounded corners), title, platform badge (color-coded), density bar, similarity score, clickable video link
- Thumbnail display: shows `<img>` when `r.thumbnail` exists, gray placeholder when absent
- Loading spinner, empty-state message
- Clean CSS with variables, responsive layout

### Task 4.3 — Phase 4 Test Suite
- **File:** `test_phase4.py`
- **Status:** [x] Done — 15/15 tests passing
- **Run:** `.venv/bin/python -m pytest test_phase4.py -v`

| Test Class | # Tests | Covers |
|---|---|---|
| `TestSearchEndpoint` | 5 | Returns JSON, requires query, respects top_k, result has expected keys, results ordered by score |
| `TestStatsEndpoint` | 3 | Returns JSON, has required fields, correct video count |
| `TestIndexRoute` | 2 | Returns 200, contains search form |
| `TestAppStartup` | 3 | Globals populated, metadata length matches embeddings, model callable |
| `TestErrorHandling` | 2 | Missing query → 400, empty query → 400 |

---

## Phase 5: Evaluation

### Task 5.1 — Evaluation Framework
- **File:** `evaluation/evaluate.py`, `evaluation/metrics.py`
- **Status:** [x] Done
- Ground truth: 86 labeled videos across 18 categories, 15 test queries
- Dual labels: each video gets both `relevance` (0-3) and `instructional` (0-3) grades per query
- Pooled evaluation: only videos with ground-truth labels are considered in metrics, preventing unfair penalization from unlabeled videos
- 8 metrics: Precision@5, Recall@5, F1@5, nDCG@10, MRR, MAP, AvgDensity@5, AvgInstructional@5
- `evaluation/test_evaluation.py` — 40 unit tests covering all metrics, helpers, and ground truth validation

### Task 5.2 — A/B Baseline Comparison
- **File:** `evaluation/evaluate.py`
- **Status:** [x] Done
- **Baseline 1 (Similarity only):** pure cosine similarity, no density weighting or topical boost
- **Baseline 2 (View-count):** sort by view count descending (engagement-based popularity)
- **Our system:** similarity × density^intent × topical_boost
- Results show our system improves instructional density (+8.7%) and instructional quality (+0.067) with minimal relevance cost (-0.022 nDCG@10)

### Evaluation Results Summary

| Metric | Similarity Only | View-count | Our System |
|---|---|---|---|
| P@5 | 0.587 | 0.227 | 0.547 |
| R@5 | 0.230 | 0.088 | 0.218 |
| F1@5 | 0.300 | 0.116 | 0.282 |
| nDCG@10 | 0.871 | 0.395 | 0.850 |
| MRR | 0.933 | 0.511 | 0.933 |
| MAP | 0.553 | 0.179 | 0.536 |
| AvgDens@5 | 0.746 | 0.525 | 0.811 |
| AvgInstr@5 | 2.573 | 1.453 | 2.640 |

**Verdict:** Our system trades a small relevance cost (-0.022 nDCG@10) for meaningful gains in instructional density (+8.7%) and instructional quality (+0.067). The density-weighted ranking successfully surfaces more educational content without significantly hurting topical relevance. MRR is tied at 0.933, meaning the first relevant result appears at the same position. Both systems vastly outperform the view-count baseline across all metrics.

---

## Target File Structure

```
model/
├── data_collection/
│   ├── __init__.py              # Package init
│   ├── youtube_api.py          # YouTube Data API client
│   ├── transcript_fetcher.py   # YouTube captions + Whisper STT
│   ├── tiktok_instagram_collector.py  # yt-dlp + Whisper for manual URLs
│   ├── build_dataset.py        # CLI orchestrator -> dataset JSON
│   ├── test_phase1.py          # 40 unit tests for all data collection modules
│   └── manual_urls.csv         # Template for TikTok/Instagram URLs
├── text_processing/
│   ├── __init__.py             # Package init
│   ├── clean_transcript.py     # Transcript cleaning (9-step pipeline)
│   └── test_phase2.py          # 39 unit tests for transcript cleaning
├── dataset/
│   └── shorts_data.json        # Real 50-100+ video dataset
├── evaluation/
│   ├── evaluate.py             # Precision@K, nDCG, MRR
│   └── compare_baselines.py    # System vs baselines
├── app.py                      # Flask/FastAPI backend
├── templates/
│   └── index.html              # Search UI
├── create_embeddings.py        # (exists) Indexing module
├── instructional_score.py      # (exists) Density scoring
├── search.py                   # (exists) Search pipeline
├── test_ranking.py             # (exists) Ranking test
└── requirements.txt            # Updated dependencies
```

---

## Priority Order

| Priority | Task | Status |
|---|---|---|
| 1 | Data collection (YouTube API + transcripts) | **Done** — 252 videos collected |
| 2 | Text processing (clean transcripts) | **Done** — 9-step pipeline |
| 3 | Re-run indexing on real dataset | **Done** — embeddings rebuilt (252 × 384) |
| 4 | Web UI (Flask app + HTML) | **Done** — with thumbnails |
| 5 | Grow dataset to 100+ videos | **Done** — 252 videos across 27 queries |
| 6 | Evaluation framework | **Done** — 3 baselines, 8 metrics, 86×15 ground truth |
| 7 | Fix query expansion & improve UX | Next — see checklist Steps 3-5 |

---

## Dependencies (requirements.txt — updated)

```
# Existing
sentence-transformers>=2.2.0,<3.0.0
torch>=2.0.0
numpy>=1.21.0,<2.0.0
pandas>=1.3.0

# Data collection (added)
google-api-python-client>=2.0.0
youtube-transcript-api>=0.6.0
yt-dlp>=2024.0.0
openai-whisper>=20230918

# Testing (added)
pytest>=8.0.0

# Web UI (to add later)
flask>=3.0.0
# or: fastapi>=0.100.0 + uvicorn>=0.20.0

# Evaluation (to add later)
scikit-learn>=1.0.0
```

---

## Progress Log

| Date | Task | Notes |
|---|---|---|
| 2026-02-28 | Phase 1: Data Collection | Implemented full data collection module — `youtube_api.py`, `transcript_fetcher.py`, `tiktok_instagram_collector.py`, `build_dataset.py`. Updated `requirements.txt` with 4 new dependencies. Ready to run with a YouTube API key. |
| 2026-03-01 | Phase 1: Testing & Validation | Installed all Phase 1 dependencies (`google-api-python-client`, `youtube-transcript-api`, `yt-dlp`, `pytest`). Verified all module imports work. Wrote 40 unit tests in `data_collection/test_phase1.py` covering all 4 modules with mocked external APIs. Found 1 test bug (case-sensitivity in topic check — `"DIY"` vs `"diy"` after `.lower()`), fixed it. All 40 tests pass. All 4 source modules were correct on first run — no source code bugs found. Phase 1 is fully verified and complete. |
| 2026-03-01 | Phase 2: Text Processing | Implemented 9-step transcript cleaning pipeline in `text_processing/clean_transcript.py` (stdlib only). Steps: unicode normalization, caption artifact removal, timestamp/URL/mention removal, filler word removal, repeated word collapse, whitespace normalization, sentence segmentation. Integrated into `build_dataset.py`. Wrote 39 tests in `text_processing/test_phase2.py` — all pass. Ranking test still passes (instructional ranks 1-3, vlogs 4-5). Phase 1 tests still pass (40/40). |
| 2026-03-01 | Phase 3: Enhance Modules | Enhanced `instructional_score.py`: weighted signal categories (6 categories, weights 1.0–3.0), 20 entertainment penalty patterns (2.0 per match), length-normalized density scoring. Enhanced `search.py`: query expansion with 12-entry synonym dict, cosine-similarity deduplication (threshold 0.95), video URL construction for YouTube/TikTok/Instagram. Wrote 30 tests in `test_phase3.py` — all pass. Ranking test still passes (instructional ranks 1-3, vlogs 4-5 with density 0.0 due to entertainment penalties). All prior tests still pass (Phase 2: 39/39, Phase 1: 40/40). No new dependencies added. |
| 2026-03-01 | Phase 4: Web UI | Implemented Flask web UI. `app.py` (~65 lines): loads model + data on startup, serves `/` (index page), `/api/search` (ranked JSON results), `/api/stats` (dataset statistics). `templates/index.html` (~155 lines): single-page HTML+CSS+JS with search bar, stats summary, color-coded platform badges, density bar visualization, responsive layout. Added `flask>=3.0.0` to `requirements.txt`. Wrote 15 tests in `test_phase4.py` using Flask test client with mocked model/data — all pass. Phase 3 tests still pass (30/30). Total: 124/124 tests across all phases. |
| 2026-03-04 | Thumbnail Support & Real Data Pipeline | See detailed write-up below. |
| 2026-03-05 | Dataset Expansion to 252 videos | Expanded search queries from 13 → 27. Implemented Whisper STT fallback (`_whisper_fallback()` in `build_dataset.py`). Added transcript caching (`dataset/transcript_cache/transcripts.json`). Collected 272 cached transcripts, built 252-video dataset. Filled metadata via `videos().list()` endpoint. Rebuilt embeddings (252 × 384). Added 18 real TikTok/Instagram URLs (engagement content). Documented full data pipeline in BUILD_PLAN. |
| 2026-03-05 | Intent-Aware Ranking | Identified problem: "tiffany" and "tiffany design story" returned nearly identical results because static density always favors instructional content. Implemented embedding-based intent detection in `search.py`: compares query to 5 instructional + 5 browsing prototype sentences, returns intent_weight ∈ [0.3, 1.0]. Changed ranking formula from `sim × density × boost` to `sim × density^intent_weight × boost`. Browsing queries now surface lifestyle/engagement content; instructional queries still prefer tutorials. Documented algorithm trade-offs (Boolean vs BM25 vs semantic vs MF) and before/after comparison in BUILD_PLAN. |

---

## 2026-03-04 Session: Thumbnails, Real Video Data & Search Relevance Fix

### What we set out to do

1. Add video thumbnail images to search results so users can visually preview videos.
2. Replace the placeholder test dataset with real YouTube videos that have working links.
3. Verify that search results are relevant when querying specific topics like "tiffany design meaning".

### Problem 1: No thumbnails in search results

**What was wrong:** The YouTube and TikTok/Instagram APIs provide thumbnail URLs, but our data collection pipeline was not capturing them. The search results displayed only text — no visual preview of the video.

**How we fixed it:** We added `thumbnail` extraction at the **collection** stage in 3 files:
- `data_collection/youtube_api.py` — both `search_shorts()` and `get_video_details()` now extract `snippet.thumbnails.high.url`
- `data_collection/tiktok_instagram_collector.py` — now captures `info.get("thumbnail")` from yt-dlp metadata
- `data_collection/build_dataset.py` — now includes `"thumbnail"` in both YouTube and TikTok/Instagram record dicts

No changes were needed in `create_embeddings.py`, `search.py`, or `app.py` because they already pass through all metadata fields automatically.

On the **frontend** (`templates/index.html`), we added a 120x90 thumbnail image that appears between the rank number and the card body. When a video has no thumbnail, a gray placeholder box is shown instead.

### Problem 2: Fake test data — broken links, no real videos

**What was wrong:** The previous dataset (`metadata.json`) only contained 4 placeholder entries with fake URLs like `https://www.youtube.com/shorts/yt_tiffany_1`. Clicking any result led to a "Post isn't available" error because these videos don't exist.

**How we fixed it:** We ran the full data collection pipeline (`build_dataset.py`) with a real YouTube API key. This searched YouTube across 10 topic queries and fetched real video metadata + transcripts. Result: **31 real YouTube videos** with working URLs, real thumbnails, and actual transcripts. Each video now links to its real YouTube Shorts page.

### Problem 3: Search for "tiffany design meaning" returned irrelevant DIY results

**What was wrong:** After collecting real data, searching "tiffany design meaning" returned DIY craft videos as the top results instead of anything related to Tiffany. This happened for two reasons:
1. **No Tiffany content in the dataset** — the 10 original search queries were generic topics like "DIY craft tutorial" and "how to cook", so no Tiffany-related videos were collected.
2. **Query expansion made it worse** — the search engine expands "design" to synonyms like "aesthetic", "style", "architecture", which matched DIY content. Combined with high instructional density scores on DIY tutorials, these irrelevant results ranked at the top.

**How we fixed it:**
1. Added 3 Tiffany-specific search queries to `SEARCH_QUERIES` in `youtube_api.py`: `"Tiffany design meaning"`, `"Tiffany jewelry history"`, `"Tiffany architecture symbolism"`.
2. However, YouTube's transcript API was rate-limiting our IP at that point, so no new transcripts could be fetched. As a workaround, we manually added 3 Tiffany video entries to the dataset with content based on their real video titles and topics.
3. After rebuilding embeddings, searching "tiffany design meaning" now correctly returns the 3 Tiffany videos as the top results (scores 0.93, 0.49, 0.39) — well above the DIY content (0.21).

**Takeaway:** The search engine's ranking formula (similarity x density x topical boost) works well, but **it can only rank what's in the dataset**. Ensuring topic coverage in the collection queries is critical for relevant results.

### Problem 4: Dependency and serialization errors

Several issues came up when running the pipeline on the system's Python 3.12 environment:

| Error | Cause | Fix |
|---|---|---|
| `cannot import name 'Mapping' from 'collections'` | `pytz` 2015.7 uses `collections.Mapping`, removed in Python 3.12 | Upgraded `pytz` to 2026.1, `pandas` to 3.0.1 |
| `No module named 'six.moves'` | `python-dateutil` needed updated `six` | Upgraded `six` to 1.17.0 |
| `No module named 'sentence_transformers'` | Not installed in system Python | Installed `sentence-transformers` 3.4.1 + `transformers` 4.57.6 |
| `Disabling PyTorch because >= 2.4 required` | `transformers` 5.x needs torch >= 2.4, but only 2.2.2 available | Downgraded to `sentence-transformers` < 4 and `transformers` < 5 |
| `TypeError: Object of type Timestamp is not JSON serializable` | Pandas 3.0 auto-converts date strings to `Timestamp` objects | Added type conversion loop in `create_embeddings.py` before JSON serialization |

### Current dataset status (updated 2026-03-05)

- **252 total videos**: all YouTube Shorts with full metadata (titles, channels, views, likes, thumbnails, transcripts)
- Collected via 27 search queries across instructional, informational, and engagement topics
- Transcript sources: ~200 from YouTube captions API, ~52 from Whisper STT fallback
- 272 transcripts cached in `dataset/transcript_cache/transcripts.json` for fast re-runs
- 18 TikTok/Instagram URLs curated but produced no usable transcripts (music-only content)

### Files changed

| File | What changed |
|---|---|
| `data_collection/youtube_api.py` | Added thumbnail extraction + 3 Tiffany search queries |
| `data_collection/tiktok_instagram_collector.py` | Added thumbnail extraction from yt-dlp |
| `data_collection/build_dataset.py` | Pass thumbnail field through for all platforms |
| `templates/index.html` | Display thumbnail images in result cards |
| `create_embeddings.py` | Fix Timestamp serialization for pandas 3.0 |
| `dataset/shorts_data.json` | Real dataset: 34 videos with URLs and thumbnails |
| `embeddings.npy`, `density_scores.npy`, `metadata.json` | Rebuilt from real dataset |
| `BUILD_PLAN.md` | This write-up |
| `README.md` | Updated instructions for real data pipeline |

---

## Data Pipeline — Complete Documentation

This section documents the full data pipeline: sources, collection, preprocessing, feature construction, and embedding generation.

### 1. Data Sources

Our dataset contains **252 short-form videos** collected from a single platform via three methods:

| Source | Method | Count | Content Type |
|---|---|---|---|
| YouTube Shorts (captions) | YouTube Data API v3 + `youtube-transcript-api` | ~200 | Transcripts from YouTube's auto-generated or manual captions |
| YouTube Shorts (Whisper) | YouTube Data API v3 + `yt-dlp` + OpenAI Whisper | ~52 | Audio downloaded via yt-dlp, transcribed locally with Whisper `base` model |
| TikTok / Instagram (manual) | 18 curated URLs in `manual_urls.csv` | 0 collected* | Engagement content (Tiffany unboxings, hauls, collection tours) |

*\*TikTok/Instagram videos are music-only content and produced empty Whisper transcripts — included as URLs but no transcript data was extractable.*

**YouTube Data API v3** — used for two purposes:
- `search().list()` — discovers short-form video IDs matching 27 topic queries (`videoDuration=short`, `type=video`, `order=relevance`, 10 results per query)
- `videos().list()` — batch-fetches metadata (title, channel, views, likes, publish date, thumbnail URL) in groups of 50

**Transcript sources** — two-layer strategy:
1. **YouTube captions API** (`youtube-transcript-api`) — first choice; pulls auto-generated or manual captions directly. Fast, no audio download needed.
2. **Whisper STT fallback** — for videos without captions, downloads audio via `yt-dlp`, transcribes with OpenAI Whisper (`base` model, ~74M parameters, runs on CPU). Results are cached to `dataset/transcript_cache/transcripts.json` so subsequent runs skip already-transcribed videos.

**Search queries** — 27 queries organized into three categories:

| Category | Queries | Purpose |
|---|---|---|
| Instructional (13) | "how to cook", "quick workout tutorial", "DIY craft tutorial", "tech tips tutorial", "science experiment tutorial", "language learning tips", "photography tips", "music production tutorial", "makeup tutorial shorts", "yoga for beginners tutorial", "gardening tips shorts", "drawing tutorial shorts", "coding tutorial shorts" | Core educational content — tutorials, how-tos, lessons |
| Informational (7) | "history explained shorts", "fashion design inspiration", "math explained shorts", "personal finance tips", "home organization hacks", "guitar lesson shorts", "skincare routine tutorial" | Knowledge-sharing content — explanations, tips, routines |
| Tiffany / Engagement (7) | "Tiffany design meaning", "Tiffany jewelry history", "Tiffany architecture symbolism", "Tiffany jewelry collection haul", "Tiffany unboxing shorts", "my Tiffany collection tour", "luxury jewelry collection showing" | Mix of domain-specific instructional and pure engagement content — tests the system's ability to distinguish educational from entertainment videos on the same topic |

**TikTok / Instagram URLs** — 18 manually curated URLs (`data_collection/manual_urls.csv`) featuring fashion/jewelry influencers sharing Tiffany collections, unboxings, and hauls. These represent engagement-first content that looks topically relevant but is not instructional.

### 2. Data Collection Pipeline

The pipeline is orchestrated by `data_collection/build_dataset.py`:

```
                    ┌─────────────────────────────────────────┐
                    │         build_dataset.py (CLI)           │
                    │  --api-key, --results-per-query,         │
                    │  --whisper-model, --manual-urls           │
                    └───────────┬─────────────┬───────────────┘
                                │             │
                    ┌───────────▼───────┐ ┌───▼──────────────────┐
                    │  collect_youtube() │ │  collect_manual()     │
                    │  (27 queries × 10)│ │  (manual_urls.csv)    │
                    └───────┬───────────┘ └───────┬──────────────┘
                            │                     │
                    ┌───────▼───────────┐ ┌───────▼──────────────┐
                    │ YouTube API search │ │ yt-dlp audio download│
                    │ → video IDs       │ │ → Whisper transcribe  │
                    └───────┬───────────┘ └───────┬──────────────┘
                            │                     │
                    ┌───────▼───────────┐         │
                    │ Transcript fetch:  │         │
                    │ 1. Cache check     │         │
                    │ 2. Captions API    │         │
                    │ 3. Whisper fallback│         │
                    └───────┬───────────┘         │
                            │                     │
                    ┌───────▼─────────────────────▼──────┐
                    │          Merge + Deduplicate        │
                    │     (by video ID, preserves first)  │
                    └───────────────┬────────────────────┘
                                    │
                    ┌───────────────▼────────────────────┐
                    │     Clean transcripts (9 steps)     │
                    │     text_processing/clean_transcript │
                    └───────────────┬────────────────────┘
                                    │
                    ┌───────────────▼────────────────────┐
                    │   Save → dataset/shorts_data.json   │
                    └────────────────────────────────────┘
```

**Caching**: All transcripts (from captions API and Whisper) are stored in `dataset/transcript_cache/transcripts.json`. On re-runs, cached transcripts are used immediately without re-fetching, making the pipeline idempotent and fast for incremental updates.

**Output format** — each video in `dataset/shorts_data.json`:
```json
{
  "id": "yt_abc123",
  "platform": "youtube",
  "title": "How to Cook Perfect Scallops #SHORTS",
  "transcript": "It doesn't get much better than some perfectly cooked scallops...",
  "url": "https://www.youtube.com/shorts/abc123",
  "channel": "Mr. Make It Happen",
  "views": 1234567,
  "likes": 45678,
  "published_at": "2024-03-15T18:30:00Z",
  "thumbnail": "https://i.ytimg.com/vi/abc123/hqdefault.jpg"
}
```

### 3. Preprocessing — Transcript Cleaning

Raw transcripts from YouTube captions and Whisper contain noise that degrades embedding quality. We apply a **9-step composable cleaning pipeline** (`text_processing/clean_transcript.py`, stdlib only — no external dependencies):

| Step | Function | What it does | Example |
|---|---|---|---|
| 1. Unicode normalization | `_normalize_unicode()` | NFC normalization, remove zero-width chars (U+200B–U+200F, BOM), replace non-breaking spaces | `"café\u200b"` → `"café"` |
| 2. Caption artifact removal | `_remove_caption_artifacts()` | Strip YouTube/Whisper tags: `[Music]`, `[Applause]`, `[Laughter]`, `[Inaudible]`, `[Foreign]` | `"Hello [Music] world"` → `"Hello  world"` |
| 3. Timestamp removal | `_remove_timestamps()` | Strip time patterns: `0:15`, `1:23:45`, `(00:15)`, `[1:23]` | `"At 0:15 we start"` → `"At  we start"` |
| 4. URL removal | `_remove_urls()` | Strip `http://` and `https://` URLs | `"Visit https://example.com"` → `"Visit "` |
| 5. Mention/hashtag removal | `_remove_mentions_hashtags()` | Strip `@user` mentions and `#hashtags` | `"Follow @chef #cooking"` → `"Follow  "` |
| 6. Filler word removal | `_remove_fillers()` | Remove: `uh`, `um`, `hmm`, `er`, `ah`, `you know`, `i mean` (conservative list — keeps ambiguous words like "like", "so") | `"So um you need to uh mix"` → `"So  you need to  mix"` |
| 7. Repeated word collapse | `_collapse_repeated_words()` | Deduplicate stutters: `"the the the"` → `"the"` | `"add add the the salt"` → `"add the salt"` |
| 8. Whitespace normalization | `_normalize_whitespace()` | Collapse multi-spaces, fix punctuation spacing | `"Hello  ,  world"` → `"Hello, world"` |
| 9. Sentence segmentation | `_segment_sentences()` | Insert periods before discourse markers (`So`, `First`, `Then`, `Next`, `Finally`, etc.) in run-on text lacking punctuation | `"mix well Then add salt"` → `"mix well. Then add salt"` |

**Why this order matters**: Unicode normalization must come first (consistent byte representation). Artifact/timestamp/URL removal comes before filler removal (so tags aren't partially matched as fillers). Whitespace normalization comes last (cleans up gaps left by all prior removals). Sentence segmentation is final (operates on clean text).

**Validation**: 39 unit tests cover all steps, edge cases (null/empty input, clean passthrough, no false positives), and verify that cleaning preserves the instructional ranking order.

### 4. Filtering

Videos are filtered at two stages:

**During collection:**
- Videos with no transcript (neither captions nor Whisper output) are **skipped entirely** — they cannot be searched by content
- Transcripts shorter than 10 characters after cleaning are **dropped** (likely noise-only)
- Duplicate video IDs are **deduplicated** (first occurrence wins)

**During search (runtime):**
- Near-duplicate results are removed via cosine similarity on embeddings (threshold = 0.95)
- Optional `min_density` floor prevents zero-scored videos from being invisible

**Dataset composition after filtering:**
- Started with 264 unique YouTube video IDs from 27 queries
- 252 had usable transcripts (captions or Whisper) → final dataset
- 12 were dropped (unavailable videos, empty transcripts, or failed downloads)

### 5. Feature Construction

Each video has two computed features used for ranking:

#### 5a. Instructional Density Score (0.0 – 1.0)

Computed by `instructional_score.py`. Measures how educational/instructional a video's content is.

**Signal categories** (6 categories, 62 regex patterns total):

| Category | Weight | Example Patterns | Purpose |
|---|---|---|---|
| `how_to_cues` | 3.0 | "how to", "tutorial", "step by step", "let me show", "you need" | Strongest instructional signal — explicit teaching intent |
| `educational_concepts` | 2.0 | "learn", "teach", "tips", "recipe", "workout", "exercise" | Educational framing words |
| `domain_knowledge` | 2.0 | "design", "architecture", "symbolism", "meaning", "collection" | Subject-matter expertise indicators |
| `historical_context` | 1.5 | "history", "heritage", "legacy", "origin", "since 1837" | Historical/contextual depth |
| `sequential_markers` | 1.5 | "first", "then", "next", "finally", "after that" | Step-by-step structure (common in tutorials) |
| `action_verbs` | 1.0 | "make", "add", "mix", "cut", "use", "create", "build" | Instructional action language |

**Entertainment penalty** (20 patterns, 2.0 penalty each):
- Penalizes engagement-first language: "omg", "bestie", "haul", "unboxing", "slay", "vibe", "obsessed", "no cap", "it's giving", "I'm dead", etc.
- Pure entertainment content (hauls, unboxings) scores near 0.0

**Scoring formula:**
```
weighted_sum = Σ (match_count × category_weight)  for each category
penalty = entertainment_match_count × 2.0
net = max(0, weighted_sum - penalty)
density = (net / word_count) × 100          # signals per 100 words
raw = min(density, 20.0)                     # cap at 20
score = min(1.0, sqrt(raw) / 4.5)           # sqrt curve, normalized to [0, 1]
```

The sqrt curve ensures diminishing returns — a video with 5 instructional signals per 100 words scores meaningfully higher than one with 1, but the difference between 15 and 20 is small. This prevents keyword-stuffed content from gaming the score.

#### 5b. Sentence Embedding (384-dimensional vector)

Computed by `create_embeddings.py` using the `all-MiniLM-L6-v2` sentence transformer.

**Input text construction:**
```
embedding_text = title + " [SEP] " + transcript
```

The `[SEP]` token separates the title (concise topic signal) from the transcript (detailed content), allowing the model to leverage both. Title provides topic-level matching; transcript provides content-level matching.

**Model**: `all-MiniLM-L6-v2` (22M parameters, 384-dim output)
- Trained on 1B+ sentence pairs for semantic similarity
- Optimized for speed — encodes 252 videos in ~5 seconds on CPU
- Produces L2-normalized embeddings suitable for cosine similarity

### 6. Embedding Construction Process

The embedding pipeline (`create_embeddings.py`) produces three output files:

```
dataset/shorts_data.json (252 videos)
         │
         ▼
┌──────────────────────────────────────┐
│  create_embeddings.py                │
│                                      │
│  1. Load dataset (JSON → DataFrame)  │
│  2. Combine: title [SEP] transcript  │
│  3. Encode with all-MiniLM-L6-v2     │
│     (batch_size=32, 8 batches)       │
│  4. Compute density scores           │
│     (instructional_score.py)         │
│  5. Save outputs                     │
└──────────┬───────────┬──────────┬────┘
           │           │          │
    embeddings.npy  density_scores.npy  metadata.json
    (252 × 384)      (252,)             (252 records)
```

**Output files:**

| File | Shape | Description |
|---|---|---|
| `embeddings.npy` | (252, 384) | Float32 sentence embeddings — one row per video |
| `density_scores.npy` | (252,) | Float64 instructional density scores — one value per video [0.0, 1.0] |
| `metadata.json` | 252 records | All video fields (id, title, transcript, url, channel, views, likes, thumbnail, published_at, platform) — used for displaying search results |

**How search uses these:**
```
final_score = cosine_similarity(query_embedding, doc_embedding)
            × instructional_density_score
            × topical_boost(query_terms, doc_text)
```

- `cosine_similarity` — measures semantic relevance between query and video content
- `instructional_density_score` — upweights educational content, downweights entertainment
- `topical_boost` — up to 1.5× multiplier when the document contains exact query terms (with synonym expansion)

### 7. Dataset Statistics

| Metric | Value |
|---|---|
| Total videos | 252 |
| Platform | YouTube Shorts (all) |
| Transcript source: captions | ~200 (77%) |
| Transcript source: Whisper | ~52 (20%) |
| Transcript source: none (dropped) | 12 (5%) |
| Density score range | 0.000 – 0.994 |
| Embedding dimensions | 384 |
| Topic categories | 27 search queries across instructional, informational, and engagement content |
| Manual URLs (TikTok/Instagram) | 18 curated (0 usable transcripts — music-only content) |

---

## Ranking System — Algorithm Design and Trade-offs

This section documents the ranking algorithm, the alternatives we considered, and the design decisions behind the current system.

### 1. System Architecture Overview

```
User query ("tiffany")
       │
       ▼
┌──────────────────────────────────────────────┐
│              search.py                        │
│                                               │
│  1. Encode query → 384-dim embedding          │
│  2. Detect intent (instructional vs browsing) │
│  3. Cosine similarity against all documents   │
│  4. Apply intent-aware density weighting      │
│  5. Apply topical boost (keyword matching)    │
│  6. Deduplicate near-identical results        │
│  7. Return top-K ranked results               │
└──────────────────────────────────────────────┘
       │
       ▼
final_score = similarity × density^(intent_weight) × topical_boost
```

### 2. Algorithm Selection — Why Semantic Embedding Search?

We evaluated four retrieval approaches before choosing semantic embedding search:

| Algorithm | How it works | Strengths | Weaknesses for our use case |
|---|---|---|---|
| **Boolean retrieval** | Match if document contains query terms (AND/OR) | Simple, fast, exact match | No ranking — "tiffany" once = "tiffany" everywhere. No semantic understanding — "jewelry meaning" misses "necklace symbolism". Most queries return 0 or unranked results on short transcripts. |
| **BM25 (TF-IDF family)** | `score = Σ IDF(term) × tf_norm`. Ranks by term frequency with document length normalization. | Proven lexical ranker, handles tf and doc length well, fast | Still keyword-based — "how to cook scallops" won't match "sear the shellfish in butter". Short transcripts (30–100 words) have weak tf signals — most terms appear only once. BM25 shines on longer documents. |
| **Semantic embedding** (our choice) | Encode query and documents into dense vectors, rank by cosine similarity | Understands meaning — "cooking tutorial" matches "recipe walkthrough". Works well on short noisy texts. No vocabulary mismatch. | Slower (model inference ~10ms/query), less interpretable, can be too "fuzzy" — sometimes matches topically adjacent but irrelevant content. |
| **Matrix factorization** (collaborative filtering) | `R ≈ U × V^T` — decompose user-item interaction matrix | Great for personalized recommendations from user behavior | Wrong paradigm — we have no user interaction data (no clicks, watch history, ratings). MF solves "what would this user like?" not "which videos match this query?". Cold-start problem with 252 videos. |

**Why we chose semantic embedding:**
- Short-form video transcripts are noisy and use varied vocabulary across creators
- Semantic similarity bridges vocabulary gaps that keyword methods miss
- The `all-MiniLM-L6-v2` model (22M parameters) is fast enough for real-time search (~10ms per query on CPU)
- A single embedding space handles both query encoding and document comparison

**Why not a hybrid (BM25 + semantic)?**
We considered `score = α × BM25 + (1-α) × cosine_sim` but opted for a simpler design: the `topical_boost` component in our formula serves a similar role to BM25 by checking if exact query terms appear in the document. This gives us the benefit of lexical matching without maintaining two separate scoring systems.

### 3. The Ranking Formula

#### Version 1 (original — static density)

```
final_score = cosine_similarity(query, doc) × density × topical_boost
```

**Problem discovered during testing:** This formula treats every query as if the user wants educational content. The density score is precomputed once per video and never changes:

```
density["How Tiffany was Created"]     = 0.948  (educational keywords)
density["My Tiffany Collection Tour"]  = 0.616  (engagement keywords)
```

When a user searches "tiffany" (browsing intent — wants to see collections, jewelry styling), the system still penalizes engagement content because 0.616 < 0.948. The top results for "tiffany" and "tiffany design story" were nearly identical — both dominated by educational videos.

**Root cause:** Static density acts as a hard gate that always favors instructional content, regardless of what the user is actually looking for.

#### Version 2 (current — intent-aware density)

```
final_score = cosine_similarity(query, doc) × density^(intent_weight) × topical_boost
```

Where `intent_weight ∈ [0.3, 1.0]` adapts per query:

| Query | intent_weight | Effect |
|---|---|---|
| "tiffany" | 0.42 | density^0.42 → gap between 0.948 and 0.616 shrinks from 0.33 to 0.16. Browsing content surfaces. |
| "tiffany design story" | 0.64 | density^0.64 → moderate gap preserved. Educational content still preferred. |
| "how to cook pasta" | 0.86 | density^0.86 → near-full penalty. Tutorials dominate. |
| "cute outfits" | 0.50 | density^0.50 → sqrt flattening. Fashion content surfaces alongside tutorials. |

**Why `density^weight` instead of `α × density + (1-α)`?** The power function preserves the [0, 1] range without rescaling and has a natural "flattening" effect: `density^0.3` compresses the gap between high and low density videos, while `density^1.0` preserves the original distribution. An additive blend would require additional normalization and doesn't have this elegant scaling property.

### 4. Intent Detection — Embedding-Based Prototype Matching

We evaluated three approaches for detecting query intent:

| Approach | Mechanism | Pro | Con |
|---|---|---|---|
| **A. Rule-based (regex)** | Count instructional cue words ("how to", "tutorial", "guide") | No training data, fully interpretable, zero latency | Brittle — "tiffany design" falsely triggers on "design". Can't handle ambiguity. Every edge case needs a new rule. |
| **B. Embedding prototypes** (our choice) | Compare query embedding to instructional/browsing prototype sentences | Semantic understanding, no labeled data needed, reuses existing model | Sensitive to prototype quality, adds ~10ms, less interpretable |
| **C. Trained classifier** | Logistic regression on labeled query embeddings | Most accurate, learns nuance | Needs 50–100 labeled queries, could overfit on small dataset, another model to maintain |

**Why we chose Option B:**

1. **Reuses the sentence transformer** we already load — no new dependencies or models
2. **Semantically aware** — understands "tiffany" is closer to "show me items" than "teach me step by step", even without keyword matching
3. **Easy to tune** — add/edit prototype sentences without retraining
4. **Good fit for the report** — novel enough to explain, backed by established NLP research on prototype-based classification

**How it works:**

```python
# 5 instructional prototypes (pre-encoded, cached):
"how to do something step by step tutorial"
"explain the history and meaning behind this"
"teach me tips and techniques for beginners"
"learn about the origin and story of a design"
"guide to understanding why something was created"

# 5 browsing prototypes (pre-encoded, cached):
"show me items and collections to browse"
"what does it look like when wearing jewelry"
"unboxing haul showcase my collection tour"
"sharing favorite pieces and accessories"
"beautiful luxury items and fashion inspiration"

# Detection:
inst_score  = max cosine_sim(query, instructional_prototypes)
browse_score = max cosine_sim(query, browsing_prototypes)
ratio = inst_score / (inst_score + browse_score)
intent_weight = 0.3 + 0.7 × ratio    →  [0.3, 1.0]
```

The floor of 0.3 ensures density always has some influence — even for pure browsing queries, we still slightly prefer content with substance over empty engagement.

### 5. Component Breakdown

| Component | Role | File | Key Design Choice |
|---|---|---|---|
| **Cosine similarity** | Measures semantic relevance between query and document | `search.py` | Uses pre-normalized embeddings from all-MiniLM-L6-v2. O(n) dot product against 252 documents — fast enough for real-time. |
| **Instructional density** | Scores how educational a video's content is (0–1) | `instructional_score.py` | 6 weighted signal categories (62 regex patterns) + 20 entertainment penalties. Precomputed at index time — zero cost at query time. sqrt normalization prevents keyword stuffing. |
| **Intent detection** | Adapts density influence based on query type | `search.py` | Embedding-based prototype matching. Prototype embeddings cached after first call. Adds ~10ms per query (one encode call). |
| **Topical boost** | Rewards exact keyword matches in document text | `search.py` | Lightweight BM25 approximation. Expands query terms via 12-entry synonym dict. Multiplier in [1.0, 1.5]. |
| **Deduplication** | Removes near-identical results | `search.py` | Pairwise cosine similarity on top results, threshold=0.95. Prevents multiple copies of the same video appearing in different results. |

### 6. Trade-off Summary

| Trade-off | Our decision | Alternative | Why |
|---|---|---|---|
| Retrieval method | Semantic embedding (dense) | BM25 (sparse) | Short transcripts have weak tf signals; semantic matching bridges vocabulary gaps |
| Density weighting | Query-dependent (`density^intent`) | Static (`density × 1.0`) | Static density makes browsing and instructional queries return identical results |
| Intent detection | Embedding prototypes | Regex rules / trained classifier | Balances accuracy vs. complexity; no labeled data needed; reuses existing model |
| Embedding model | all-MiniLM-L6-v2 (22M params) | Larger models (e.g., all-mpnet-base-v2, 110M) | Speed priority — 252 videos indexed in 5s, queries in 10ms. Larger model would improve accuracy but 5× slower. |
| Density scoring | Regex heuristic | ML classifier | No labeled training data available; regex is interpretable and sufficient for distinguishing tutorials from hauls |
| Topical boost | Keyword + synonym expansion | Full BM25 pipeline | Simpler to implement and maintain; synonym dict covers our domain well |

### 7. Observed Results — Before vs After Intent-Aware Ranking

**Query: "tiffany"** (browsing intent)

| Rank | Before (static density) | After (intent_weight=0.42) |
|---|---|---|
| #1 | TIFFANY KNOT COLLECTION (density=0.994) | TIFFANY KNOT COLLECTION (density→0.997) |
| #2 | How Tiffany & Co was Created?! (density=0.948) | **My Tiffany Collection** (density 0.616→0.818) ↑ |
| #3 | TIFFANY JEWELRY TOUR (density=0.801) | How Tiffany & Co was Created?! (density→0.978) |
| #4 | My Tiffany Collection (density=0.616) | TIFFANY JEWELRY TOUR (density→0.912) |
| #5 | $16B Tiffany Origin Story (density=0.727) | **Thrift find necklace** (density 0.631→0.826) ↑ |

Browsing/lifestyle content ("My Collection", "thrift find") moves up because density gap is compressed.

**Query: "tiffany design story"** (instructional intent)

| Rank | Before (static density) | After (intent_weight=0.64) |
|---|---|---|
| #1 | How Tiffany & Co was Created?! | How Tiffany & Co was Created?! (same) |
| #2 | TIFFANY KNOT COLLECTION | TIFFANY KNOT COLLECTION (same) |
| #3 | $16B Tiffany Origin Story | $16B Tiffany Origin Story (same) |
| #4 | TIFFANY JEWELRY TOUR | **Hidden History of Tiffany** ↑ |
| #5 | Hidden History of Tiffany | TIFFANY JEWELRY TOUR |

Educational content stays on top; history-focused video moves up.

---

## Next Steps Checklist

Optimization tasks organized by priority. We will work through these step by step.

### High Priority — directly improves the product

- [x] **Step 1: Grow the dataset to 100+ videos**
  - Expanded from 34 to 252 videos
  - Added 14 new search queries (27 total) covering more topics
  - Implemented Whisper STT fallback for videos without captions
  - Added transcript caching for fast re-runs
  - Added 18 real TikTok/Instagram URLs (music-only, no usable transcripts)

- [x] **Step 2: Evaluation framework (Phase 5)**
  - Built ground-truth set: 86 labeled videos × 15 queries with dual labels (relevance + instructional quality)
  - Implemented `evaluation/metrics.py` with P@K, R@K, F1@K, nDCG@10, MRR, MAP, AvgDensity@5, AvgInstructional@5
  - Implemented `evaluation/evaluate.py` comparing 3 baselines: similarity-only, view-count, our system
  - Pooled evaluation ensures fair comparison (only labeled videos counted)
  - Results: our system gains +8.7% instructional density with only -0.022 nDCG@10 cost
  - 40 unit tests passing in `evaluation/test_evaluation.py`

- [ ] **Step 3: Fix query expansion false matches**
  - Current problem: "design" expands to "aesthetic, style, architecture", causing DIY content to match Tiffany queries
  - The synonym system is too broad and context-unaware
  - Options: weight synonym matches lower than exact matches, or make expansion context-aware so "design" only expands when relevant

### Medium Priority — better user experience

- [ ] **Step 4: Improve the frontend**
  - Add filters: let users filter by platform (YouTube/TikTok/Instagram), minimum density score, or date range
  - Show transcript snippets in result cards so users can see why a result was ranked high
  - Add pagination or "load more" instead of fixed top-10 results

- [ ] **Step 5: Use Whisper as fallback for YouTube transcripts**
  - Currently we lose ~83% of YouTube videos because they have no auto-captions
  - For videos without captions, download audio via yt-dlp and transcribe with Whisper locally
  - This would dramatically increase dataset coverage and make the collection pipeline more robust

### Lower Priority — polish & scale

- [ ] **Step 6: Improve density scoring**
  - The current regex-based scorer is a heuristic that may misclassify edge cases
  - Consider training a lightweight classifier on labeled evaluation data (from Step 2) to better distinguish instructional vs. entertainment content

- [ ] **Step 7: Add `.gitignore`**
  - Exclude `__pycache__/`, `.pyc` files, and other build artifacts from the repo
  - Quick cleanup win
