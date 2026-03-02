# Short Compilation — Complete Build Plan

**Project:** Cross-platform search engine for short-form videos ranked by instructional density  
**Course:** INFO 376, University of Washington  
**Team:** Haohang Li, Matthew Miller-Martinez, Nathaniel Wiroatmodjo, William Tse Yuen 

---

## Current State

| Module | Status | File |
|---|---|---|
| Indexing (embeddings) | Done | `create_embeddings.py` — all-MiniLM-L6-v2 sentence transformer |
| Scoring (density heuristic) | Done | `instructional_score.py` — regex-based keyword scoring |
| Search (ranking pipeline) | Done | `search.py` — cosine similarity x density x topical boost |
| Ranking test | Done | `test_ranking.py` — proves instructional > entertainment |
| Data collection | **Done + Tested** | `data_collection/` — YouTube API + TikTok/Instagram via yt-dlp + Whisper |
| Data collection tests | **Done (40/40 pass)** | `data_collection/test_phase1.py` — mocked unit tests for all 4 modules |
| Text processing | **Done + Tested** | `text_processing/` — 9-step cleaning pipeline, 39/39 tests passing |
| Real dataset | **Ready to generate** | Run `build_dataset.py` with API key to produce 50-100+ videos |
| Web UI | **Missing** | No frontend at all |
| Evaluation framework | **Missing** | No systematic metrics (precision, recall, nDCG) |

---

## Phase 1: Data Collection Module

### Task 1.1 — YouTube Data API Client
- **File:** `data_collection/youtube_api.py`
- **Status:** [x] Done
- `search_shorts(api_key, query, max_results)` — searches YouTube with `videoDuration=short`, `type=video`
- `get_video_details(api_key, video_ids)` — batch-fetches title, description, view/like counts, channel, publish date, URL
- 10 predefined search queries across diverse topics (cooking, fitness, DIY, tech, history, fashion, science, language, photography, music)
- Each query fetches ~10 results → ~80-100 YouTube videos total

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
- Cleans up temp audio files after transcription
- Outputs standard `{id, platform, title, transcript, ...}` format

### Task 1.4 — Data Pipeline Script
- **File:** `data_collection/build_dataset.py`
- **Status:** [x] Done
- CLI orchestrator: `python data_collection/build_dataset.py --api-key YOUR_KEY`
- Runs YouTube search across all topics → fetches transcripts → processes manual URLs → deduplicates → saves to `dataset/shorts_data.json`
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
- **File:** `app.py` (Flask or FastAPI)
- **Status:** [ ] Not started
- `GET /api/search?q=...&k=10` — returns ranked JSON results
- `GET /api/stats` — dataset stats (video count, avg density, etc.)
- Load model + embeddings + metadata on startup

### Task 4.2 — Frontend
- **File:** `templates/index.html` (or `frontend/`)
- **Status:** [ ] Not started
- Search bar with query input
- Results list showing: rank, title, platform, density score, similarity score, video link/embed
- Click-through to the original video on YouTube/TikTok/Instagram
- Simple, clean design (single-page HTML+JS or React)

---

## Phase 5: Evaluation

### Task 5.1 — Evaluation Framework
- **File:** `evaluation/evaluate.py`
- **Status:** [ ] Not started
- Define ground truth: for 10-15 test queries, manually label which videos are "relevant + instructional"
- Compute metrics: Precision@K, nDCG@10, Mean Reciprocal Rank (MRR)
- Compare rankings: our system vs. a baseline (pure cosine similarity without density scoring)
- Generate evaluation report table

### Task 5.2 — A/B Baseline Comparison
- **File:** `evaluation/compare_baselines.py`
- **Status:** [ ] Not started
- Baseline 1: cosine similarity only (no density)
- Baseline 2: view-count ranking (engagement-based)
- Our system: similarity x density x topical boost
- Show that our system surfaces instructional content better

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

| Priority | Task | Reason |
|---|---|---|
| 1 | Data collection (YouTube API + transcripts) | Everything depends on having real data |
| 2 | Text processing (clean transcripts) | Embedding quality depends on clean input |
| 3 | Re-run indexing on real dataset | Generate real embeddings + density scores |
| 4 | Web UI (Flask app + HTML) | The proposal requires a searchable interface |
| 5 | Evaluation framework | Week 8 milestone — need metrics for the report |
| 6 | Enhancements to scoring/search | Polish after core pipeline works end-to-end |

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
