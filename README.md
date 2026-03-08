# Instructional-Density-Search-Engine-for-short-videos

A cross-platform search engine for short-form videos ranked by instructional density.

## How to run this project

### 1. Activate the virtual environment
```
source .venv/bin/activate
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Collect real video data (requires YouTube API key)
```
python data_collection/build_dataset.py --api-key YOUR_YOUTUBE_API_KEY
```
This searches YouTube across 13 topic queries (cooking, DIY, tech, Tiffany, etc.), fetches transcripts, and saves to `dataset/shorts_data.json`. Each video record includes title, transcript, URL, thumbnail, view/like counts, and more.

To add TikTok/Instagram videos, edit `data_collection/manual_urls.csv` with real URLs and re-run.

### 4. Generate embeddings and density scores
```
python create_embeddings.py --input dataset/shorts_data.json
```
This creates `embeddings.npy`, `density_scores.npy`, and `metadata.json` for the search engine.

### 5. Run searches (CLI)
```
python search.py "tiffany design meaning"
python search.py "how to cook steak"
```

### 6. Run the web UI
```
python app.py
```
Open `http://localhost:5001` in your browser. Search results show thumbnails, platform badges, density scores, and link to the original videos.

### 7. Run evaluation
```
python evaluation/evaluate.py
```
This compares 3 ranking strategies against ground-truth relevance labels:
- **Similarity only** — pure cosine similarity (no density weighting)
- **View-count** — sort by popularity (view count descending)
- **Our system** — similarity × density^intent × topical boost

The evaluation uses **pooled evaluation** (only the 86 labeled videos are counted in metrics) with **dual labels** (relevance 0-3 and instructional quality 0-3) across 15 test queries.

**Metrics computed:** Precision@5, Recall@5, F1@5, nDCG@10, MRR, MAP, AvgDensity@5, AvgInstructional@5

**Key results:**
| Metric | Similarity Only | Our System | Interpretation |
|---|---|---|---|
| nDCG@10 | 0.871 | 0.850 | Minimal relevance cost (-2.4%) |
| MRR | 0.933 | 0.933 | First relevant result at same position |
| AvgDens@5 | 0.746 | 0.811 | +8.7% more instructional content surfaced |
| AvgInstr@5 | 2.573 | 2.640 | Higher human-rated instructional quality |

Our system trades a small relevance cost for meaningful gains in instructional density, confirming that the density-weighted ranking successfully surfaces more educational content.

You can also access evaluation results via the API at `http://localhost:5001/api/evaluate`.

### 8. Run tests
```
python -m pytest data_collection/test_phase1.py -v    # 40 tests — data collection
python -m pytest text_processing/test_phase2.py -v    # 39 tests — text processing
python -m pytest test_phase3.py -v                     # 30 tests — scoring & search
python -m pytest test_phase4.py -v                     # 15 tests — web UI
python -m pytest evaluation/test_evaluation.py -v      # 40 tests — evaluation metrics & ground truth
```

