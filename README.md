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
Open `http://localhost:5000` in your browser. Search results show thumbnails, platform badges, density scores, and link to the original videos.

### 7. Run tests
```
python -m pytest data_collection/test_phase1.py -v    # 40 tests — data collection
python -m pytest text_processing/test_phase2.py -v    # 39 tests — text processing
python -m pytest test_phase3.py -v                     # 30 tests — scoring & search
python -m pytest test_phase4.py -v                     # 15 tests — web UI
```

