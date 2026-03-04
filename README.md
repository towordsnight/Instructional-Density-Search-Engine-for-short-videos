# Instructional-Density-Search-Engine-for-short-videos

Here's how to run this project:                                               
## 1. Activate the virtual environment                                                                                                   
```
  source .venv/bin/activate
```                                                 

## 2. Install dependencies
```
  pip install -r requirements.txt
```
                                                                                
## 3. Generate embeddings and density scores                                     
```
  python create_embeddings.py -i tiffany_sample_shorts_data.json -o             
  embeddings.npy --density-output density_scores.npy --metadata-output          
  metadata.json
```
  This processes the sample data and creates the embedding/scoring files needed
  for search.

## 4. Run searches (CLI)
```
  python search.py "Design inspiration of Tiffany collections"
  python search.py "history of Tiffany"
```

## 5. Run the web UI
```
  python app.py
```
  Then open `http://localhost:5000` in your browser. It provides a search interface with /api/search?q=... and /api/stats endpoints.

## 6. Run tests
```
  pytest test_ranking.py test_phase3.py test_phase4.py
```

