# Instructional-Density-Search-Engine-for-short-videos

```
# 1.install the env
source .venv/bin/activate
pip install -r requirements.txt

# 2. generate embeddings and density scores with Tiffany data samples
python create_embeddings.py -i tiffany_sample_shorts_data.json -o embeddings.npy --density-output density_scores.npy --metadata-output metadata.json

# 3. query
python search.py "Design inspiration of Tiffany collections"
python search.py "architectural meaning of Tiffany collections"
python search.py "history of Tiffany"

# 4.run the script
python test_ranking.py
```

