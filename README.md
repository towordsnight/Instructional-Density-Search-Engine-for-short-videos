# Instructional-Density-Search-Engine-for-short-videos

```
# 1. create embeddings and density scores
python create_embeddings.py -i sample_shorts_data.json

# 2. search (calculate similarity and density)
python search.py "how to do ab workout"
python search.py "cooking noodles" -k 5
```