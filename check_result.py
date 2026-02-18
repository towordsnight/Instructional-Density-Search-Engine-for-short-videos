import numpy as np
import json

# 加载 embeddings
emb = np.load("embeddings.npy")
print(f"✓ Embeddings: {emb.shape} (样本数 x 384维)")

# 加载 metadata
with open("metadata.json") as f:
    meta = json.load(f)
print(f"✓ Metadata: {len(meta)} 条记录")
print(f"  第 1 条: {meta[0]}")