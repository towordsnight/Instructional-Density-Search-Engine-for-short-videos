import numpy as np
from numpy.linalg import norm

emb = np.load("embeddings.npy")

# 余弦相似度
def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# 第 1 和第 2 个视频的相似度
sim = cosine_sim(emb[0], emb[1])
print(f"视频 0 和 1 的相似度: {sim:.4f}")  # 越接近 1 越相似