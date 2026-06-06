import os
# 配置huggingface国内镜像，解决WinError10060超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer, util

# 自动从镜像拉取bge-base-zh-v1.5
model = SentenceTransformer("BAAI/bge-base-zh-v1.5")

q = model.encode("保险理赔", normalize_embeddings=True)
docs = ["理赔流程说明", "索赔方法介绍", "等待期是多久"]
d = model.encode(docs, normalize_embeddings=True)

# 余弦相似度排序输出
scores = util.cos_sim(q, d)[0].tolist()
for doc, score in sorted(zip(docs, scores), key=lambda x: -x[1]):
    print(f"cos={score:.2f} {doc}")
