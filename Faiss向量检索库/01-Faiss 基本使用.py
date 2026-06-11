import faiss

# 1.基本操作
def test01():
    # 创建索引（数据库）
    # 参数用来指定存储的向量维度
    index = faiss.IndexFlatL2(256)  # Flat线性搜索，  L2表示使用相似度计算是：欧式距离
    index = faiss.IndexFlatIP(256)  # Flat线性搜索，  IP表示使用相似度计算是：点积相似度
    