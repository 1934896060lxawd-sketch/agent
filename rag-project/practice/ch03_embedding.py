"""
第三章：向量嵌入（Embedding）语义检索

回答一个问题：关键词匹配不到的（"便宜" vs "经济实惠"），怎么搜？

核心思路：
  文本 → BGE 模型 → 768 维向量 → 意思相近的文本向量距离近

你要实现的核心功能：
  1. 加载 BGE Embedding 模型（本地路径优先，网络不通也能跑）
  2. 把文档批量编码成向量（normalize 让点积 = 余弦相似度）
  3. 用向量余弦相似度检索，替代第一章的关键词匹配
  4. 对比实验：同一个 query，关键词检索 vs 语义检索结果并排展示

依赖：pip install sentence-transformers numpy modelscope torch transformers
（复用第一章的 load_data / search，所以 ch01 也要能跑）
"""

import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

# 复用第一章的函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# TODO: from ch01_naive_rag import load_data, search as keyword_search
# （练习时可以把 ch01 和 ch03 放在同一个目录）

# ============================================================
# 模块级初始化：加载 Embedding 模型
# ============================================================

# TODO: 1. 构造本地模型路径（practice/../models/bge-base-zh-v1.5/BAAI/bge-base-zh-v1___5）
# TODO: 2. 如果本地路径存在 → SentenceTransformer(本地路径, prompts={})
# TODO: 3. 如果本地不存在 → 设置 HF_ENDPOINT="https://hf-mirror.com"，从 HuggingFace 加载
#        SentenceTransformer("BAAI/bge-base-zh-v1.5", prompts={})
#
# 注意：
#   - prompts={} 是 sentence-transformers 5.x 兼容必需的，禁用自动 prompt 模板
#   - 模型约 390MB，首次加载需要下载


# ============================================================
# 1. 批量向量化
# ============================================================

def embed_documents(documents: list[dict]) -> list[dict]:
    """
    给每条文档的 content 生成 embedding 向量，直接写回 dict 的 "embedding" 字段。

    流程：
      ① 提取所有文档的 content → texts 列表
      ② _embed_model.encode(texts, normalize_embeddings=True)
         → 得到 shape=(N, 768) 的 numpy 数组
      ③ 逐条写回 doc["embedding"] = vec

    为什么 normalize_embeddings=True？
      归一化后向量模长 = 1，两个向量的点积直接等于余弦相似度
      不需要每次检索时除以模长，省计算

    返回：写入了 embedding 字段的 documents（原地修改 + 返回）
    """
    # TODO: 实现
    pass


# ============================================================
# 2. 语义检索
# ============================================================

def semantic_search(query: str, documents: list[dict], top_k: int = 3) -> str:
    """
    用向量余弦相似度检索，替代第一章的关键词匹配。

    流程：
      ① query 编码成向量（注意：encode() 必须传 list，v5.x 不接受单字符串）
         q_vec = _embed_model.encode([query], normalize_embeddings=True)[0]
      ② 遍历所有文档，算 q_vec 和 doc["embedding"] 的点积（=余弦相似度）
      ③ 按相似度降序排序，取 top_k
      ④ 拼成上下文文本返回

    返回格式（和第一章 search() 一致）：
      【来源：xxx】(相似度：0.6438)
      文档内容...

    提示：np.dot(vec_a, vec_b) 算两个向量的点积
    """
    # TODO: 实现
    pass


# ============================================================
# 3. 对比实验
# ============================================================

if __name__ == "__main__":
    """
    流程：
      ① 确定 data_dir 路径（practice/../data）
      ② 调用 load_data(data_dir) 加载文档
      ③ 调用 embed_documents(documents) 向量化（启动时跑一次）
      ④ 打印加载数量

      ⑤ while 循环：
          - 读取用户输入（EOFError/KeyboardInterrupt 处理退出）
          - q 退出，空输入跳过
          - 调用 keyword_search(query, documents) 获取关键词结果
          - 调用 semantic_search(query, documents) 获取语义结果
          - 两栏并排打印 Top-3（分别只打印来源 + 前 150 字预览）

    示例输出对比：
      查询："大空间SUV"

      【关键词检索 Top-3】
        来源：report_05_buying_guide  购车指南...
        来源：极氪 7X                  极氪7X 大型纯电豪华SUV...
        来源：比亚迪 海狮08            比亚迪海狮08 中大型混动...

      【语义检索 Top-3】
        来源：理想 L8    (0.6438)    理想L8 中大型增程家用SUV...
        来源：AITO M7    (0.6426)    AITO M7 中大型增程SUV...
        来源：理想 L6    (0.6421)    理想L6 中型增程SUV...
    """
    # TODO: 实现
    pass
