"""
第五章：完整 RAG Agent —— 集成前四章全部优点

管线全貌（一次查询走完这 6 步）：
  ① Query 改写      — LLM 把省略/模糊的问题扩展成完整检索查询
  ② 混合检索        — FAISS 向量 + BM25 关键词 → RRF 融合（第四章）
  ③ 元数据过滤      — 价格区间/车型/类别等结构化字段硬过滤
  ④ Reranker 精排   — CrossEncoder 对候选文档和 query 算真正相关性
  ⑤ 上下文拼接      — Top-K 文档 + 对话历史 → 结构化 prompt
  ⑥ LLM 生成        — 带引用标注的回答

前四章 vs 第五章：
  第一章：关键词检索 → LLM
  第二章：三种分块策略对比
  第三章：Embedding + 暴力遍历
  第四章：FAISS + BM25 + RRF 混合检索
  第五章：上面所有 + Reranker + Query 改写 + 多轮对话 + 元数据过滤
"""

import os
import sys
import math
import jieba
import numpy as np
import faiss
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from naive_rag import load_data, build_prompt, ask_llm
from embedding_test import _embed_model, embed_documents
from retrieval_test import VectorIndex, BM25, hybrid_rrf


# ============================================================
# 1. Reranker — CrossEncoder 精排
# ============================================================

class Reranker:
    """
    为什么需要 Reranker？
      向量检索：doc 和 query 分别编码，然后算余弦 — 速度快但精度有损（双塔模型）
      Reranker：  doc 和 query 拼在一起编码，算真正的相关性 — 速度慢但精度高（交叉编码器）

    管线位置：粗排（混合检索 Top-K×2）→ 精排（Reranker）→ 取 Top-K 喂 LLM
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        BGE Reranker 和 BGE Embedding 同一个团队出品，中文效果好。
        本地加载优先（和第三章模型加载逻辑一样）。
        """
        # TODO: 先检查本地路径，再从 HF 加载
        # 提示：from sentence_transformers import CrossEncoder
        ...

    def rerank(self, query: str, candidates: list[dict], top_k: int = 3
               ) -> list[tuple[float, dict]]:
        """
        对候选文档逐个和 query 算交叉编码分数。
        输入：混合检索返回的候选文档列表（建议 top_k*2）
        输出：按相关性精排后的 top_k
        """
        # TODO:
        #   1. 构造 [[query, doc["content"]], ...] pairs
        #   2. self.model.predict(pairs) 得到分数
        #   3. 排序取 top_k
        ...


# ============================================================
# 2. Query 改写 — 把模糊问题展开
# ============================================================

QUERY_REWRITE_SYSTEM = """你是一个查询改写助手。结合对话历史，把用户问题改写成独立的、信息完整的检索查询。

规则：
- 如果用户问题包含指代词（"它""那个""上次说的"），替换成具体实体
- 如果用户问题过于简短（<5个字），根据历史补全上下文
- 如果用户问题已经完整，直接原样返回
- 只输出改写后的查询，不要解释"""


def rewrite_query(query: str, chat_history: list[dict]) -> str:
    """
    把依赖上下文的问题改写成独立查询，消除指代歧义。

    例子：
      chat_history: "推荐一款大空间家用SUV" → "理想L6，24.98万起"
      query:        "那它的续航呢"
      rewritten:   "理想L6的续航里程"

    为什么放在检索之前？
      Embedding 模型不理解"它"指谁，必须先消解指代再向量化。
    """
    # TODO:
    #   1. 如果是第一轮对话（chat_history 为空），直接返回 query
    #   2. 否则构造 messages = [system prompt] + 历史 + 当前问题
    #   3. 调 ask_llm() 获取改写结果，返回
    ...


# ============================================================
# 3. 元数据过滤 — 结构化字段硬约束
# ============================================================

def extract_filters(query: str) -> dict:
    """
    从查询中提取结构化过滤条件。

    例子：
      "25万以内的SUV" → {"max_price": 25, "category": "SUV"}
      "纯电轿车"      → {"powertrain": "纯电", "category": "轿车"}
      "小米SU7"       → {"brand": "小米", "model": "SU7"}

    实现思路：用规则匹配（jjieba 分词 + 关键词字典），不调 LLM（太慢）。
    进阶：用 LLM 做 few-shot 提取，准确率更高但多一次 API 调用。
    """
    # TODO:
    #   1. 准备价格关键词字典：{"万以内": "max_price", "万以上": "min_price", ...}
    #   2. 准备类别关键词字典：{"SUV": "SUV", "轿车": "轿车", ...}
    #   3. 准备动力类型字典：{"纯电": "纯电", "增程": "增程", ...}
    #   4. 用 jieba 切 query，匹配各字典，返回过滤条件 dict
    ...


def apply_filters(candidates: list[tuple[float, dict]], filters: dict
                  ) -> list[tuple[float, dict]]:
    """
    对候选文档应用硬过滤。不通过的直接剔除（不是降权）。

    为什么用硬过滤而不是降权？
      价格超出预算 → 完全不应出现。"25 万以内"搜到 32 万的车就是错误。
    """
    # TODO:
    #   1. 从 doc["content"] 中提取价格、类别等字段（正则提取）
    #   2. 逐条比对 fiter 条件，不满足的剔除
    #   3. 返回过滤后的列表
    ...


# ============================================================
# 4. RAGAgent — 统一入口
# ============================================================

class RAGAgent:
    """
    完整 RAG Agent，把前四章 + Reranker + Query 改写打包成一个对象。

    使用方式：
      agent = RAGAgent(data_dir)
      answer = agent.chat("25万以内的大空间SUV")
      answer = agent.chat("那它的续航呢")  # 自动结合上一轮上下文
    """

    def __init__(self, data_dir: str):
        """
        离线建库阶段（和第四章 VectorIndex + BM25 初始化一样）：
          ① 加载文档
          ② 向量化
          ③ 建 FAISS 索引
          ④ 建 BM25 索引
          ⑤ 加载 Reranker
        """
        # ① 加载（第一章）
        self.documents = load_data(data_dir)

        # ② 向量化（第三章）
        self.documents = embed_documents(self.documents)

        # ③ FAISS 索引（第四章）
        self.vector_index = VectorIndex(self.documents)

        # ④ BM25 索引（第四章）
        self.bm25 = BM25(self.documents)

        # ⑤ Reranker（第五章新增）
        self.reranker = Reranker()

        # 对话状态
        self.chat_history: list[dict] = []  # [{"role": "user", "content": ...}, ...]

        print(f"[OK] RAG Agent 就绪：{len(self.documents)} 条文档，"
              f"索引 {self.vector_index.index.ntotal} 个向量，"
              f"词表 {len(self.bm25.df)} 个词")

    # ---- 检索管线 ----

    def retrieve(self, query: str, top_k: int = 3
                 ) -> tuple[list[tuple[float, dict]], str]:
        """
        完整检索管线：改写 → 混合检索 → 元数据过滤 → Reranker

        返回：
          [(最终分数, 文档), ...] 和 改写后的 query（用于日志/调试）
        """
        # ① Query 改写
        rewritten = rewrite_query(query, self.chat_history)

        # ② 混合检索（第四章）：候选池取 top_k*3，给后面过滤和精排留余量
        vec_candidates = self.vector_index.search(rewritten, top_k=top_k * 3)
        bm25_candidates = self.bm25.search(rewritten, top_k=top_k * 3)
        hybrid_results = hybrid_rrf(vec_candidates, bm25_candidates,
                                    k=60, top_k=top_k * 2)

        # ③ 元数据过滤
        filters = extract_filters(query)
        if filters:
            hybrid_results = apply_filters(hybrid_results, filters)

        # 如果过滤后一条都不剩，退回不过滤的结果
        if not hybrid_results:
            hybrid_results = hybrid_rrf(vec_candidates, bm25_candidates,
                                        k=60, top_k=top_k)

        # ④ Reranker 精排
        # TODO: 把 hybrid_results 传给 self.reranker.rerank()
        ...

        return reranked_results, rewritten

    # ---- 回答管线 ----

    def _build_prompt_with_history(self, query: str, context: str
                                   ) -> list[dict]:
        """
        和第一章 build_prompt 的区别：多了对话历史和引用格式要求。
        """
        # TODO:
        #   1. 把 chat_history 格式化成文本
        #   2. 在 system prompt 里加上：引用来源时用 [来源: xxx] 格式
        #   3. 构造 messages 返回
        ...

    def chat(self, query: str, top_k: int = 3) -> dict:
        """
        一次完整的 RAG 问答。

        返回 dict 而不是纯文本，方便前端展示：
          {
            "answer":       "LLM 回答",
            "sources":      [{"source": "...", "content": "...", "score": 0.85}, ...],
            "rewritten":    "改写后的 query",     # 方便调试
            "retrieval_time_ms": 123,           # 性能监控
          }
        """
        import time
        t0 = time.time()

        # ① 检索
        retrieved, rewritten = self.retrieve(query, top_k=top_k)

        # ② 组 prompt
        docs = [doc for _, doc in retrieved]
        context = "\n\n---\n\n".join(
            [f"[{i+1}] 来源：{doc['source']}\n{doc['content']}"
             for i, doc in enumerate(docs)]
        )
        messages = self._build_prompt_with_history(rewritten, context)

        # ③ 调 LLM
        answer = ask_llm(messages)

        retrieval_time = (time.time() - t0) * 1000

        # ④ 更新对话历史
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": [{"source": doc["source"],
                         "content": doc["content"][:200],
                         "score": round(score, 4)}
                        for score, doc in retrieved],
            "rewritten": rewritten,
            "retrieval_time_ms": round(retrieval_time, 1),
        }

    # ---- 对话管理 ----

    def reset(self):
        """重置对话历史（新话题开始）"""
        self.chat_history = []

    def get_history_summary(self) -> str:
        """返回对话摘要，调试用"""
        ...


# ============================================================
# 5. 评估内置 — 快速自检
# ============================================================

# TODO: 准备 10-15 条测试用例，每条包含 query + 期望出现的文档关键词
TEST_CASES = [
    # {"query": "小米SU7的续航是多少", "must_contain": ["小米", "SU7", "续航"]},
    # {"query": "25万以内的大空间SUV", "must_contain": ["理想L6", "海狮"]},
    # ...
]


def self_eval(agent: RAGAgent) -> dict:
    """
    快速自检：对每条测试 case 调 chat()，检查 answer 或 sources 里
    是否包含 must_contain 关键词。

    返回 {"hit_rate": float, "mrr": float, "details": [...]}
    """
    # TODO:
    #   1. 遍历 TEST_CASES
    #   2. agent.chat(case["query"])
    #   3. 检查 result["sources"] 里是否命中 must_contain
    #   4. 统计 Hit Rate 和 MRR
    ...


# ============================================================
# 6. 交互 / 启动
# ============================================================

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")

    agent = RAGAgent(DATA_DIR)

    print("\n命令：输入问题开始对话 | /reset 重置对话 | /hist 查看历史 | /eval 自检 | q 退出\n")

    while True:
        try:
            query = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() == "q":
            break
        if not query:
            continue
        if query == "/reset":
            agent.reset()
            print("[OK] 对话已重置")
            continue
        if query == "/hist":
            for msg in agent.chat_history:
                print(f"[{msg['role']}] {msg['content'][:100]}...")
            continue
        if query == "/eval":
            result = self_eval(agent)
            print(f"[Eval] Hit Rate: {result['hit_rate']:.1%}, MRR: {result['mrr']:.3f}")
            continue

        # 核心调用
        result = agent.chat(query)

        print(f"\n{'='*60}")
        print(f"[改写后查询] {result['rewritten']}")
        print(f"[检索耗时] {result['retrieval_time_ms']}ms")
        print(f"{'='*60}")

        print(f"\n{result['answer']}")

        print(f"\n{'─'*60}")
        print("参考来源：")
        for i, src in enumerate(result["sources"]):
            print(f"  [{i+1}] {src['source']} (score: {src['score']})")
            print(f"      {src['content'][:80]}...")
        print()
