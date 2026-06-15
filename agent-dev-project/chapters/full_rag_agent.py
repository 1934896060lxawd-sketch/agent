"""
第五章：完整 RAG Agent —— 集成前四章全部优点

管线全貌（一次查询走完这 6 步）：
  ① Query 改写      — LLM 把省略/模糊的问题扩展成完整检索查询
  ② 混合检索        — FAISS 向量 + BM25 关键词 → RRF 融合（第四章）
  ③ 元数据过滤      — 价格区间/车型/类别等结构化字段硬过滤
  ④ Reranker 精排   — CrossEncoder 对候选文档和 query 算真正相关性
  ⑤ 上下文拼接      — Top-K 文档 + 对话历史 → 结构化 prompt
  ⑥ LLM 生成        — 带引用标注的回答
"""

import os
# ⚠️ 必须在 import sentence_transformers 之前设置，否则 huggingface_hub
#    在 CrossEncoder 构造时已经解析了 huggingface.co 域名，镜像就失效了
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import re
import sys
import math
import jieba
import numpy as np
import faiss
from collections import defaultdict
from sentence_transformers import CrossEncoder
from typing import List, Tuple, Dict, Optional

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

    # 加载本地模型路径
    _LOCAL_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "models",
        "bge-reranker-base", "BAAI", "bge-reranker-base"
    )

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        BGE Reranker 和 BGE Embedding 同一个团队出品，中文效果好。
        加载策略：本地优先 → 镜像下载 → 降级跳过
        """
        if os.path.isdir(self._LOCAL_DIR):
            self.model = CrossEncoder(self._LOCAL_DIR)
        else:
            try:
                self.model = CrossEncoder(model_name)
            except Exception as e:
                print(f"[WARN] Reranker 模型加载失败: {e}")
                print(f"[WARN] 精排步骤将跳过，请手动下载 bge-reranker-base 到:")
                print(f"       {self._LOCAL_DIR}")
                self.model = None

    def rerank(self, query: str, candidates: list[tuple[float, dict]], top_k: int = 3
               ) -> list[tuple[float, dict]]:
        """
        对候选文档逐个和 query 算交叉编码分数。
        输入：混合检索返回的候选文档列表 [(score, doc), ...]
        输出：按相关性精排后的 top_k，格式同输入
        """
        # 模型未加载 → 降级：不做精排，直接返回原始候选
        if self.model is None:
            return candidates[:top_k]

        # ① 从 (score, doc) 中提取纯文档列表，丢掉旧分数（RRF 分数在精排阶段无意义）
        docs = [doc for _, doc in candidates]

        # ② 构造 [[query, doc["content"]], ...] pairs
        pairs = [[query, doc["content"]] for doc in docs]

        # ③ self.model.predict(pairs) 得到精排分数
        scored = self.model.predict(pairs)

        # ④ 排序取 top_k，新分数 + 原文档
        ranked = sorted(
            zip(scored, docs),
            key=lambda x: x[0],
            reverse=True
        )[:top_k]

        return [(float(s), doc) for s, doc in ranked]

    def format_results(self, scored: list[tuple[float, dict]]) -> str:
        """
        格式化精排结果，和其他检索方法保持一致。
        分数含义和向量检索不同（不是余弦相似度），所以标注为"精排分"。
        """
        lines = []
        for i, (score, doc) in enumerate(scored, 1):
            content = doc.get("content", "")
            preview = content[:200] + "..." if len(content) > 200 else content
            source = doc.get("source", "未知来源")

            line = (
                f"{i}. 【精排分: {score:.4f}】\n"
                f"   来源: {source}\n"
                f"   内容: {preview}\n"
            )
            lines.append(line)

        header = f"精排结果 (共 {len(scored)} 条):\n" + "=" * 50 + "\n"
        return header + "\n".join(lines)


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
    if not chat_history:
        return query
    #   2. 否则构造 messages = [system prompt] + 历史 + 当前问题
    messages = [{"role": "system", "content": QUERY_REWRITE_SYSTEM}]
    # 追加对话历史（只取最近 N 轮防止 context 过长）
    # 提示：如果历史很长，取 chat_history[-6:] 保留最近 3 轮
    messages.extend(chat_history[-6:])  # 最近 3 轮 = 6 条消息

    # 追加当前问题
    messages.append({
        "role": "user",
        "content": f"请把以下问题改写成独立的检索查询（直接输出改写结果，不要解释）：\n{query}"
    })

    # ③ 调 LLM 获取改写结果
    rewritten = ask_llm(messages)

    # ④ 容错：如果 LLM 返回了空或异常长的结果，退回原始 query
    if not rewritten or len(rewritten) > 100:
        return query

    return rewritten.strip()


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

    实现思路：用规则匹配（jieba 分词 + 关键词字典），不调 LLM（太慢）。
    进阶：用 LLM 做 few-shot 提取，准确率更高但多一次 API 调用。
    """
    # 价格匹配模式
    PRICE_PATTERNS = [
        (r"(?P<price>\d+)万\s*(?:以[内下]|之内|以下)", "max_price"),
        (r"(?P<price>\d+)万\s*(?:以[上外]|之上|以上)", "min_price"),
        (r"(?P<price>\d+)\s*[-到~]\s*(?P<price2>\d+)\s*万", "price_range"),
    ]
    
    # 类别关键词字典
    CATEGORY_DICT = {
        "SUV": "SUV",
        "suv": "SUV",
        "轿车": "轿车",
        "MPV": "MPV",
        "mpv": "MPV",
        "皮卡": "皮卡",
        "跑车": "跑车",
    }
    
    # 动力类型字典
    ENERGY_DICT = {
        "纯电": "纯电",
        "电动": "纯电",
        "电车": "纯电",
        "增程": "增程",
        "插混": "插混",
        "混动": "插混",
        "燃油": "燃油",
        "汽油": "燃油",
        "柴油": "燃油",
    }
    
    # 品牌字典
    BRAND_DICT = {
        "比亚迪": "比亚迪", "小米": "小米", "蔚来": "蔚来",
        "理想": "理想", "小鹏": "小鹏", "特斯拉": "特斯拉",
        "极氪": "极氪", "问界": "问界", "零跑": "零跑",
        "哪吒": "哪吒", "埃安": "埃安", "深蓝": "深蓝",
    }
    
    filters = {}

    # ① 价格提取（正则）
    for pattern, key in PRICE_PATTERNS:
        m = re.search(pattern, query)
        if m:
            if key == "price_range":
                # "15-20万" → min=15, max=20
                filters["min_price"] = int(m.group("price"))
                filters["max_price"] = int(m.group("price2"))
            else:
                filters[key] = int(m.group("price"))
            break  # 匹配到一条价格就停，避免歧义

    # ② 类别/动力/品牌提取（jieba + 字典匹配）
    words = set(jieba.cut(query))

    for w in words:
        if "category" not in filters and w in CATEGORY_DICT:
            filters["category"] = CATEGORY_DICT[w]
        if "powertrain" not in filters and w in ENERGY_DICT:
            filters["powertrain"] = ENERGY_DICT[w]
        if "brand" not in filters and w in BRAND_DICT:
            filters["brand"] = BRAND_DICT[w]

    return filters


def apply_filters(candidates: List[Tuple[float, dict]], 
                  filters: Dict[str, any]) -> List[Tuple[float, dict]]:
    """
    对候选文档应用硬过滤。不通过的直接剔除（不是降权）。

    为什么用硬过滤而不是降权？
      价格超出预算 → 完全不应出现。"25 万以内"搜到 32 万的车就是错误。
    
    Args:
        candidates: 候选文档列表，格式 [(score, doc), ...]
        filters: 过滤条件字典，如 {"max_price": 25, "category": "SUV"}
        
    Returns:
        过滤后的文档列表
    """
    if not filters:
        return candidates

    # 预编译价格正则，提高性能
    PRICE_REGEX = re.compile(r'售价\s*([\d.]+)\s*(?:[-到~]\s*([\d.]+))?\s*万')

    def _passes(doc: dict) -> bool:
        """检查一篇文档是否满足所有过滤条件"""
        doc_text = doc.get("content", "")
        doc_type = doc.get("type", "")  # load_data 的文档格式：type 在顶层，不在 metadata 下
        
        # 价格检查 - 只对车型规格文档进行严格检查
        if ("max_price" in filters or "min_price" in filters) and doc_type == "car_spec":
            price_match = PRICE_REGEX.search(doc_text)
            
            if price_match:
                # 取最低价作为判断依据（保守策略：起售价超预算才踢）
                low_price = float(price_match.group(1))
                
                if "max_price" in filters and low_price > filters["max_price"]:
                    return False  # 起售价就超预算了 → 踢掉
                    
                if "min_price" in filters:
                    # 如果有区间价，取高值比 min
                    high_price = float(price_match.group(2)) if price_match.group(2) else low_price
                    if high_price < filters["min_price"]:
                        return False  # 最高价也达不到预算下限 → 踢掉
            # 如果没匹配到售价（行业报告/用户评价），不因为价格过滤而踢掉

        # 类别检查
        if "category" in filters and filters["category"] not in doc_text:
            return False
            
        # 动力类型检查
        if "powertrain" in filters and filters["powertrain"] not in doc_text:
            return False
            
        # 品牌检查
        if "brand" in filters and filters["brand"] not in doc_text:
            return False

        return True

    # 逐条筛选，保持原有顺序
    return [(score, doc) for score, doc in candidates if _passes(doc)]


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
        reranked_results = self.reranker.rerank(rewritten, hybrid_results, top_k=top_k)

        # 如果精排返回为空（极端情况），退回混合检索结果
        if not reranked_results:
            reranked_results = hybrid_results[:top_k]

        return reranked_results, rewritten

    # ---- 回答管线 ----

    def _build_prompt_with_history(self, query: str, context: str) -> list[dict]:
        """
        和第一章 build_prompt 的区别：多了对话历史和引用格式要求。

        核心差异：
          第一章：system + user(context + query)，无历史
          第五章：system + 历史摘要 + user(context + query)，支持多轮
        """
        # ① 把对话历史格式化成文本（前 N 条完整，最近 3 轮摘要）
        history_text = ""
        if self.chat_history:
            # 取最近 6 条消息（3 轮对话），更早的做摘要
            recent = self.chat_history[-6:]
            history_lines = []
            for msg in recent:
                role_name = "用户" if msg["role"] == "user" else "助手"
                # 截断过长的内容，保留前 200 字
                content = msg["content"][:200]
                history_lines.append(f"[{role_name}] {content}")
            history_text = "## 对话历史\n" + "\n".join(history_lines) + "\n\n"

        # ② system prompt：比第一章多了引用格式要求
        system_prompt = (
            "你是一个专业的汽车导购助手，具备以下能力：\n"
            "1. 严格根据提供的参考资料回答问题，不编造数据\n"
            "2. 回答时引用具体来源，格式：[来源: 文档名]\n"
            "3. 如果资料不足以回答，明确告知用户\n"
            "4. 结合对话历史理解用户的连续问题"
        )

        # ③ user prompt：历史 + 参考资料 + 问题
        user_prompt = (
            f"{history_text}"
            f"## 参考资料\n{context}\n\n"
            f"## 用户问题\n{query}\n\n"
            f"请根据以上参考资料回答用户的问题，引用来源时使用 [来源: xxx] 格式。"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

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
        """返回对话摘要，调试用：看改写效果、检索耗时、上下文长度"""
        if not self.chat_history:
            return "(无对话历史)"

        lines = []
        total_user_chars = 0
        total_assistant_chars = 0
        for msg in self.chat_history:
            if msg["role"] == "user":
                total_user_chars += len(msg["content"])
            else:
                total_assistant_chars += len(msg["content"])

        rounds = len(self.chat_history) // 2
        lines.append(f"对话轮次: {rounds}")
        lines.append(f"消息总数: {len(self.chat_history)}")
        lines.append(f"用户总字数: {total_user_chars}")
        lines.append(f"助手总字数: {total_assistant_chars}")
        lines.append(f"最近问题: {self.chat_history[-2]['content'][:80]}..."
                     if len(self.chat_history) >= 2 else "N/A")
        return "\n".join(lines)


# ============================================================
# 5. 评估内置 — 快速自检
# ============================================================

TEST_CASES = [
    # ---- 精确匹配型：结果必须包含特定品牌/车型 ----
    {"query": "小米SU7的续航是多少", "must_contain": ["小米", "SU7"]},
    {"query": "比亚迪海豚的价格",      "must_contain": ["比亚迪", "海豚"]},
    {"query": "特斯拉Model Y的智驾配置", "must_contain": ["特斯拉", "Model Y"]},

    # ---- 条件过滤型：必须满足价格/类别约束 ----
    {"query": "25万以内的大空间SUV",  "must_contain": ["理想L6"]},
    {"query": "10万左右的纯电轿车",   "must_contain": ["海豚"]},

    # ---- 语义泛化型：不能精确匹配，但语义相关 ----
    {"query": "适合家庭出行的新能源车", "must_contain": ["SUV", "理想"]},
    {"query": "今年最火的国产电动车",   "must_contain": ["小米", "比亚迪"]},

    # ---- 多轮对话模拟（手动测时用 rewrite_query 逻辑，这里简化） ----
    {"query": "充电最快的纯电车有哪些", "must_contain": ["纯电", "充电"]},
    {"query": "智驾能力最强的车型",     "must_contain": ["L2", "自动驾驶"]},

    # ---- 边界 case ----
    {"query": "100万以上的豪华电动车", "must_contain": []},  # 数据里可能没有，验证不编造
]


def self_eval(agent: RAGAgent) -> dict:
    """
    快速自检：对每条测试 case 调 chat()，检查 sources 里
    是否包含 must_contain 关键词。

    指标：
      Hit@K:  Top-K 条来源里至少命中一个 must_contain 关键词 = 成功
              (Hit Rate = 成功数 / 总 case 数)
      MRR:    第一个命中关键词的来源排名取倒数，再平均
              (Mean Reciprocal Rank，越接近 1 越好)
              RR = 1 / 排名（第一个命中的排在第 2 位 → RR = 0.5）
    """
    details = []
    hits = 0
    rr_sum = 0.0

    for case in TEST_CASES:
        query = case["query"]
        must_contain = case["must_contain"]

        # 调用 agent
        result = agent.chat(query)

        # 检查 sources 里是否命中 must_contain
        sources = result["sources"]
        first_hit_rank = None
        hit_keywords = set()

        for rank, src in enumerate(sources, 1):
            combined = src["source"] + " " + src.get("content", "")
            for kw in must_contain:
                if kw in combined and kw not in hit_keywords:
                    hit_keywords.add(kw)
                    if first_hit_rank is None:
                        first_hit_rank = rank

        is_hit = len(hit_keywords) > 0 or len(must_contain) == 0
        if is_hit and first_hit_rank:
            hits += 1
            rr_sum += 1.0 / first_hit_rank
        elif len(must_contain) == 0:
            # 边界 case：must_contain 为空，只看是否返回了结果
            hits += 1
            rr_sum += 1.0

        details.append({
            "query": query,
            "hit": is_hit,
            "hit_keywords": list(hit_keywords),
            "expected": must_contain,
            "first_hit_rank": first_hit_rank,
        })

        # 每次调用后重置对话历史，case 之间互不干扰
        agent.reset()

    n = len(TEST_CASES)
    hit_rate = hits / n if n > 0 else 0.0
    mrr = rr_sum / n if n > 0 else 0.0

    return {"hit_rate": hit_rate, "mrr": mrr, "details": details}


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
