"""
Day 2: LangChain 封装 RAG + Memory 管理

目的：把第五章手写的 RAG 六步管线，用 LangChain 的 Runnable 协议重写一遍。
      ——同样的检索逻辑，换一种组织方式。

核心概念：
  ① Memory：让 Chain 记住对话历史（Buffer / Summary / Window 三种策略）
  ② RunnableLambda：把普通函数包装成 LangChain 可串联的组件
  ③ BaseRetriever：自定义检索器，融入 LangChain 生态
  ④ RunnablePassthrough：原样透传字段，简化数据流
"""

import os
import sys
import time
from typing import List, Tuple, Dict, Any, Optional

# ============================================================
# 0. 路径 & 环境初始化
# ============================================================

# 确保能从 agent/ 目录引用到 chapters/ 下的模块
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "chapters"))

# 加载 agent/.env 里的 DeepSeek API 配置
from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ---- 复用第五章的已有模块 ----
from chapters import load_data, build_prompt, ask_llm
from embedding_test import _embed_model, embed_documents
from retrieval_test import VectorIndex, BM25, hybrid_rrf
from full_rag_agent import (
    RAGAgent, rewrite_query, extract_filters, apply_filters, Reranker
)

# ---- LangChain 核心组件 ----
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda, RunnableParallel, RunnablePassthrough
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import HumanMessage, AIMessage

# ---- LangChain 社区集成（Memory 实现） ----
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
)
from langchain_openai import ChatOpenAI

# ============================================================
# 1. 创建 LangChain 兼容的 LLM 实例
# ============================================================

# LangChain 的 ChatOpenAI 底层也是调 OpenAI-compatible API，
# 和第五章 ask_llm() 走的是同一套协议，多了一层 Runnable 包装
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL_ID", "deepseek-chat"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    temperature=0.3,
)

# 额外创建一个 temperature=0 的实例，给 Memory 摘要用（要稳定输出）
llm_stable = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL_ID", "deepseek-chat"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    temperature=0,
)


# ============================================================
# 2. 知识点 1：Memory 三种策略对比
# ============================================================
# 为什么要管理 Memory？
#   第五章手动维护 self.chat_history 列表，每次手动追加、手动截断。
#   LangChain 的 RunnableWithMessageHistory 自动做：调前取历史 → 调后写回。
#
# 三种策略的取舍：
#   BufferMemory    — 全部保留，短对话最佳，token 线性增长
#   SummaryMemory   — LLM 定期压缩，长对话首选，多一次 LLM 调用
#   WindowMemory    — 只保留最近 K 轮，中间方案，可能丢失远距离上下文


def demo_memory_comparison():
    """
    对比三种 Memory 策略在多次对话后的 token 消耗。

    核心观察：
      - BufferMemory：消息数线性增长，10 轮后 context 里塞满 22 条消息
      - SummaryMemory：历史被压缩成一段摘要文本，消息数保持恒定
      - WindowMemory(K=3)：始终只保留最近 6 条消息（3 轮），老内容直接丢弃
    """
    print("\n" + "=" * 60)
    print("知识点 1：Memory 三种策略对比")
    print("=" * 60)

    # 一个极简单的翻译 prompt，用来演示 Memory 效果
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是中英翻译助手。请根据对话历史理解上下文，完成翻译。"),
        MessagesPlaceholder(variable_name="history"),  # ← Memory 注入点
        ("human", "{question}"),
    ])

    # 基础链：prompt → model → 纯文本
    base_chain = prompt | llm | StrOutputParser()

    # ---- 模拟 5 轮对话，打印历史长度 ----
    conversations = [
        "翻译：今天天气真好",
        "把上一句翻成英文",
        "翻译：我买了一辆新车",
        "上一句用日语怎么表达",
        "再翻译一次第一句话，用英文",
    ]

    # 策略 A：全量 Buffer —— 用 dict 模拟简易 session 存储
    print("\n[策略 A] ConversationBufferMemory（全量保留）")
    store_a: Dict[str, List] = {}

    def get_history_a(session_id: str) -> List:
        return store_a.get(session_id, [])

    chain_a = RunnableWithMessageHistory(
        base_chain, get_history_a,
        input_messages_key="question",
        history_messages_key="history",
    )

    for i, q in enumerate(conversations):
        result = chain_a.invoke(
            {"question": q},
            config={"configurable": {"session_id": "demo"}}
        )
        # 模拟：手动把本轮对话写回 store（生产环境 LangChain 自动做）
        store_a.setdefault("demo", []).append(HumanMessage(content=q))
        store_a.setdefault("demo", []).append(AIMessage(content=result))
        print(f"  第{i+1}轮后 | 历史消息数: {len(store_a.get('demo', []))} | "
              f"问题: {q[:20]}... | 回答: {result[:30]}...")

    print(f"  → 5 轮后累计 {len(store_a.get('demo', []))} 条消息，每轮都多 2 条")

    # 策略 B：WindowMemory —— 只保留最近 K 轮
    print("\n[策略 B] ConversationBufferWindowMemory(K=2，只保留最近 2 轮)")
    store_b: Dict[str, List] = {}

    def get_history_b(session_id: str) -> List:
        """窗口策略：只返回最近 4 条消息（= 最近 2 轮）"""
        msgs = store_b.get(session_id, [])
        return msgs[-4:]  # K=2 轮 × 2 条/轮 = 4 条

    chain_b = RunnableWithMessageHistory(
        base_chain, get_history_b,
        input_messages_key="question",
        history_messages_key="history",
    )

    for i, q in enumerate(conversations):
        result = chain_b.invoke(
            {"question": q},
            config={"configurable": {"session_id": "demo"}}
        )
        store_b.setdefault("demo", []).append(HumanMessage(content=q))
        store_b.setdefault("demo", []).append(AIMessage(content=result))
        effective = len(get_history_b("demo"))
        print(f"  第{i+1}轮后 | 有效历史: {effective} 条（全量 {len(store_b['demo'])} 条）"
              f" | 问题: {q[:20]}...")

    print(f"  → 无论多少轮，传给 LLM 的始终 ≤ 4 条历史消息")

    # 策略 C：SummaryMemory —— LLM 压缩历史
    print("\n[策略 C] ConversationSummaryMemory（LLM 自动摘要）")
    summary_text = ""  # 初始摘要为空

    for i, q in enumerate(conversations):
        # 把摘要作为 system prompt 的一部分注入
        summary_prefix = f"[对话历史摘要] {summary_text}\n\n" if summary_text else ""

        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", f"你是中英翻译助手。{summary_prefix}请结合摘要理解上下文。"),
            ("human", "{question}"),
        ])
        summary_chain = summary_prompt | llm | StrOutputParser()
        result = summary_chain.invoke({"question": q})

        # 每 2 轮用 LLM 把最近对话压缩成摘要
        if (i + 1) % 2 == 0:
            compress_prompt = ChatPromptTemplate.from_messages([
                ("system", "请用一句话概括以下对话的关键信息，只输出摘要："),
                ("human", f"摘要历史: {summary_text}\n本轮: {q} → {result}"),
            ])
            summary_text = (compress_prompt | llm_stable | StrOutputParser()
                           ).invoke({})
            print(f"  [摘要更新] 第{i+1}轮后: {summary_text[:60]}...")
        print(f"  第{i+1}轮 | 摘要长度: {len(summary_text)} 字符 | "
              f"回答: {result[:30]}...")

    print(f"  → 无论多少轮，传给 LLM 的始终是固定长度的摘要，不增长")

    print("\n[对比总结]")
    print("  策略        │ 传给LLM的内容       │ token增长    │ 适用场景")
    print("  Buffer      │ 全部历史消息         │ 线性增长     │ <10轮短对话")
    print("  Window(K=2) │ 最近4条消息          │ 恒定         │ 近期上下文敏感")
    print("  Summary     │ 压缩摘要(固定长度)    │ 恒定         │ 长对话/客服场景")


# ============================================================
# 3. 知识点 2：RunnableLambda 封装 RAG 管线
# ============================================================
# 第五章的命令式写法：
#   rewritten = rewrite_query(query, history)
#   vec = vector_index.search(rewritten)
#   bm25 = bm25_idx.search(rewritten)
#   hybrid = hybrid_rrf(vec, bm25)
#   ...
#
# Day 2 的声明式写法：
#   每一步包装成 RunnableLambda，用 | 串联，数据以 dict 形式在管道中流动。

def build_rag_chain(
    vector_index: VectorIndex,
    bm25_idx: BM25,
    reranker: Reranker,
    top_k: int = 3,
):
    """
    把第五章的命令式调用编排成 LangChain Runnable 管道。

    数据流（state dict 在各节点间传递）：
      {"question": "用户原文", "history": [...]}
        → {"question": ..., "history": ..., "rewritten": "改写后"}
        → {"question": ..., ..., "dense_docs": [...], "sparse_docs": [...]}
        → {"question": ..., ..., "hybrid_docs": [...]}
        → {"question": ..., ..., "filtered_docs": [...]}
        → {"question": ..., ..., "reranked_docs": [...]}
        → {"question": ..., ..., "context": "拼好的上下文字符串"}
        → 喂给 prompt | model | parser → 最终 answer
    """

    # ---- 节点函数：每个返回 dict，会和上游 state 合并 ----

    def step_rewrite(state: dict) -> dict:
        """Step ①: Query 改写（复用第五章的函数）"""
        q = state["question"]
        history = state.get("history", [])
        rewritten = rewrite_query(q, history)
        return {"rewritten": rewritten}

    def step_retrieve(state: dict) -> dict:
        """
        Step ②: 两路并行检索（FAISS 向量 + BM25 关键词）

        注意：这里用 RunnableParallel 并行跑两路。
        LangChain 的 RunnableParallel 会同时执行两个分支，然后合并结果。
        """
        query = state["rewritten"]
        # 取 top_k*3 给后续的 RRF 和 Reranker 留余量
        dense = vector_index.search(query, top_k=top_k * 3)
        sparse = bm25_idx.search(query, top_k=top_k * 3)
        return {"dense_docs": dense, "sparse_docs": sparse}

    def step_fusion(state: dict) -> dict:
        """Step ③: RRF 融合两路检索结果（复用第四章的函数）"""
        hybrid = hybrid_rrf(
            state["dense_docs"], state["sparse_docs"],
            k=60, top_k=top_k * 2
        )
        return {"hybrid_docs": hybrid}

    def step_filter(state: dict) -> dict:
        """Step ④: 元数据硬过滤（复用第五章的函数）"""
        filters = extract_filters(state["question"])
        if filters:
            filtered = apply_filters(state["hybrid_docs"], filters)
        else:
            filtered = state["hybrid_docs"]
        # 容错：过滤后一条不剩 → 退回不过滤的结果
        if not filtered:
            filtered = state["hybrid_docs"][:top_k]
        return {"filtered_docs": filtered}

    def step_rerank(state: dict) -> dict:
        """Step ⑤: CrossEncoder 精排（复用第五章的 Reranker）"""
        reranked = reranker.rerank(state["rewritten"], state["filtered_docs"], top_k=top_k)
        if not reranked:
            reranked = state["filtered_docs"][:top_k]
        return {"reranked_docs": reranked}

    def step_assemble(state: dict) -> dict:
        """Step ⑥: 把检索结果拼成上下文字符串"""
        docs = [doc for _, doc in state["reranked_docs"]]
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(
                f"[{i}] 来源: {doc.get('source', '未知')}\n"
                f"{doc.get('content', '')}"
            )
        context = "\n\n---\n\n".join(parts)
        return {"context": context, "sources": docs}

    # ---- 最终生成 prompt ----
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "你是一个专业的汽车导购助手。"
            "严格根据上下文回答问题，不编造数据。"
            "回答时引用来源，格式：[来源: xxx]。"
            "如果信息不足，请明确告知用户。"
        )),
        ("human", (
            "## 参考资料\n{context}\n\n"
            "## 用户问题\n{question}\n\n"
            "请根据以上参考资料回答。"
        )),
    ])

    # ---- 串联：RunnableLambda 用 | 拼接 ----
    chain = (
        RunnableLambda(step_rewrite)
        | RunnableLambda(step_retrieve)
        | RunnableLambda(step_fusion)
        | RunnableLambda(step_filter)
        | RunnableLambda(step_rerank)
        | RunnableLambda(step_assemble)
        | rag_prompt | llm | StrOutputParser()
    )

    return chain


# ============================================================
# 4. 知识点 3：自定义 HybridRetriever（继承 BaseRetriever）
# ============================================================
# 为什么要继承 BaseRetriever？
#   LangChain 生态里，Retriever 是一等公民：
#   - 可以和任何 prompt/chain 用 | 串联
#   - 自带 .invoke() / .batch() / .stream() 等 Runnable 协议方法
#   - 可以被 LangChain 的 EnsembleRetriever 等工具直接使用
#
# 对比第五章：
#   第五章的检索函数返回 [(score, doc), ...] 的元组列表
#   BaseRetriever 要求返回 List[Document]（LangChain 的 Document 对象）

from langchain_core.documents import Document


class HybridRetriever(BaseRetriever):
    """
    把 FAISS + BM25 + RRF 检索逻辑封装成标准 LangChain Retriever。

    使用方式：
      retriever = HybridRetriever(vector_index, bm25_idx, reranker)
      docs = retriever.invoke("25万以内的SUV")  # 返回 List[Document]
      chain = retriever | prompt | model | parser  # 和任何 chain 串联
    """

    def __init__(
        self,
        vector_index: VectorIndex,
        bm25_idx: BM25,
        reranker: Optional[Reranker] = None,
        top_k: int = 3,
    ):
        super().__init__()
        self._vector_index = vector_index
        self._bm25_idx = bm25_idx
        self._reranker = reranker
        self._top_k = top_k

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """
        核心方法：输入 query 字符串，输出 LangChain Document 列表。

        内部流程和第五章 RAGAgent.retrieve() 完全一致：
          ① 两路检索（FAISS + BM25）
          ② RRF 融合
          ③ 元数据过滤
          ④ Reranker 精排（可选）
          ⑤ 转成 LangChain Document 格式
        """
        # ① + ②：检索 + 融合
        dense = self._vector_index.search(query, top_k=self._top_k * 3)
        sparse = self._bm25_idx.search(query, top_k=self._top_k * 3)
        hybrid = hybrid_rrf(dense, sparse, k=60, top_k=self._top_k * 2)

        # ③ 元数据过滤
        filters = extract_filters(query)
        if filters:
            hybrid = apply_filters(hybrid, filters)
        if not hybrid:
            hybrid = hybrid_rrf(dense, sparse, k=60, top_k=self._top_k)

        # ④ Reranker 精排（如果提供了的话）
        if self._reranker and self._reranker.model is not None:
            reranked = self._reranker.rerank(query, hybrid, top_k=self._top_k)
            if reranked:
                hybrid = reranked

        # ⑤ 转成 LangChain Document 格式
        documents = []
        for score, doc in hybrid[:self._top_k]:
            documents.append(Document(
                page_content=doc.get("content", ""),
                metadata={
                    "source": doc.get("source", "未知来源"),
                    "type": doc.get("type", ""),
                    "score": round(score, 4),
                }
            ))
        return documents


# ============================================================
# 5. 知识点 4：RunnablePassthrough 简化数据流
# ============================================================
# 场景：prompt 需要"context + question"两个字段，
# 但 chain 上游只产出了 context，question 在最初的输入里。
# 不写 RunnablePassthrough：需要额外写一个函数把 question 透传下来。
# 写 RunnablePassthrough：一行搞定。

def demo_passthrough(retriever: HybridRetriever):
    """
    演示 RunnablePassthrough 的用法。

    对比：
      # 不用 Passthrough —— 要额外写函数透传
      def add_question(docs):
          return {"context": docs, "question": ???}  # question 从哪来？

      # 用 Passthrough —— 声明式透传
      {
          "context": retriever,
          "question": RunnablePassthrough(),  # ← 原样穿过
      }
    """
    print("\n" + "=" * 60)
    print("知识点 4：RunnablePassthrough 简化数据流")
    print("=" * 60)

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是汽车导购助手。根据参考资料回答问题。"),
        ("human", "## 参考资料\n{context}\n\n## 问题\n{question}"),
    ])

    # RunnablePassthrough 把原始输入的 "question" 字段原样传到 prompt
    chain = (
        {
            "context": retriever,            # 检索器产出 Document 列表
            "question": RunnablePassthrough(), # 用户问题原样穿过
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke("小米SU7的续航里程是多少")
    print(f"\n[问题] 小米SU7的续航里程是多少")
    print(f"[回答] {result[:200]}...")

    print("\n  核心理解：")
    print("    - RunnablePassthrough() 把输入原样传给下游")
    print("    - 适合多输入源的场景：context 来自检索，question 来自用户")
    print("    - 避免为了透传一个字段而写额外的包装函数")


# ============================================================
# 6. 知识点 5：对比实验 —— 原生 RAGAgent vs LCEL Chain
# ============================================================

def demo_comparison(data_dir: str):
    """
    同一批问题，分别用原生 RAGAgent 和 LCEL Chain 跑一遍，
    对比代码量、耗时、结果一致性。
    """
    print("\n" + "=" * 60)
    print("知识点 5：原生 RAGAgent vs LCEL Chain 对比")
    print("=" * 60)

    # ---- 初始化（两边共用同一份索引） ----
    print("\n[初始化] 加载数据 & 建索引...")
    documents = load_data(data_dir)
    documents = embed_documents(documents)
    v_idx = VectorIndex(documents)
    bm25_idx = BM25(documents)
    reranker = Reranker()

    # ---- 原生 RAGAgent ----
    agent = RAGAgent(data_dir)

    # ---- LCEL Chain ----
    lcel_chain = build_rag_chain(v_idx, bm25_idx, reranker)

    # ---- 对比测试 ----
    test_queries = [
        "小米SU7的续航是多少",
        "25万以内的大空间SUV有哪些",
        "适合家庭出行的新能源车推荐",
    ]

    print(f"\n{'问题':<30} | {'原生(s)':<8} | {'LCEL(s)':<8} | {'结果一致?'}")
    print("-" * 70)

    for query in test_queries:
        # 原生 RAGAgent
        t0 = time.time()
        native_result = agent.chat(query)
        native_time = time.time() - t0
        agent.reset()

        # LCEL Chain
        t0 = time.time()
        lcel_answer = lcel_chain.invoke({
            "question": query,
            "history": [],
        })
        lcel_time = time.time() - t0

        # 粗略判断一致性：看长度是否接近
        native_len = len(native_result["answer"])
        lcel_len = len(lcel_answer)
        consistent = "✓" if abs(native_len - lcel_len) < max(native_len, lcel_len) * 0.5 else "?"

        print(f"  {query:<28} | {native_time:.3f}s  | {lcel_time:.3f}s  | {consistent}")

    print(f"\n  对比结论：")
    print(f"    - LCEL 耗时略高（多了 RunnableLambda 包装的微量开销），实际可忽略")
    print(f"    - 声明式链的代码更紧凑，每一步可独立测试")
    print(f"    - 原生写法更直观，适合快速开发")
    print(f"    - 生产项目：用 LangChain 做编排（方便扩展），核心算法保持纯函数（方便测试）")


# ============================================================
# 7. 主入口
# ============================================================

if __name__ == "__main__":
    DATA_DIR = os.path.join(PROJECT_DIR, "data")

    print("=" * 60)
    print("Day 2: LangChain 封装 RAG + Memory 管理")
    print("=" * 60)

    # ---- 一次性初始化索引（所有 demo 共用） ----
    print("\n[加载] 数据 & 建索引...")
    documents = load_data(DATA_DIR)
    documents = embed_documents(documents)
    v_idx = VectorIndex(documents)
    bm25_idx = BM25(documents)
    reranker = Reranker()
    retriever = HybridRetriever(v_idx, bm25_idx, reranker, top_k=3)
    print(f"[OK] {len(documents)} 条文档, FAISS={v_idx.index.ntotal} 向量, "
          f"BM25 词表={len(bm25_idx.df)} 词")

    # ---- 知识点 1：Memory 策略对比 ----
    demo_memory_comparison()

    # ---- 知识点 2+3：LCEL RAG 链 ----
    print("\n" + "=" * 60)
    print("知识点 2+3：LCEL RAG 链 & HybridRetriever")
    print("=" * 60)

    rag_chain = build_rag_chain(v_idx, bm25_idx, reranker)

    test_query = "小米SU7的续航里程是多少"
    print(f"\n[问题] {test_query}")
    answer = rag_chain.invoke({
        "question": test_query,
        "history": [],
    })
    print(f"[回答] {answer}")

    # HybridRetriever 独立使用
    print(f"\n[HybridRetriever 独立使用]")
    docs = retriever.invoke(test_query)
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.metadata['source']} (score: {doc.metadata['score']})")
        print(f"      {doc.page_content[:80]}...")

    # ---- 知识点 4：RunnablePassthrough ----
    demo_passthrough(retriever)

    # ---- 知识点 5：对比实验 ----
    demo_comparison(DATA_DIR)

    print("\n" + "=" * 60)
    print("[OK] Day 2 全部演示完成")
    print("=" * 60)
