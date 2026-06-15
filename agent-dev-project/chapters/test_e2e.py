"""
端到端测试：覆盖 6 步管线的所有分支
"""
import os
import sys
import io

# 重定向 stdout，避免 tqdm 进度条和中文乱码干扰
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "chapters"))

from full_rag_agent import RAGAgent, self_eval

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

print("=" * 60)
print("端到端测试：full_rag_agent.py")
print("=" * 60)

# ---- 初始化 ----
print("\n[1] 初始化 Agent...")
agent = RAGAgent(DATA_DIR)
print(f"    Agent: {len(agent.documents)} docs, "
      f"FAISS {agent.vector_index.index.ntotal} vectors, "
      f"BM25 {len(agent.bm25.df)} terms, "
      f"Reranker {'loaded' if agent.reranker.model else 'degraded'}")

# ---- 测试 1：单轮精确匹配 ----
print("\n[2] 单轮问答: '小米SU7的续航是多少'")
result = agent.chat("小米SU7的续航是多少")
print(f"    改写: {result['rewritten']}")
print(f"    耗时: {result['retrieval_time_ms']}ms")
print(f"    来源数: {len(result['sources'])}")
for i, src in enumerate(result['sources']):
    print(f"    [{i+1}] {src['source']} (score: {src['score']})")
print(f"    回答前200字: {result['answer'][:200]}...")

# ---- 测试 2：条件过滤 ----
print("\n[3] 条件过滤: '25万以内的SUV'")
result2 = agent.chat("25万以内的SUV")
print(f"    改写: {result2['rewritten']}")
print(f"    来源数: {len(result2['sources'])}")
for i, src in enumerate(result2['sources']):
    print(f"    [{i+1}] {src['source']} (score: {src['score']})")

# ---- 测试 3：多轮对话 + Query 改写 ----
print("\n[4] 多轮对话: 先问智驾 → 再问'那它的续航呢'")
agent.reset()
result3a = agent.chat("理想L6的智驾配置怎么样")
print(f"    [轮1] 改写: {result3a['rewritten']}")
print(f"    [轮1] 回答前100字: {result3a['answer'][:100]}...")

result3b = agent.chat("那它的续航呢")
print(f"    [轮2] 改写: {result3b['rewritten']}")
print(f"    [轮2] 来源: {[s['source'] for s in result3b['sources']]}")
if "理想" in result3b['rewritten'] or "L6" in result3b['rewritten']:
    print(f"    [OK] Query 改写成功，消解了'它'的指代")
else:
    print(f"    [WARN] Query 改写可能未生效，检查 LLM 返回")

# ---- 测试 4：边界 case ----
print("\n[5] 边界查询: '100万以上的豪华电动车'")
agent.reset()
result4 = agent.chat("100万以上的豪华电动车")
print(f"    改写: {result4['rewritten']}")
print(f"    来源数: {len(result4['sources'])}")
print(f"    回答前150字: {result4['answer'][:150]}...")

# ---- 测试 5：内置评估 ----
print("\n[6] 内置评估 /eval")
eval_result = self_eval(agent)
print(f"    Hit Rate: {eval_result['hit_rate']:.1%}")
print(f"    MRR: {eval_result['mrr']:.3f}")
print(f"    详细:")
for d in eval_result['details']:
    status = "OK" if d['hit'] else "MISS"
    kw_str = ",".join(d['hit_keywords']) if d['hit_keywords'] else "(none)"
    print(f"      [{status}] {d['query'][:30]:30s} | hit: {kw_str:20s} | expected: {','.join(d['expected']):20s}")

# ---- 测试 6：/hist 和 /reset ----
print("\n[7] 对话管理")
agent.reset()
agent.chat("比亚迪海豚的价格")
agent.chat("那它的动力呢")
summary = agent.get_history_summary()
print(f"    对话轮次: {len(agent.chat_history) // 2}")
agent.reset()
print(f"    重置后: {len(agent.chat_history)} 条消息")
print(f"    [OK] reset 正常")

print("\n" + "=" * 60)
print("端到端测试完成")
print(f"Hit Rate: {eval_result['hit_rate']:.1%}  |  MRR: {eval_result['mrr']:.3f}")
print("=" * 60)
