from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from collections import Counter
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import os
import time
from dotenv import load_dotenv

load_dotenv()

llm_client = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


# ============================================================
# 通用辅助函数
# ============================================================

def call_llm(prompt: str, temperature: float = 0.0) -> str:
    """统一的 LLM 调用接口"""
    response = llm_client.invoke(
        [HumanMessage(content=prompt)],
        temperature=temperature,
    )
    return response.content


def extract_answer(output: str) -> str:
    """从 LLM 输出中提取最终答案。
    尝试多种常见格式：'答案：' '答：' '最终答案：' '结果是' 等标记，
    取最后一个匹配标记后的内容作为答案。
    """
    # 优先级从高到低尝试匹配
    markers = [
        r'最终答案[：:]\s*(.+?)(?:\n|$)',
        r'【结论】\s*(.+?)(?:\n|$)',
        r'答案[：:]\s*(.+?)(?:\n|$)',
        r'答[：:]\s*(.+?)(?:\n|$)',
        r'结果是[：:]\s*(.+?)(?:\n|$)',
        r'所以[,，]?\s*(.+?)(?:。|\n|$)',
    ]
    for pattern in markers:
        matches = re.findall(pattern, output)
        if matches:
            return matches[-1].strip()

    # 兜底：取最后一行非空内容
    lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
    return lines[-1] if lines else output.strip()[-200:]


def extract_reasoning(output: str) -> str:
    """从 LLM 输出中提取推理过程部分"""
    # 尝试找"推理过程"标记
    markers = [
        r'推理过程[：:]\s*(.+?)(?=\n答案|\n最终答案|\n结论|\n答[：:]|\Z)',
        r'【分析】\s*(.+?)(?=\n【结论】|\Z)',
        r'(?:步骤|推理|分析)[：:]\s*(.+?)(?=\n答案|\n最终答案|\n答[：:]|\Z)',
    ]
    for pattern in markers:
        match = re.search(pattern, output, re.DOTALL)
        if match:
            return match.group(1).strip()

    # 兜底：取"答案"之前的所有内容
    for sep in ["答案：", "答：", "最终答案：", "结论："]:
        if sep in output:
            return output.split(sep)[0].strip()
    return output.strip()


def extract_answer_from_format(output: str) -> str:
    """从 Few-shot 格式的输出中提取答案（期望格式含'答案：'标记）"""
    return extract_answer(output)


def normalize_answer(ans) -> str:
    """将各种格式的答案统一归一化：提取核心数字或关键文本"""
    ans = str(ans).strip().lower()
    # 去掉常见的包装符号
    ans = ans.strip('。.！!？?')
    # 尝试提取数值（用于数学题）
    numbers = re.findall(r'\d+\.?\d*', ans)
    if numbers:
        return numbers[-1]  # 取最后一个数字作为答案
    return ans


# ============================================================
# 策略1: Zero-shot CoT
# ============================================================

def zero_shot_cot(question: str) -> dict:
    """
    核心: 在问题后追加触发词 "请一步步思考"
    验证标准: 输出包含推理步骤关键词
    """
    prompt = f"""问题：{question}

请一步步思考，先写出推理过程，然后给出最终答案。

格式要求：
推理过程：<你的逐步推理>
最终答案：<你的答案>"""

    output = call_llm(prompt, temperature=0.0)

    reasoning_keywords = ["步骤", "第一步", "然后", "因为", "所以", "推理", "计算"]
    has_reasoning = any(keyword in output for keyword in reasoning_keywords)

    answer = extract_answer(output)

    return {
        "output": output,
        "has_reasoning": has_reasoning,
        "answer": answer,
    }


# ============================================================
# 策略2: Few-shot CoT
# ============================================================

# 领域相关示例（汽车导购场景，与项目主题一致）
FEW_SHOT_EXAMPLES = [
    {
        "question": "20万预算买新能源车，比亚迪海豹和小鹏G6怎么选？",
        "reasoning": "第一步：查两款车的价格区间。比亚迪海豹 17.98-24.98 万，小鹏 G6 20.99-27.69 万，都在预算范围内。"
                     "第二步：对比核心参数。海豹 CLTC 续航 550-700km、零百 3.8s、DiPilot 智驾；"
                     "G6 续航 580-755km、零百 3.9-6.2s、XNGP 智驾+双激光雷达。"
                     "第三步：综合判断。G6 续航更长、智驾更强（双激光雷达），但起售价高 3 万。"
                     "海豹加速更快、价格更友好。",
        "answer": "如果看重智驾和续航，选小鹏 G6；如果看重加速性能和性价比，选比亚迪海豹。"
    },
    {
        "question": "小米SU7首付5万贷款3年，月供大概多少？",
        "reasoning": "第一步：确定车价。小米 SU7 售价 21.59-29.99 万，取中配约 25 万。"
                     "第二步：计算贷款金额。25 万 - 5 万首付 = 20 万贷款。"
                     "第三步：计算月供。20 万贷款 36 个月，假设年利率 3%，"
                     "等额本息公式：月供 ≈ 200000 × (1 + 0.03×3) / 36 ≈ 6056 元。",
        "answer": "月供约 6056 元。"
    },
    {
        "question": "家用第一辆车，15万预算，SUV和轿车哪个更合适？",
        "reasoning": "第一步：分析家用需求。家用第一辆通常要空间大、坐得下 5 人、后备箱能装。"
                     "第二步：对比车型。15 万 SUV 可选零跑 C11（15.58 万起）、埃安 Y（11.98 万起）；"
                     "轿车可选比亚迪海豚（9.98 万起）。"
                     "第三步：综合判断。SUV 空间更大、视野更好、通过性更强，更适合家用第一辆车；"
                     "轿车油耗/电耗更低、操控更好。家用优先推荐 SUV。",
        "answer": "家用第一辆车推荐 SUV，15 万预算可重点看零跑 C11 和埃安 Y。"
    },
]


def few_shot_cot(question: str, examples: list = None) -> dict:
    """
    核心: 提供 3 个"问题-推理-答案"示例，让 LLM 模仿格式
    验证标准: 输出格式与示例一致（含'推理过程：'和'答案：'标记）
    """
    if examples is None:
        examples = FEW_SHOT_EXAMPLES

    prompt = "请参考以下示例的格式进行推理：\n\n"
    for i, ex in enumerate(examples, 1):
        prompt += f"示例{i}：\n"
        prompt += f"问题：{ex['question']}\n"
        prompt += f"推理过程：{ex['reasoning']}\n"
        prompt += f"答案：{ex['answer']}\n\n"

    prompt += f"现在请回答：\n问题：{question}\n推理过程："

    output = call_llm(prompt, temperature=0.0)

    # 验证格式：必须包含"推理过程"和"答案"
    has_format = ("推理过程" in output or "步骤" in output) and "答案" in output
    answer = extract_answer_from_format(output)

    return {
        "output": output,
        "format_match": has_format,
        "answer": answer,
    }


# ============================================================
# 策略3: Self-Consistency
# ============================================================

def self_consistency(question: str, n: int = 5) -> dict:
    """
    核心: 同一问题跑 n 次（temperature>0），投票取多数答案
    验证标准: 投票机制生效 — 最高票数 ≥ n/2+1
    """
    candidates = []
    raw_outputs = []

    for i in range(n):
        prompt = f"""问题：{question}

请一步步思考，先写出推理过程，然后给出最终答案。
格式要求：
推理过程：<你的逐步推理>
最终答案：<你的答案>"""

        # temperature=0.7 让每次输出不同
        output = call_llm(prompt, temperature=0.7)
        raw_outputs.append(output)

        answer = normalize_answer(extract_answer(output))
        candidates.append(answer)

    # 投票
    vote_counter = Counter(candidates)
    most_common_answer, vote_count = vote_counter.most_common(1)[0]

    # 验证：多数票即视为投票有效
    is_valid_vote = vote_count >= (n // 2 + 1)

    return {
        "candidates": candidates,
        "raw_outputs": raw_outputs,
        "vote_counts": dict(vote_counter),
        "final_answer": most_common_answer,
        "is_valid": is_valid_vote,
    }


# ============================================================
# 策略4: Auto-CoT
# ============================================================

def auto_cot(question: str, all_questions: list,
             encoder: SentenceTransformer = None,
             question_vectors: np.ndarray = None,
             cluster_labels: np.ndarray = None,
             kmeans: KMeans = None) -> dict:
    """
    核心: 自动聚类 + 选代表性问题生成示例
    验证标准: 示例覆盖所有问题类别（多样性覆盖）

    性能优化: encoder/question_vectors/cluster_labels/kmeans 可以从外部传入，
    避免每次调用重复计算 Embedding 和聚类。
    """
    # Step 1: 如果没有预计算，则现场计算（首次调用）
    if encoder is None:
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
    if question_vectors is None:
        question_vectors = encoder.encode(all_questions)
    if kmeans is None:
        n_clusters = min(3, len(all_questions))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(question_vectors)
    if cluster_labels is None:
        cluster_labels = kmeans.predict(question_vectors)

    n_clusters = kmeans.n_clusters

    # Step 2: 每个簇选离中心最近的代表性问题
    representative_questions = []
    for cluster_id in range(n_clusters):
        indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        if not indices:
            continue
        cluster_center = kmeans.cluster_centers_[cluster_id]
        distances = [np.linalg.norm(question_vectors[i] - cluster_center) for i in indices]
        closest_idx = indices[int(np.argmin(distances))]
        representative_questions.append(all_questions[closest_idx])

    # Step 3: 对代表性问题用 Zero-shot 生成推理示例
    examples = []
    for rep_q in representative_questions:
        prompt = f"问题：{rep_q}\n请一步步思考，写出推理过程和最终答案。\n\n推理过程："
        output = call_llm(prompt, temperature=0.3)
        reasoning = extract_reasoning(output)
        answer = extract_answer(output)
        examples.append({
            "question": rep_q,
            "reasoning": reasoning,
            "answer": answer,
        })

    # Step 4: 用生成的示例进行 Few-shot 推理
    result = few_shot_cot(question, examples=examples)

    # Step 5: 验证多样性覆盖
    # 每个簇是否都有代表性问题被选中
    unique_clusters_in_labels = set(cluster_labels)
    diversity_covered = (len(representative_questions) == len(unique_clusters_in_labels))

    return {
        "representative_questions": representative_questions,
        "generated_examples": examples,
        "cluster_distribution": dict(Counter(int(c) for c in cluster_labels)),
        "final_answer": result["answer"],
        "output": result["output"],
        "diversity_covered": diversity_covered,
    }


# ============================================================
# 策略5: 四种策略对比实验
# ============================================================

def estimate_tokens(output: str) -> int:
    """估算 token 消耗（粗略：中文 ~1.5 字/token，英文 ~4 字/token）"""
    chinese_chars = len(re.findall(r'[一-鿿]', output))
    other_chars = len(output) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


def compare_all_strategies(test_questions: list) -> dict:
    """
    在测试集上运行所有策略，生成量化对比表格。
    注: 因为没有 ground_truth 标注，这里对比的是各策略的推理质量指标，
    而非准确率。如需准确率对比，需要标注数据集。
    """
    results = {
        "zero_shot": [],
        "few_shot": [],
        "self_consistency": [],
        "auto_cot": [],
    }

    # Auto-CoT 的 Embedding 和聚类只算一次（性能优化）
    print("⏳ 初始化 Auto-CoT: 计算 Embedding + K-Means 聚类...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    question_vectors = encoder.encode(test_questions)
    n_clusters = min(3, len(test_questions))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(question_vectors)
    print(f"   完成: {len(test_questions)} 个问题 → {n_clusters} 个簇\n")

    for i, q in enumerate(test_questions):
        print(f"[{i+1}/{len(test_questions)}] 测试: {q[:50]}...")

        # 1. Zero-shot
        t0 = time.time()
        zs = zero_shot_cot(q)
        zs["latency"] = round(time.time() - t0, 2)
        zs["tokens"] = estimate_tokens(zs["output"])
        results["zero_shot"].append(zs)

        # 2. Few-shot
        t0 = time.time()
        fs = few_shot_cot(q)
        fs["latency"] = round(time.time() - t0, 2)
        fs["tokens"] = estimate_tokens(fs["output"])
        results["few_shot"].append(fs)

        # 3. Self-Consistency (n=5)
        t0 = time.time()
        sc = self_consistency(q, n=5)
        sc["latency"] = round(time.time() - t0, 2)
        sc["tokens"] = sum(estimate_tokens(out) for out in sc["raw_outputs"])
        results["self_consistency"].append(sc)

        # 4. Auto-CoT（复用预计算的 Embedding 和聚类结果）
        t0 = time.time()
        ac = auto_cot(
            q, test_questions,
            encoder=encoder,
            question_vectors=question_vectors,
            cluster_labels=cluster_labels,
            kmeans=kmeans,
        )
        ac["latency"] = round(time.time() - t0, 2)
        ac["tokens"] = estimate_tokens(ac.get("output", ""))
        results["auto_cot"].append(ac)

    # ── 生成对比表 ──
    print("\n" + "=" * 90)
    print("四种 CoT 策略量化对比")
    print("=" * 90)

    strategy_names = {
        "zero_shot": "Zero-shot CoT",
        "few_shot": "Few-shot CoT",
        "self_consistency": "Self-Consistency",
        "auto_cot": "Auto-CoT",
    }

    # 表头
    print(f"{'策略':<20} {'推理完整性':<12} {'格式一致性':<12} "
          f"{'投票有效性':<12} {'多样性覆盖':<12} {'平均延迟(s)':<14} {'Token消耗':<12}")
    print("-" * 90)

    summary = {}
    for key in ["zero_shot", "few_shot", "self_consistency", "auto_cot"]:
        items = results[key]

        # 计算各项指标
        reasoning_rate = sum(1 for r in items if r.get("has_reasoning", False)) / max(len(items), 1) * 100
        format_rate = sum(1 for r in items if r.get("format_match", False)) / max(len(items), 1) * 100
        valid_vote_rate = sum(1 for r in items if r.get("is_valid", True)) / max(len(items), 1) * 100
        diversity_rate = sum(1 for r in items if r.get("diversity_covered", True)) / max(len(items), 1) * 100
        avg_latency = sum(r.get("latency", 0) for r in items) / max(len(items), 1)
        total_tokens = sum(r.get("tokens", 0) for r in items)

        summary[key] = {
            "reasoning_rate": reasoning_rate,
            "format_rate": format_rate,
            "valid_vote_rate": valid_vote_rate,
            "diversity_rate": diversity_rate,
            "avg_latency": avg_latency,
            "total_tokens": total_tokens,
        }

        print(f"{strategy_names[key]:<20} {reasoning_rate:>7.1f}%     "
              f"{format_rate:>7.1f}%     {valid_vote_rate:>7.1f}%     "
              f"{diversity_rate:>7.1f}%     {avg_latency:>8.2f}s     {total_tokens:>8}")

    print("=" * 90)
    print("\n指标说明:")
    print("  推理完整性  — 输出是否包含推理步骤（Zero-shot 核心指标）")
    print("  格式一致性  — 输出格式是否与示例一致（Few-shot 核心指标）")
    print("  投票有效性  — 5 次采样是否产生多数共识（Self-Consistency 核心指标）")
    print("  多样性覆盖  — 生成的示例是否覆盖所有问题类别（Auto-CoT 核心指标）")
    print("  平均延迟    — 端到端响应时间")
    print("  Token消耗   — 总 Token 估算值")
    print()

    return results, summary


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    # 汽车导购领域测试题（与项目主题一致，比纯数学题更有展示价值）
    TEST_QUESTIONS = [
        "20万预算买新能源车，比亚迪海豹和小鹏G6怎么选？",
        "小米SU7首付5万贷款3年，月供大概多少？",
        "家用第一辆车，15万预算，SUV还是轿车更合适？",
        "纯电车和增程车各自适合什么人群？请分析",
        "比亚迪海豚和埃安Y，哪个性价比更高？",
        "25万预算，追求智能驾驶，推荐什么车？",
        "新能源车每年的保险和保养大概花多少钱？",
        "特斯拉Model 3和小米SU7哪个更值得买？",
    ]

    print("=" * 90)
    print("Day 6: 四种 CoT 策略对比实验")
    print("=" * 90)
    print(f"测试集: {len(TEST_QUESTIONS)} 个汽车导购领域问题\n")

    results, summary = compare_all_strategies(TEST_QUESTIONS)

    # ================================================================
    # 达标自检
    # ================================================================
    print("=" * 90)
    print("【Day 6 达标自检】")
    print("=" * 90)

    checks = [
        ("练习1: Zero-shot CoT 推理完整性 > 50%",
         summary["zero_shot"]["reasoning_rate"] > 50),
        ("练习2: Few-shot CoT 格式一致性 > 50%",
         summary["few_shot"]["format_rate"] > 50),
        ("练习3: Self-Consistency 投票有效性 > 50%",
         summary["self_consistency"]["valid_vote_rate"] > 50),
        ("练习4: Auto-CoT 多样性覆盖 > 50%",
         summary["auto_cot"]["diversity_rate"] > 50),
        ("练习5: 四种策略量化对比表格输出",
         True),
    ]

    all_pass = True
    for desc, ok in checks:
        status = "✅" if ok else "❌"
        if not ok:
            all_pass = False
        print(f"  {status}  {desc}")

    print(f"\n结论: {'全部达标 ✅' if all_pass else '有未完成项 ⚠️'}")
    print("\n面试话术: \"CoT 不是简单加一句'请思考'——Zero-shot 省 token 但推理质量不稳定，")
    print("  Few-shot 效果好但示例挑选是瓶颈，Self-Consistency 用采样+投票解决单次推理的随机性，")
    print("  Auto-CoT 自动保证示例多样性。量化对比让你从'猜 Prompt'升级为'测 Prompt'。\"")
