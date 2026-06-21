"""
Day 7: 动态 Few-shot 选择器
============================
练习1: KeywordSelector   — 基于关键词重叠选择 Top-K 示例
练习2: SemanticSelector  — 基于 Embedding 语义相似度选择 Top-K 示例
练习3: MMRSelector       — MMR 算法平衡相似度 + 多样性
练习4: 三种选择器对比    — 同一查询对比三种选择器的输出 + 平均相似度

面试话术: "Few-shot 示例不是越多越好，关键是多样性和代表性。
MMR 算法在语义相关度和示例间差异度之间做了平衡，避免选出的 3 个示例回答思路完全一样。"
"""

import re
import os
import time
from collections import Counter

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# ═══════════════════════════════════════════════════════════════
# 环境 & LLM 初始化（无需修改）
# ═══════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

llm_client = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


def call_llm(prompt: str, temperature: float = 0.0) -> str:
    """统一的 LLM 调用接口"""
    response = llm_client.invoke(
        [HumanMessage(content=prompt)],
        temperature=temperature,
    )
    return response.content


# ═══════════════════════════════════════════════════════════════
# 默认 Embedding 模型（无需修改）
# ═══════════════════════════════════════════════════════════════

# 使用 BGE 中文模型（768 维），比 all-MiniLM-L6-v2（英文优化）更适合中文场景
# 国内网络无法访问 HuggingFace 时，可替换为本地缓存的其他模型
DEFAULT_ENCODER_NAME = 'BAAI/bge-base-zh-v1.5'


# ═══════════════════════════════════════════════════════════════
# 通用辅助函数（无需修改）
# ═══════════════════════════════════════════════════════════════

def tokenize_chinese(text: str) -> set:
    """简单中文分词：提取中文字段、英文单词、数字作为 token"""
    tokens = re.findall(r'[一-鿿]+|[a-zA-Z]+|\d+', text)
    # 去重 + 去停用词
    stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
                 '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
                 '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些',
                 '什么', '怎么', '如何', '哪个', '多少', '可以', '还是', '应该'}
    return {t for t in tokens if t not in stopwords}


# ═══════════════════════════════════════════════════════════════
# 示例库（无需修改 — 覆盖汽车导购 5 种推理类型）
# ═══════════════════════════════════════════════════════════════

EXAMPLE_POOL = [
    # ── 对比型（4 条）──
    {
        "question": "20万预算买新能源车，比亚迪海豹和小鹏G6怎么选？",
        "reasoning": "第一步：查价格。海豹 17.98-24.98 万，G6 20.99-27.69 万，都在预算内。"
                     "第二步：比续航。海豹 550-700km vs G6 580-755km，G6 略胜。"
                     "第三步：比智驾。海豹 DiPilot vs G6 XNGP+双激光雷达，G6 明显更强。"
                     "第四步：综合。G6 智驾和续航更强但贵 3 万，海豹加速更好、性价比更高。",
        "answer": "看重智驾选 G6，看重性价比选海豹。",
    },
    {
        "question": "特斯拉Model 3和小米SU7哪个更值得买？",
        "reasoning": "第一步：价格。Model 3 23.19-33.59 万，SU7 21.59-29.99 万，SU7 更便宜。"
                     "第二步：续航。Model 3 556-713km vs SU7 700-830km，SU7 更长。"
                     "第三步：智驾。Model 3 FSD vs SU7 Orin-X+激光雷达，FSD 更成熟但国内受限。"
                     "第四步：生态。特斯拉充电网络更完善，小米有手机×汽车联动。",
        "answer": "看重续航和性价比选 SU7，看重品牌和充电便利选 Model 3。",
    },
    {
        "question": "理想L6和问界M7怎么选？",
        "reasoning": "第一步：类型。两者都是增程 SUV，L6 24.98-27.98 万，M7 24.98-32.98 万。"
                     "第二步：空间。L6 中大型 5 座，M7 有 5/6 座可选，M7 更灵活。"
                     "第三步：智驾。L6 AD Max vs M7 ADS 2.0，都是第一梯队，各有千秋。",
        "answer": "需要 6 座选 M7，5 座够用且追求性价比选 L6。",
    },
    {
        "question": "比亚迪海豚和埃安Y哪个性价比更高？",
        "reasoning": "第一步：价格。海豚 9.98-13.98 万，埃安 Y 11.98-18.98 万，海豚起售更低。"
                     "第二步：续航。海豚 301-405km vs 埃安 Y 430-610km，埃安 Y 明显更长。"
                     "第三步：空间。海豚是小型车，埃安 Y 是紧凑型 SUV，埃安 Y 空间更大。",
        "answer": "纯代步选海豚（更便宜），家用需要空间选埃安 Y（续航+空间更好）。",
    },

    # ── 计算型（3 条）──
    {
        "question": "小米SU7首付5万贷款3年，月供大概多少？",
        "reasoning": "第一步：定车价。SU7 售价 21.59-29.99 万，取中配约 25 万。"
                     "第二步：贷款额。25 - 5 = 20 万。"
                     "第三步：算月供。20 万 ÷ 36 月 ≈ 5556 元/月（不含利息），"
                     "加 3% 年利率简单算法：20×1.09/36 ≈ 6056 元/月。",
        "answer": "月供约 6000-6100 元。",
    },
    {
        "question": "养一辆20万的新能源车一年花多少钱？",
        "reasoning": "第一步：保险。20 万车价 × 3% ≈ 6000 元/年。"
                     "第二步：充电。年 2 万公里 × 0.3 元/km = 6000 元/年。"
                     "第三步：保养。新能源车保养简单，约 1000-2000 元/年。"
                     "第四步：加总 ≈ 1.3-1.4 万/年。",
        "answer": "年均用车成本约 1.3-1.5 万元。",
    },
    {
        "question": "15万的车落地要多少钱？",
        "reasoning": "第一步：裸车 15 万。"
                     "第二步：新能源免购置税，燃油车约 1.3 万。"
                     "第三步：保险约 4500 元，上牌 500 元。"
                     "第四步：新能源落地 ≈ 15.5 万，燃油车落地 ≈ 16.8 万。",
        "answer": "新能源落地约 15.5 万，燃油车约 16.8 万。",
    },

    # ── 推荐型（3 条）──
    {
        "question": "15万预算买纯电轿车，有什么推荐？",
        "reasoning": "第一步：筛选。15 万左右纯电轿车：比亚迪海豚 9.98-13.98 万、埃安 Y 11.98-18.98 万。"
                     "第二步：对比。海豚起售价更低且比亚迪品牌更成熟，埃安 Y 续航和空间更好。"
                     "第三步：推荐。海豚 405km 版 12 万落地，适合城市代步。",
        "answer": "首推比亚迪海豚 405km 版，预算稍多可看埃安 Y。",
    },
    {
        "question": "25万预算，看重智能驾驶，推荐什么车？",
        "reasoning": "第一步：筛选。25 万 + 强智驾：小鹏 G6（XNGP+双激光雷达）、问界 M7（ADS 2.0）。"
                     "第二步：对比。G6 20.99 万起智驾标配，M7 智驾版 28.98 万起超预算。"
                     "第三步：推荐。25 万预算内小鹏 G6 是最优选择，XNGP 城市+高速全场景。",
        "answer": "强烈推荐小鹏 G6，25 万预算智驾最强选择。",
    },
    {
        "question": "家用第一辆车，20万以内SUV推荐",
        "reasoning": "第一步：定位。家用第一辆 → 空间、安全、可靠性优先。"
                     "第二步：筛选。20 万内 SUV：零跑 C11 15.58-19.98 万、埃安 Y 11.98-18.98 万、比亚迪宋系列。"
                     "第三步：对比。C11 空间最大+配置最高，埃安 Y 最便宜，宋系列品牌最稳。",
        "answer": "首推零跑 C11（空间+配置），保守选比亚迪宋。",
    },

    # ── 决策型（3 条）──
    {
        "question": "家用第一辆车，SUV和轿车哪个更合适？",
        "reasoning": "第一步：分析需求。家用要空间大、坐 5 人、后备箱能装婴儿车。"
                     "第二步：SUV 优势。空间大、视野好、底盘高通过性好、后备箱开口大装东西方便。"
                     "第三步：轿车优势。操控好、电耗更低、价格通常更便宜。"
                     "第四步：结论。家用优先 SUV，空间是刚需。",
        "answer": "家用推荐 SUV，空间是家用第一刚需。",
    },
    {
        "question": "纯电车和增程车各自适合什么人群？",
        "reasoning": "第一步：纯电适合有家充桩、日常通勤 < 100km、偶尔长途 ≤ 300km 的人。"
                     "第二步：增程适合无固定充电条件、经常跑长途（> 500km）、有里程焦虑的人。"
                     "第三步：成本对比。纯电用车成本更低（0.3 元/km vs 0.6 元/km），增程购车价通常更低。",
        "answer": "有家充桩选纯电，经常跑长途选增程。",
    },
    {
        "question": "等新车还是现在买？新能源车降价太快了",
        "reasoning": "第一步：新能源车确实在降价周期，但降幅在收窄（从 20%/年降到 5-10%/年）。"
                     "第二步：等待成本。等半年省 1-2 万 vs 半年没车用的便利性损失。"
                     "第三步：建议。刚需（没车用/旧车要换）现在买，非刚需可以等到年底冲量季。",
        "answer": "刚需现在买，不急可以等到 11-12 月冲量季。",
    },

    # ── 参数查询型（3 条）──
    {
        "question": "小鹏G6的智驾能力怎么样？",
        "reasoning": "第一步：硬件。双激光雷达 + Orin-X 芯片（254TOPS）+ 12 个摄像头。"
                     "第二步：功能。XNGP 覆盖高速+城区，支持点到点领航。"
                     "第三步：对比。同价位智驾最强之一，与华为 ADS 2.0 处于第一梯队。",
        "answer": "小鹏 G6 的 XNGP 是同价位智驾最强的选择之一。",
    },
    {
        "question": "CLTC续航和实际续航差多少？",
        "reasoning": "第一步：CLTC 是中国标准测试工况，偏理想化（低速+常温）。"
                     "第二步：实际续航通常打 6-8 折。高速 120km/h 打 6-7 折，城市路况打 7-8 折。"
                     "第三步：冬季低温再额外打 1-2 折（开暖风耗电）。",
        "answer": "实际续航 ≈ CLTC × 0.7（综合），高速 ≈ CLTC × 0.6，冬季更差。",
    },
    {
        "question": "新能源车保险为什么比燃油车贵？",
        "reasoning": "第一步：新能源车出险率高。加速快、新手多、维修贵（一体压铸+电池包）。"
                     "第二步：电池包是高风险部件，碰撞后整包更换费用高（5-15 万）。"
                     "第三步：专属险种增加了三电系统保障，保费自然上浮 10-20%。",
        "answer": "新能源车出险率高+电池维修贵，保费比同价位燃油车贵 10-20%。",
    },
]


# ═══════════════════════════════════════════════════════════════
# 练习 1: KeywordSelector — 关键词匹配选择器
# ═══════════════════════════════════════════════════════════════

class KeywordSelector:
    """基于关键词重叠的选择器 — 选与 query 共同词最多的 K 个示例。

    你的任务：实现 select() 方法。
    思路：
      1. 用 tokenize_chinese() 对 query 分词
      2. 对 pool 中每个示例的 question 也分词
      3. 计算交集大小（重叠词数）作为得分
      4. 按得分降序排序，返回 Top-K 个示例
    """

    def __init__(self, example_pool: list):
        """
        Args:
            example_pool: 示例列表，每个元素是 {"question", "reasoning", "answer"} 字典
        """
        self.pool = example_pool

    def select(self, query: str, k: int = 3) -> list:
        """选与 query 关键词重叠最多的 k 个示例。

        Args:
            query: 用户当前问题
            k: 需要选择的示例数量

        Returns:
            得分最高的 k 个示例（list of dict）
        """
        query_tokens = tokenize_chinese(query)
        
        scores = []
        for i, example in enumerate(self.pool):
            example_tokens = tokenize_chinese(example["question"])
            overlap = len(query_tokens & example_tokens)
            scores.append((overlap, example))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scores[:k]]
        


# ═══════════════════════════════════════════════════════════════
# 练习 2: SemanticSelector — 语义相似度选择器
# ═══════════════════════════════════════════════════════════════

class SemanticSelector:
    """基于 Embedding 语义相似度的选择器 — 选与 query 余弦相似度最高的 K 个示例。

    你的任务：实现 select() 方法。
    思路：
      1. 用 self.encoder 编码 query（得到 768 维向量）
      2. 计算 query 向量与 self.example_vectors（已预计算）的余弦相似度
      3. 取相似度最高的 K 个索引

    提示：余弦相似度 = dot(A, B) / (|A| × |B|)
          np.argsort 可用来找 Top-K 索引，[::-1] 实现降序
    """

    def __init__(self, example_pool: list, encoder: SentenceTransformer = None):
        """
        Args:
            example_pool: 示例列表
            encoder: Sentence-BERT 编码器，默认 BGE 中文模型
        """
        self.pool = example_pool
        self.encoder = encoder or SentenceTransformer(DEFAULT_ENCODER_NAME)

        # 预计算：所有示例的向量化（只需算一次，之后每次 select 复用）
        self.example_texts = [ex["question"] for ex in self.pool]
        self.example_vectors = self.encoder.encode(self.example_texts)
        # shape: (N, 768)

    def select(self, query: str, k: int = 3) -> list:
        """选与 query 语义最相近的 k 个示例。

        Returns:
            相似度最高的 k 个示例
        """
        # 1. 编码 query
        query_vec = self.encoder.encode([query])[0]
        
        # 2. 计算余弦相似度（向量化操作，同时算 query 与所有示例）
        dot = np.dot(self.example_vectors, query_vec)
        q_norm = np.linalg.norm(query_vec)
        e_norms = np.linalg.norm(self.example_vectors, axis=1)
        similarities = dot / (q_norm * e_norms + 1e-8)
        
        # 3. 取 Top-K 索引
        top_k = np.argsort(similarities)[::-1][:k]  # argsort 升序 → [::-1] 降序
        return [self.pool[i] for i in top_k]


# ═══════════════════════════════════════════════════════════════
# 练习 3: MMRSelector — 最大边际相关度选择器
# ═══════════════════════════════════════════════════════════════

class MMRSelector:
    """MMR 选择器 — 兼顾语义相似度（相关性）+ 示例间差异（多样性）。

    你的任务：实现 select() 方法。
    算法：贪心逐步选择。
      第 1 个：选 Sim(query, d_i) 最大的（此时已选集合为空，多样性惩罚=0）
      第 2 个：选 argmax[ λ·Sim(query, d_i) - (1-λ)·Sim(d_i, 第1个) ]
               既要和 query 相关，又不能和第 1 个太像
      第 3 个：选 argmax[ λ·Sim(query, d_i) - (1-λ)·max(Sim(d_i, 已选中的每个)) ]
               既要相关，又不能和任何已选示例太像

    MMR 公式: score(i) = λ × relevance(i) - (1-λ) × max_sim_to_selected(i)

    λ 参数：
      λ=1   → 退化为纯语义选择器（只看相关性）
      λ=0   → 退化为纯多样性选择器（不看相关性，只选彼此最不像的）
      λ=0.7 → 经验值，7 分给相关性，3 分给多样性
    """

    def __init__(self, example_pool: list, encoder: SentenceTransformer = None):
        self.pool = example_pool
        self.encoder = encoder or SentenceTransformer(DEFAULT_ENCODER_NAME)

        # 预计算：所有示例向量
        self.example_texts = [ex["question"] for ex in self.pool]
        self.example_vectors = self.encoder.encode(self.example_texts)
        self.N = len(self.pool)

        # 预计算：示例间的余弦相似度矩阵（只需算一次）
        # 归一化后，矩阵乘法直接得到相似度矩阵
        norms = np.linalg.norm(self.example_vectors, axis=1, keepdims=True)
        normalized = self.example_vectors / (norms + 1e-8)
        self._pairwise_sim = np.dot(normalized, normalized.T)
        # shape: (N, N)，_pairwise_sim[i][j] = 示例 i 和 j 的相似度

    def select(self, query: str, k: int = 3, lambda_: float = 0.7) -> list:
        """用 MMR 算法贪心选出 k 个示例。

        Args:
            query: 用户当前问题
            k: 需要选择的示例数量
            lambda_: 相关性权重（0~1）。λ 越大越看重相关性，越小越看重多样性。

        Returns:
            选中的 k 个示例
        """
        # 1. 计算 query 与所有示例的相似度
        query_vec = self.encoder.encode([query])[0]
        dot = np.dot(self.example_vectors, query_vec)
        q_norm = np.linalg.norm(query_vec)
        e_norms = np.linalg.norm(self.example_vectors, axis=1)
        sim_to_query = dot / (q_norm * e_norms + 1e-8)

        # 2. 贪心选择
        selected = []                    # 已选中的示例索引
        candidates = set(range(self.N))  # 候选池

        for _ in range(k):
            best_idx = None
            best_score = -float('inf')

            for i in candidates:
                # 相关性奖励
                relevance = lambda_ * sim_to_query[i]

                # 多样性惩罚：与已选示例的最大相似度
                diversity_penalty = 0
                if selected:
                    max_sim = max(self._pairwise_sim[i][j] for j in selected)
                    diversity_penalty = (1 - lambda_) * max_sim

                mmr_score = relevance - diversity_penalty

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(best_idx)
            candidates.remove(best_idx)

        return [self.pool[i] for i in selected]


# ═══════════════════════════════════════════════════════════════
# 练习 4: 三种选择器对比
# ═══════════════════════════════════════════════════════════════

def compute_avg_pairwise_sim(examples: list,
                              encoder: SentenceTransformer = None) -> float:
    """计算 K 个示例两两之间的平均余弦相似度（衡量多样性）。

    越低说明示例间差异越大、多样性越好。
    如果 examples 少于 2 个，直接返回 0。

    Args:
        examples: 选中的 K 个示例列表
        encoder: Embedding 编码器

    Returns:
        平均两两相似度（0~1，越低表示多样性越好）
    """
    if len(examples) < 2:
        return 0.0

    if encoder is None:
        encoder = SentenceTransformer(DEFAULT_ENCODER_NAME)

    # 编码所有选中示例
    texts = [ex["question"] for ex in examples]
    vectors = encoder.encode(texts)

    # 归一化 → 余弦相似度矩阵
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / (norms + 1e-8)
    sim_matrix = np.dot(normalized, normalized.T)
    # sim_matrix[i][j] = 示例 i 和 j 的余弦相似度

    # 取上三角（不含对角线），算均值
    n = len(examples)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    return float(np.mean(upper_tri))


def compare_selectors(query: str, pool: list, k: int = 3):
    """用同一个 query + pool，对比三种选择器的选出的示例及其多样性。

    输出每种选择器选出的 K 个示例的 question，以及平均两两相似度。
    预期结果：MMR 的平均两两相似度最低（多样性最好）。
    """
    encoder = SentenceTransformer(DEFAULT_ENCODER_NAME)

    kw_sel = KeywordSelector(pool)
    sem_sel = SemanticSelector(pool, encoder)
    mmr_sel = MMRSelector(pool, encoder)

    print(f"\n查询: {query}")
    print("=" * 60)

    selectors = {
        "关键词选择器": kw_sel,
        "语义选择器": sem_sel,
        "MMR选择器": mmr_sel,
    }

    results = {}
    for name, sel in selectors.items():
        selected = sel.select(query, k=k)
        results[name] = selected
        print(f"\n{name}:")
        for i, ex in enumerate(selected, 1):
            print(f"  {i}. {ex['question']}")

    # 多样性对比
    print(f"\n多样性对比（平均两两相似度，越低 -> 多样性越好）:")
    for name, selected in results.items():
        div = compute_avg_pairwise_sim(selected, encoder)
        print(f"  {name}: {div:.4f}")


# ═══════════════════════════════════════════════════════════════
# Day 6 + Day 7 串联 — 动态 Few-shot CoT（无需修改）
# ═══════════════════════════════════════════════════════════════

def dynamic_few_shot_cot(query: str, selector, k: int = 3) -> dict:
    """Day 6 Few-shot CoT + Day 7 动态选择器的串联。

    1. 用 selector 根据 query 动态选出 K 个最优示例
    2. 用这些示例构造 Few-shot Prompt，调 LLM 推理
    3. 返回答案 + 被选中的示例（用于可解释性调试）
    """
    examples = selector.select(query, k=k)

    # 构造 Few-shot Prompt（与 Day 6 few_shot_cot 逻辑一致）
    prompt = "请参考以下示例的格式进行推理：\n\n"
    for i, ex in enumerate(examples, 1):
        prompt += f"示例{i}：\n"
        prompt += f"问题：{ex['question']}\n"
        prompt += f"推理过程：{ex['reasoning']}\n"
        prompt += f"答案：{ex['answer']}\n\n"
    prompt += f"现在请回答：\n问题：{query}\n推理过程："

    output = call_llm(prompt, temperature=0.0)

    # 提取答案
    import re as _re
    answer = ""
    for pattern in [r'答案[：:]\s*(.+?)(?:\n|$)', r'最终答案[：:]\s*(.+?)(?:\n|$)']:
        matches = _re.findall(pattern, output)
        if matches:
            answer = matches[-1].strip()
            break
    if not answer:
        answer = output.strip().split('\n')[-1][:200]

    return {
        "output": output,
        "answer": answer,
        "selected_examples": [ex["question"] for ex in examples],
    }


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Day 7: 动态 Few-shot 选择器")
    print("=" * 70)
    print(f"示例库: {len(EXAMPLE_POOL)} 条汽车导购示例")
    print(f"覆盖类型: 对比型(4) 计算型(3) 推荐型(3) 决策型(3) 参数查询型(3)\n")

    # ── 测试查询 ──
    TEST_QUERIES = [
        "20-25万纯电SUV怎么选？看重智驾和续航",
        "比亚迪海豚和埃安Y哪个性价比高",
        "买新能源车每年要花多少钱保养",
    ]

    # ═══════════════════════════════════════════════════════
    # 场景 A: 三种选择器对比
    # ═══════════════════════════════════════════════════════
    print("=" * 70)
    print("【场景 A】三种选择器对比")
    print("=" * 70)

    kw_sel = KeywordSelector(EXAMPLE_POOL)
    sem_sel = SemanticSelector(EXAMPLE_POOL)
    mmr_sel = MMRSelector(EXAMPLE_POOL)

    for query in TEST_QUERIES:
        print(f"\n查询: {query}")
        print("-" * 50)

        # 关键词
        kw_result = kw_sel.select(query, k=3)
        print("关键词选择器:")
        for ex in kw_result:
            print(f"  - {ex['question']}")

        # 语义
        sem_result = sem_sel.select(query, k=3)
        print("语义选择器:")
        for ex in sem_result:
            print(f"  - {ex['question']}")

        # MMR
        mmr_result = mmr_sel.select(query, k=3)
        print("MMR 选择器:")
        for ex in mmr_result:
            print(f"  - {ex['question']}")

        # 多样性对比
        kw_div = compute_avg_pairwise_sim(kw_result)
        sem_div = compute_avg_pairwise_sim(sem_result)
        mmr_div = compute_avg_pairwise_sim(mmr_result)
        print(f"平均相似度: 关键词={kw_div:.3f}  语义={sem_div:.3f}  MMR={mmr_div:.3f}")
        if mmr_div < sem_div:
            print(f"  -> MMR 多样性比语义好 {sem_div/max(mmr_div, 0.01):.1f}x [PASS]")

    # ═══════════════════════════════════════════════════════
    # 场景 B: Day 6 串联 — 动态 Few-shot CoT
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("【场景 B】Day 6 + Day 7 串联 — 动态 Few-shot CoT")
    print("=" * 70)
    query = "25万预算看重智驾，推荐一款新能源SUV"
    result = dynamic_few_shot_cot(query, mmr_sel, k=3)
    print(f"查询: {query}")
    print(f"选中示例: {result['selected_examples']}")
    print(f"答案: {result['answer'][:200]}")

    # ═══════════════════════════════════════════════════════
    # 达标自检
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("【Day 7 达标自检】")
    print("=" * 70)

    test_q = "25万纯电SUV推荐"
    checks = []

    # 练习 1: 关键词选择器
    try:
        kw = kw_sel.select(test_q, k=3)
        ok1 = len(kw) == 3 and all("question" in ex for ex in kw)
    except Exception:
        ok1 = False
    checks.append(("练习1: 关键词选择器正确选出 3 个示例", ok1))

    # 练习 2: 语义选择器
    try:
        sem = sem_sel.select(test_q, k=3)
        ok2 = len(sem) == 3 and all("question" in ex for ex in sem)
    except Exception:
        ok2 = False
    checks.append(("练习2: 语义选择器正确选出 3 个示例", ok2))

    # 练习 3: MMR 选择器
    try:
        mmr = mmr_sel.select(test_q, k=3)
        ok3 = len(mmr) == 3 and all("question" in ex for ex in mmr)
    except Exception:
        ok3 = False
    checks.append(("练习3: MMR 选择器正确选出 3 个示例", ok3))

    # 练习 4: 多样性对比 — MMR 多样性应不低于语义选择器
    try:
        kw_div = compute_avg_pairwise_sim(kw)
        sem_div = compute_avg_pairwise_sim(sem)
        mmr_div = compute_avg_pairwise_sim(mmr)
        ok4 = mmr_div <= sem_div + 0.05  # 允许小幅波动
    except Exception:
        ok4 = False
    checks.append(("练习4: MMR 多样性不低于语义选择器", ok4))

    all_pass = True
    for desc, ok in checks:
        status = "[PASS]" if ok else "[FAIL]"
        if not ok:
            all_pass = False
        print(f"  {status}  {desc}")

    print(f"\n结论: {'全部达标 [PASS]' if all_pass else '有未完成项 [WARN] — 请实现标记 TODO 的方法'}")
