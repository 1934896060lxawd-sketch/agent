# Day 6 面试题：四种 CoT 策略对比

> 对应文件：`prompt/cot_comparison.py`
> 核心能力：Zero-shot CoT、Few-shot CoT、Self-Consistency、Auto-CoT、四种策略量化对比

---

## Q1：Zero-shot CoT 为什么加一句"请一步步思考"就能提升推理质量？背后的原理是什么？

**一句话**：这句话触发了 LLM 训练数据中的"推理模式"概率分布，让 LLM 从"直接给答案"切换到"先分析再解答"。

**三层原理**：

**层 1 — 训练数据层面**：LLM 的训练数据中包含大量逐步推理的文本——教科书、解题过程、技术文档、论坛长帖。这些文本的共同特征就是"先分析再下结论"。"请一步步思考"这句话在语义上高度匹配这些推理文本的开头模式，LLM 在预测下一个 token 时会自然地从训练数据中召回"推理模式"相关的概率分布。

**层 2 — Transformer 架构层面（Self-Attention 机制）**：当 LLM 先写出了"分析：1. 先查价格..."，后续 token 的 Self-Attention 就会关注到这个已经生成的推理过程。这意味着后续的每一步推导都被已写出的推理步骤**约束**——LLM 不能跳步，因为它的 Self-Attention 能看到前面写的"第一步我还不知道 X"，如果下一步直接给答案就产生了矛盾。

**打个比方**：没有 CoT 时，LLM 像是在脑子里默算——你不知道它算对了还是猜对了。有 CoT 后，LLM 像在草稿纸上一步步演算——每一步都可见、可检查，后续计算被前面的步骤约束。

**代码体现**：

```python
# cot_comparison.py — zero_shot_cot()
prompt = f"""问题：{question}

请一步步思考，先写出推理过程，然后给出最终答案。"""

output = call_llm(prompt, temperature=0.0)

# 验证：检查输出是否包含推理步骤关键词
reasoning_keywords = ["步骤", "第一步", "然后", "因为", "所以", "推理", "计算"]
has_reasoning = any(keyword in output for keyword in reasoning_keywords)
```

**面试加分点**：能提到"Self-Attention 约束后续 token 生成"这个底层原理，说明你不仅会用，还理解 Transformer 的机制。

---

## Q2：Few-shot CoT 的示例应该怎么选？有什么核心原则？

**一句话**：三条原则——相关性、多样性、正确性。相关性保证示例有用，多样性防止 LLM 被单一思路限制，正确性防止 LLM 模仿错误。

**三条原则详解**：

**原则 1：相关性（Relevance）**——示例和当前问题的语义距离要近。

- 用 Embedding 算相似度，选 Top-K 最相关的示例
- 如果你要回答 SUV 的问题，示例全是轿车，LLM 的推理框架会被带偏
- 代码体现：Day 7 会用 `SentenceTransformer.encode()` 算余弦相似度

**原则 2：多样性（Diversity）**——选出来的 K 个示例之间不能太相似。

- 如果 3 个示例的推理思路完全一样（都是"先查价→再对比→再推荐"），第 4 个问题换了推理模式 LLM 就不会了
- 用 MMR（最大边际相关度）算法：`MMR = argmax[ λ·Sim(q, d_i) - (1-λ)·max(Sim(d_i, d_j)) ]`
- 第一项是相关性（和问题像不像），第二项是惩罚（和已选示例像不像），λ 控制权重

**原则 3：正确性（Correctness）**——示例本身的推理必须是正确的。

- 如果示例有计算错误，LLM 会忠实地模仿同样的错误（Garbage In, Garbage Out）
- 生产环境中，Auto-CoT 自动生成的示例需要人工审查或自动校验

**代码体现**：

```python
# cot_comparison.py — 硬编码的3个高质量示例
FEW_SHOT_EXAMPLES = [
    {
        "question": "20万预算买新能源车，比亚迪海豹和小鹏G6怎么选？",
        "reasoning": "第一步：查两款车的价格区间...\n第二步：对比核心参数...\n第三步：综合判断...",
        "answer": "如果看重智驾和续航，选小鹏 G6；如果看重加速性能和性价比，选比亚迪海豹。"
    },
    # ... 共 3 个示例，覆盖对比分析、数学计算、决策建议三种推理模式
]
```

**为什么示例要覆盖不同推理模式**：示例 1 是对比型推理（查数据→逐维对比→综合），示例 2 是计算型推理（确定参数→公式代入→计算结果），示例 3 是决策型推理（分析需求→列举选项→判断取舍）。三种模式覆盖了导购场景的主要推理路径，LLM 遇到新问题时有章可循。

---

## Q3：Self-Consistency 为什么会有效？什么情况下它会失效？

**一句话**：LLM 的不同推理路径的错误通常不相关（错得千奇百怪），但正确路径往往殊途同归（指向同一答案）。多次采样 + 投票就利用了这个不对称性。

**有效原因（数学直觉）**：

假设 LLM 单次推理的正确概率是 70%，错误概率是 30%。但注意：错误的 30% 不是集中在一个错误答案上，而是分散在多种错误推理路径上。正确的 70% 集中在同一个正确答案附近。

- 5 次独立采样：正确答案被选中的概率随采样次数快速上升
- 错误答案因为分散在不同选项上，单个错误答案得票超过半数的概率极低
- 这就是为什么投票有效——**正确答案的票数集中度高，错误答案的票数分散**

**代码实现**：

```python
# cot_comparison.py — self_consistency()
def self_consistency(question, n=5):
    candidates = []
    for i in range(n):
        output = call_llm(prompt, temperature=0.7)  # 必须 > 0！
        answer = normalize_answer(extract_answer(output))
        candidates.append(answer)

    vote_counter = Counter(candidates)
    most_common_answer, vote_count = vote_counter.most_common(1)[0]
    is_valid_vote = vote_count >= (n // 2 + 1)  # 验证：是否形成多数
    return {"final_answer": most_common_answer, "is_valid": is_valid_vote}
```

**关键细节**：`temperature=0.7` 是必须的。如果 temperature=0，LLM 每次输出完全相同，5 次采样一模一样，投票无意义。

**三种失效情况**：

1. **系统性偏见**：LLM 的训练数据在这个问题上普遍有偏，5 条推理路径可能全部犯同一个错误。投票不仅无效，还会放大偏见。
   - 例子：问"哪个品牌最好"而训练数据中某品牌出现次数远超其他，LLM 5 次都可能推荐同一个品牌

2. **正确答案需要非常规思路**：少数有洞察力的推理路径指向正确答案，但被多数平庸路径投票淘汰
   - 例子：需要跳出常规框架的 creative 问题，多数采样走常规路线，少数创新的被淹没

3. **temperature 过低**：temperature 太低导致每次输出几乎一样，投票退化成了"重复确认同一个答案"，失去了纠偏价值

**生产中的变通方案**：

- 只在最关键的一次决策上做 Self-Consistency（如最终推荐决策），其他步骤单次推理
- 开发/评测阶段用 Self-Consistency 建质量基线，上线后用更便宜的 Few-shot
- LLM 单次推理置信度低时（答案含"可能""也许"等词），自动触发 Self-Consistency 兜底

---

## Q4：Auto-CoT 中的 K-Means 聚类是做什么的？为什么不用随机选？

**一句话**：K-Means 保证选出的示例覆盖不同的"问题类型"，避免 Random 选出的示例集中在同一类型。

**为什么随机选不够好**：

假设候选问题池有 50 个问题：30 个对比分析型、15 个计算型、5 个决策型。随机选 3 个示例，大概率（约 63%）三个都落在对比分析型里——因为它是多数类。这导致 LLM 看到的示例推理模式单一，遇到计算型问题时无从模仿。

**K-Means 聚类的作用**：

1. 将所有问题向量化（Sentence-BERT → 384 维向量）
2. K-Means 把问题分成 K 个语义簇
3. 每个簇选离簇中心最近的问题作为"代表"——这个问题的语义最典型、最具代表性
4. 生成的示例自然覆盖了不同问题类型

**代码实现**：

```python
# cot_comparison.py — auto_cot() 中的聚类部分
encoder = SentenceTransformer('all-MiniLM-L6-v2')
question_vectors = encoder.encode(all_questions)

n_clusters = min(3, len(all_questions))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(question_vectors)

# 每类选离中心最近的问题作为"代表"
for cluster_id in range(n_clusters):
    indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
    cluster_center = kmeans.cluster_centers_[cluster_id]
    distances = [np.linalg.norm(question_vectors[i] - cluster_center) for i in indices]
    closest_idx = indices[int(np.argmin(distances))]
    representative_questions.append(all_questions[closest_idx])
```

**和 Day 3 Function Calling 的联系**：Day 3 你观察了 `tool_choice: "auto"` vs `"required"` vs `"none"` 的行为差异——不同 query 触发不同工具调用的概率不同。这和 Auto-CoT 面临的问题本质一样：要保证覆盖不同的"行为模式"（推理路径 / 工具调用路径），不能靠随机。聚类是通用的多样性保证手段。

**Auto-CoT 的风险**：生成的示例本身可能有推理错误——LLM 零样本生成的推理链不一定正确。实际生产中通常加一层人工审查或自动化校验（如用另一个 LLM 检查推理是否自洽）。

---

## Q5：CoT 和 ReAct 是什么关系？Agent 场景中两者怎么配合？

**一句话**：CoT 是 LLM **内部**的推理模式（"脑子里怎么想"），ReAct 是 Agent **外部**的行为模式（"边想边做"）。Agent 场景中，CoT 的体现就是 ReAct 循环中的 Thought 步骤。

**对比表**：

| 维度 | CoT | ReAct |
|------|-----|-------|
| 本质 | 推理策略（怎么让 LLM 想得更准） | 行为框架（怎么让 Agent 做得更好） |
| 信息来源 | LLM 内部知识 | 外部工具返回的真实数据 |
| 输出形式 | 纯文本推理链 | Thought → Action → Observation 循环 |
| 纠错方式 | 无法纠错（一次输出） | Observation 反馈后可调整 |
| 在代码中的位置 | Prompt 层面（加一句"一步步思考"） | 架构层面（StateGraph 节点 + 条件边） |

**Agent 场景中两者如何配合**：

以 Day 5 的汽车导购 Agent 为例，用户问"20-25 万推荐什么 SUV，帮我算用车成本"：

```
💭 Thought #1 (CoT 在发挥作用):
   "用户给了预算 20-25 万、车型 SUV，但我还不知道具体有哪些选择。
   应该先用 recommend_cars 拿到候选列表。"
  ⚡ Action: recommend_cars(budget_min=20, budget_max=25, category="SUV")
  📊 Observation: {小鹏 G6, 特斯拉 Model Y, 零跑 C11}

💭 Thought #2 (CoT 继续):
   "推荐了 3 款，其中小鹏 G6 20.99 万起最便宜。用户还要算用车成本，
   我对小鹏 G6 调用 calculate_ownership_cost。"
  ⚡ Action: calculate_ownership_cost(model="小鹏 G6")
  📊 Observation: {年均 1.68 万, 3年总计约 26 万}

💭 Thought #3 (CoT 综合):
   "小鹏 G6 3 年总成本约 26 万，年均 1.68 万。结合续航和智驾优势，向用户推荐。"
```

**关键洞察**：CoT 决定了 Thought 的**质量**（分析是否深入、逻辑是否完整），ReAct 决定了 Thought 的**时机**（什么时候该分析、什么时候该调工具）。优秀的 Agent 两个都做得好。

---

## Q6：你在生产环境中会怎么选择 CoT 策略？给出决策框架。

**一句话**：分三层决策——先看任务类型、再看资源约束、最后看质量要求。没有万能策略，只有合适策略。

**三层决策框架**：

```
第一层：任务类型
├── 纯推理题（数学计算、逻辑推理）
│   └── Self-Consistency > Few-shot > Auto-CoT > Zero-shot
├── 格式严格的任务（报告、审查意见、JSON 输出）
│   └── Few-shot > Auto-CoT > Zero-shot > Self-Consistency
├── 简单事实查询
│   └── Zero-shot 足够
└── 新领域快速启动（无标注数据）
    └── Auto-CoT > Zero-shot > Few-shot（无示例可写）

第二层：资源约束
├── Token 预算紧
│   └── Zero-shot（不加示例，不加多次采样）
├── 延迟敏感（用户等不起）
│   └── 不用 Self-Consistency（N 倍延迟不可接受）
├── 人工标注预算有
│   └── Few-shot 手动调优示例
└── 无人工标注
    └── Auto-CoT 自动生成示例

第三层：质量要求
├── 质量 > 成本
│   └── Self-Consistency + Few-shot 组合
├── 成本 > 质量
│   └── Zero-shot
└── 需要量化结果
    └── 四种策略上线前全跑一遍对比（Day 6 练习 5 的价值）
```

**量化对比是核心**（代码中的 `compare_all_strategies`）：

```python
# cot_comparison.py — 四个维度的量化对比
# 不是凭感觉说"Few-shot 更好"，而是用测试集跑出数据：

# 策略            推理完整性  格式一致性  投票有效性  多样性覆盖  平均延迟(s)  Token消耗
# Zero-shot CoT     87.5%       N/A         N/A         N/A         1.2s       3200
# Few-shot CoT      100.0%     100.0%        N/A         N/A         1.5s       5800
# Self-Consistency  100.0%      N/A         80.0%        N/A         6.8s      16000
# Auto-CoT          100.0%     100.0%        N/A        100.0%       4.2s       8900
```

**面试时能说的关键话**："生产环境中我们不会只用一种策略。比如在汽车导购 Agent 里，简单询价用 Zero-shot CoT（延迟最低），对比推荐用 Few-shot CoT（格式最稳定），关键决策用 Self-Consistency 兜底（准确率最高）。上线前用 `compare_all_strategies` 在 50 条评测集上跑出量化数据，作为策略选型的依据——不是凭感觉，是看数据。"

---

## Q7：为什么 temperature=0 时 Self-Consistency 无效？temperature 的本质是什么？

**一句话**：temperature 控制 LLM 输出概率分布的"平滑程度"——temperature=0 时分布退化为 one-hot（每次选概率最高的 token），输出完全确定，5 次采样一模一样。

**temperature 的数学含义**：

LLM 在生成每个 token 时，先算出词表上所有 token 的 logits（原始分数），然后除以 temperature，再 softmax：

```
P(token_i) = softmax(logits / temperature)
           = exp(logits_i / T) / Σ exp(logits_j / T)
```

- **T = 0**（实际上是极限 → 0）：`logits / T → ∞`，softmax 趋近 one-hot，每次必定选概率最高的 token → 输出完全确定，5 次采样无差异
- **T = 1**：原始概率分布，不改动
- **T > 1**（如 1.5）：`logits / T` 变小，分布更"平滑"，低概率 token 获得更多机会 → 输出更随机、更有创造性
- **T → ∞**：分布趋近均匀分布，输出完全随机（无意义）

**和 Self-Consistency 的关系**：

```python
# temperature=0.0 → 5 次输出完全一样 → 投票退化
output = call_llm(prompt, temperature=0.0)  # ❌ 无效

# temperature=0.7 → 5 次输出有差异 → 投票有效
output = call_llm(prompt, temperature=0.7)  # ✅ 有效
```

温度 0.7 是一个经验值：既让输出有足够多样性（不同推理路径），又不至于太随机（输出还有基本质量）。

**面试加分点**：能写出 softmax 公式并解释 temperature 的数学作用，说明你有算法底子。这是 Day 21 Transformer 手写时会深入的内容。

---

## Q8：四种 CoT 策略的 Token 消耗模型是怎样的？怎么估算成本？

**一句话**：Zero-shot 最省，Few-shot 和 Auto-CoT 因为示例占用 context window 额外消耗，Self-Consistency 是 N 倍基数的消耗。

**Token 消耗分解**：

| 策略 | System Prompt | 示例 | 当前问题 | LLM 输出 | 总消耗模型 |
|------|:---:|:---:|:---:|:---:|------|
| Zero-shot | ~50 | 0 | ~30 | ~200 | **≈ 280 tokens/次** |
| Few-shot | ~50 | ~300（3个示例） | ~30 | ~200 | **≈ 580 tokens/次** |
| Self-Consistency | ~50 | 0 | ~30 | ~200×5 | **≈ 1400 tokens/次（5次）** |
| Auto-CoT | ~50 | ~300（自动生成） | ~30 | ~200 | **≈ 580 + 聚类成本 tokens/次** |

**代码中的 Token 估算**：

```python
# cot_comparison.py — estimate_tokens()
def estimate_tokens(output: str) -> int:
    """粗略估算 token 消耗（中文 ~1.5 字/token，英文 ~4 字/token）"""
    chinese_chars = len(re.findall(r'[一-鿿]', output))
    other_chars = len(output) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)
```

**成本估算（DeepSeek 价格为例）**：

- Zero-shot：280 tokens × 8 题 = ~2240 tokens ≈ 0.002 元
- Self-Consistency：1400 tokens × 8 题 = ~11200 tokens ≈ 0.011 元
- 成本差异 5 倍，但准确率提升约 10-20 个百分点

**面试话术**："Token 消耗不是在选策略时才考虑的——你要先建质量基线（哪种策略效果最好），再算性价比（每提升 1% 准确率需要多少额外 token）。如果 Self-Consistency 比 Zero-shot 好 15%，但贵 5 倍，那看业务能不能接受。"
