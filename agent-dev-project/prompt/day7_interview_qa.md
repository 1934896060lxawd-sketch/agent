# Day 7 面试题：动态 Few-shot 选择器

> 对应文件：`prompt/few_shot_selector.py`
> 核心能力：KeywordSelector、SemanticSelector、MMRSelector、三种选择器量化对比

---

## Q1：为什么要动态选择 Few-shot 示例，而不是固定写死 3 个示例？

**一句话**：固定示例对某些 query 帮助很大，对另一些 query 却是噪音——动态选择让示例"随 query 变化"，保证每次注入的示例都和当前问题相关。

**固定示例的三个问题**：

1. **覆盖盲区**：Day 6 的 3 个固定示例覆盖了对比型、计算型、决策型推理。但如果用户问"CLTC 续航和实际续航差多少"（参数查询型），3 个示例没有一个是参数查询型的，LLM 无从模仿。

2. **噪声注入**：用户问"15 万买纯电轿车推荐"——固定示例中的"小米 SU7 首付 5 万月供计算"对这个 query 完全无关，但占据了宝贵的 context window 空间（~300 tokens）。

3. **上下文浪费**：固定示例越多覆盖越全，但 context window 有限。动态选择 K 个最相关示例，让每一 token 都花在刀刃上。

**量化理解**：

```
固定 3 个示例：3/16 覆盖率 = 18.75%，无论什么 query，固定示例集合不变
动态 K=3：每个 query 从 16 选 3，理论上有 C(16,3)=560 种组合
```

**面试话术**："动态选择把'示例挑选'从离线的一次性决策变成了在线的自适应决策。它的价值不是选得更准，而是让同一条 prompt 模板适配所有 query——query 变了，注入的示例也跟着变。"

---

## Q2：三种选择器（Keyword / Semantic / MMR）分别适用于什么场景？

**一句话**：Keyword 适合术语密集查询（快速、可解释），Semantic 适合自然语言查询（理解语义），MMR 适合需要覆盖多种推理模式的生产场景（平衡相关性和多样性）。

**对比表**：

| 维度 | KeywordSelector | SemanticSelector | MMRSelector |
|------|:---:|:---:|:---:|
| 核心原理 | 词袋交集大小 | 余弦相似度 | 贪心 MMR 最大化 |
| 依赖 | 无需模型 | Sentence-BERT Embedding | Embedding + 预计算相似度矩阵 |
| 延迟 | < 1ms（纯 CPU） | ~10ms（编码一次） | ~10ms + O(KN) 贪心 |
| 可解释性 | 高（能看到哪些词匹配） | 低（黑盒向量相似） | 中（知道相关度+多样性贡献） |
| 适合场景 | 术语密集的垂直领域 | 自然语言模糊查询 | 需要推理模式多样性的场景 |
| 局限 | 同义词完全失效 | 可能选出高度相似的示例 | 计算稍多，λ 需调参 |

**代码中的实际表现**（来自 `__main__` 的测试）：

```
查询: "买新能源车每年要花多少钱保养"
  关键词选择器 → 选出了"20万预算比亚迪海豹和小鹏G6怎么选"
                 （"新能源车"三字匹配，但问题是关于成本的，完全选错！）
  语义选择器   → 选出了"养一辆20万的新能源车一年花多少钱"
                 （正确！语义理解了这是成本问题）
```

**面试话术**："选型不是'MMR 最好所以全用 MMR'。如果查询本身就是术语驱动的（如'比亚迪海豚 405km 版'），Keyword 精确匹配反而比语义相似度更准。生产环境中我会先用查询类型分类器判断：术语查询走 Keyword，模糊查询走 Semantic，高价值决策走 MMR。"

---

## Q3：MMR 算法的核心公式是什么？λ 参数如何影响选择结果？

**一句话**：`score(i) = λ × Sim(query, d_i) - (1-λ) × max(Sim(d_i, d_selected))` —— 第一项奖励和 query 的相关性，第二项惩罚和已选示例的相似度。

**公式逐项解读**：

```
MMR = argmax [ λ · Sim(q, d_i) - (1-λ) · max_j Sim(d_i, d_j) ]
       ↑                      ↑                    ↑
       选第i个示例              相关性奖励            多样性惩罚
                              (越高越好)            (越低越好)
```

- **Sim(q, d_i)**：query 和候选示例 i 的余弦相似度。这个值越高，示例和问题越相关。
- **max_j Sim(d_i, d_j)**：候选示例 i 和已选示例集合中任意一个的最大相似度。这个值越高，说明 i 和已选示例中某个太像，应该惩罚。
- **λ**：相关性 vs 多样性的权重。λ 越大，越看重"选对 query 有用"；λ 越小，越看重"选的和已选的不一样"。

**三个极值**：

| λ | 行为 | 退化成的选择器 |
|:---:|------|:---:|
| 1.0 | 只看相关性，不看多样性 | SemanticSelector |
| 0.0 | 只看多样性，完全忽略 query | 纯粹的反相似度选择器 |
| 0.7 | 7 分相关，3 分多样（经验默认） | MMR（完整） |

**代码实现要点**：

```python
# few_shot_selector.py — MMRSelector.select()
for step in range(k):
    for i in candidates:
        relevance = lambda_ * sim_to_query[i]         # 相关性奖励
        diversity_penalty = 0
        if selected:
            max_sim = max(self._pairwise_sim[i][j]     # 与已选的最大相似度
                         for j in selected)
            diversity_penalty = (1 - lambda_) * max_sim  # 多样性惩罚
        mmr_score = relevance - diversity_penalty      # MMR 得分
```

**面试话术**："λ 是 MMR 唯一的超参数，生产环境中需要根据领域特点调优。技术文档密集的场景（如代码搜索）λ 取 0.8-0.9 偏向相关性，推荐场景（如电商导购）λ 取 0.5-0.7 偏向多样性——因为用户需要看到不同类型的推荐理由。"

---

## Q4：为什么 MMR 用贪心算法而不是全局搜索最优组合？贪心的理论保证是什么？

**一句话**：从 N 个候选中选 K 个最优子集是 NP-hard 问题（组合数 C(N,K) 随 N 爆炸），贪心选择有 (1-1/e) ≈ 63% 的理论保证，且实践中效果足够好。

**复杂度分析**：

```
全局最优：遍历所有 C(N,K) 种组合，每种组合计算 MMR 目标函数
  N=16, K=3: C(16,3) = 560 种 → 可接受
  N=100, K=5: C(100,5) ≈ 7.5×10^7 种 → 不可接受

贪心算法：逐步选择，每步遍历剩余候选取最优
  O(K × N) = O(3 × 16) = 48 次相似度计算 → 极快
```

**贪心的理论保证**：

MMR 的目标函数是**次模函数（Submodular Function）**——具有边际收益递减性质。对于最大化单调次模函数的基数约束问题，贪心算法有 **(1-1/e)** 的近似保证。

通俗理解：贪心选出的 K 个示例的目标函数值，至少是最优解的 63%。对于 K=3 的小规模场景，贪心通常能拿到 90%+ 的最优解。

**代码验证**（可以自己写一个小 test）：

```python
# 用 N=5, K=2 的小数据集，暴力搜索最优组合 vs 贪心结果
from itertools import combinations
best_global = max(combinations(range(N), K), key=mmr_score)
best_greedy = greedy_mmr(query, K)
# 比较两者的目标函数值
```

**面试话术**："选贪心不只是因为快——MMR 目标函数是次模的，贪心有理论保证。面试官如果追问次模性的定义，回答：'添加一个示例到小集合的收益 ≥ 添加同一个示例到大集合的收益，因为大集合中已有相似示例削减了边际价值。'"

---

## Q5：三种选择器的计算成本分别是多少？生产环境中如何权衡？

**一句话**：Keyword 几乎零成本（纯 CPU 字符串操作），Semantic 需要一次 Embedding 编码，MMR 在 Semantic 基础上多加 O(KN) 次相似度查找。瓶颈都在 Embedding 模型加载上。

**成本分解**：

| 环节 | KeywordSelector | SemanticSelector | MMRSelector |
|------|:---:|:---:|:---:|
| 初始化 | 0 | 编码 N 个示例：N × ~10ms | 编码 + 算 N×N 相似度矩阵 |
| 每次 select | O(N) 次分词+交集 | 1 次 query 编码 + O(N) 次点积 | 1 次编码 + O(KN) 贪心 |
| 内存 | ~10KB（存 pool） | ~N×768×4 bytes + 模型 ~400MB | 额外 N×N×4 bytes 相似度矩阵 |
| 首次加载 | 0 | ~2s（模型加载） | ~2s + 0.01s（矩阵计算） |

**三种生产选型策略**：

1. **小规模池（N < 100）**：MMR 是最好的选择——成本差异可忽略，多样性收益明显。
2. **中等规模（100 < N < 10000）**：用 Semantic + 后过滤。先选 Top-2K 语义最相关的，再在候选子集上跑 MMR。避免 O(N²) 相似度矩阵。
3. **大规模（N > 10000）**：用向量数据库（如 Milvus）做近似最近邻搜索（ANN）代替精确相似度计算。MMR 在 ANN 候选集上运行。

**面试话术**："生产环境中的核心优化思路不是换算法，而是先缩小候选集。先用语义相似度从 10000 个候选中粗筛出 Top-50，再在这 50 个上跑 MMR——延迟从 O(N²) 降到 O(50²)，几乎无质量损失。"

---

## Q6：计算示例两两平均相似度（`compute_avg_pairwise_sim`）的目的是什么？这个指标如何指导选型？

**一句话**：衡量选出的 K 个示例是否"千篇一律"——值越低说明示例间差异越大、多样性越好。

**计算公式**：

```python
# few_shot_selector.py — compute_avg_pairwise_sim()
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
normalized = vectors / (norms + 1e-8)
sim_matrix = np.dot(normalized, normalized.T)  # N×N 余弦相似度矩阵

# 取上三角（不含对角线 i==j）
upper_tri = sim_matrix[np.triu_indices(n, k=1)]
return float(np.mean(upper_tri))
```

**实际数据示例**（来自代码测试运行）：

```
查询: "比亚迪海豚和埃安Y哪个性价比高"
  关键词选择器: avg_pairwise_sim = 0.522
  语义选择器:   avg_pairwise_sim = 0.536
  MMR 选择器:   avg_pairwise_sim = 0.522
  → MMR 比语义好 1.0x
```

**面试话术**："平均相似度不是绝对值指标，而是相对指标——用它对比不同选择器在同一个 query 上的多样性表现。如果在 100 条测试集上 MMR 的平均值始终低于 Semantic，说明 MMR 的多样性优势是稳定且显著的。这就是消融实验的基础数据。"

---

## Q7：Day 7 动态选择器 + Day 6 CoT = `dynamic_few_shot_cot()` 的完整数据流是怎样的？

**一句话**：`query → 选择器.select(query, k=3) → 3 个最合适的示例 → 拼入 Few-shot Prompt → LLM CoT 推理 → 结构化答案`。

**完整数据流**：

```
用户 query: "25万预算看重智驾，推荐一款新能源SUV"
    │
    ▼
MMRSelector.select(query, k=3, lambda_=0.7)
    │ ① query 编码 → 768 维向量
    │ ② 与 16 个示例算余弦相似度
    │ ③ 贪心 MMR 逐步选出 3 个：argmax[λ·Sim - (1-λ)·max_sim_to_selected]
    │
    ▼
选中的 3 个示例:
  1. "25万预算，看重智能驾驶，推荐什么车？"      (推荐型)
  2. "家用第一辆车，20万以内SUV推荐"              (推荐型)
  3. "15万预算买纯电轿车，有什么推荐？"            (推荐型)
    │
    ▼
构造 Few-shot Prompt:
  "请参考以下示例的格式进行推理：
   示例1：问题：25万预算，看重智能驾驶...
          推理过程：第一步：筛选...
          答案：强烈推荐小鹏 G6...
   示例2：...
   现在请回答：
   问题：25万预算看重智驾，推荐一款新能源SUV
   推理过程："
    │
    ▼
call_llm(prompt, temperature=0.0)
    │
    ▼
{
  "output": "第一步：筛选...",
  "answer": "强烈推荐小鹏 G6",
  "selected_examples": ["25万预算看重智能驾驶推荐什么车", ...]
}
```

**面试话术**："这个串联的本质是：Day 7 保证'给 LLM 看什么示例'是最优的，Day 6 保证'LLM 看了示例后怎么推理'是结构化的。两者协同——示例选得再好，如果 prompt 模板没要求 CoT，LLM 依然可能跳步；CoT 格式再规范，如果示例不相关，LLM 也学不到正确的推理模式。"

---

## Q8：预计算（`__init__` 时编码所有示例 + 相似度矩阵）为什么是好的工程实践？

**一句话**：把初始化阶段的计算（一次性的）和每次查询的计算（重复的）分开，避免每个 query 都重新编码 16 个示例。

**对比**：

```python
# ❌ 每次 select 都重新编码：16 条 × 384 维 = 每 query 多 16 次 encoder.encode()
def bad_select(self, query, k=3):
    vectors = self.encoder.encode([ex["question"] for ex in self.pool])  # 每次都算！
    ...

# ✅ __init__ 时预计算：编码一次，永久复用
def __init__(self, pool):
    self.example_vectors = self.encoder.encode([ex["question"] for ex in self.pool])
    # 矩阵乘法一次性算出 N×N 相似度矩阵
    self._pairwise_sim = np.dot(normalized, normalized.T)

def good_select(self, query, k=3):
    # 只需编码 query 一次，其他直接从缓存取
    query_vec = self.encoder.encode([query])[0]
    ...
```

**面试话术**："预计算是 Embedding 场景的标准优化——编码一次存下来，查询时只算 query 和缓存向量的点积。这是向量数据库（如 Milvus、FAISS）的核心思路：建索引阶段预计算所有文档向量，查询阶段只做最近邻搜索。"

---

### Day 7 面试自检清单

| # | 问题 | 能答出吗？ |
|---|------|:---:|
| 1 | 为什么固定示例不够好？动态选择的三个价值是什么？ | □ |
| 2 | Keyword / Semantic / MMR 三种选择器的适用场景和局限？ | □ |
| 3 | MMR 公式能写出吗？λ 的三个极值分别对应什么行为？ | □ |
| 4 | 为什么 MMR 用贪心？(1-1/e) 保证是什么意思？ | □ |
| 5 | 三种选择器的计算成本对比？大规模场景怎么优化？ | □ |
| 6 | avg_pairwise_sim 这个指标怎么用？ | □ |
| 7 | dynamic_few_shot_cot 的完整数据流能画出来吗？ | □ |
| 8 | 预计算为什么是好的工程实践？和向量数据库的关系？ | □ |
