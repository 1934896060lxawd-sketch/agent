# 第五章：完整 RAG Agent —— 集成前四章全部优点

## 从第四章遗留下来的问题

第四章用 FAISS + BM25 + RRF 实现了混合检索，64 条文档的召回已经足够好。但把它当成一个生产系统，还有四个缺口：

```
缺口①：多轮对话不会指代消解

  用户第 1 轮："推荐一款大空间SUV"
  AI 回答：    "理想L6，24.98万起，轴距2920mm..."
  用户第 2 轮："那它的续航呢"

  → 向量模型把"那它的续航呢"编码成一个奇怪的向量
  → FAISS 检索出来的结果和"理想L6"毫无关系
  → 整个管线从第一步就偏了


缺口②：结构化约束靠检索靠不住

  用户问："25万以内的纯电SUV"
  混合检索可能返回：Model Y（26万起）← 语义最像，但超预算了
  检索能理解"像什么"，但理解不了"必须是什么"

  → 需要硬过滤，不是降权——超预算的车不该出现


缺口③：粗排结果有噪音，喂给 LLM 的上下文质量不够

  混合检索 Top-5（RRF 融合后）：
    1. 比亚迪海豚  ← 正确
    2. 用户评价A    ← 有关，但二手信息
    3. 比亚迪海豹   ← 误召回！和海豚是不同车
    4. 行业报告     ← 太泛
    5. 用户评价B    ← 弱相关

  → LLM 看到海豚和海豹混在一起，可能混淆
  → 需要在进入 LLM 之前再做一次精排


缺口④：每章各自一个独立脚本，没有统一的对外接口

  第一章 naiverag.py  → search() + build_prompt() + ask_llm()
  第三章 embedding_test.py → semantic_search()
  第四章 retrieval_test.py  → VectorIndex.search() + BM25.search() + hybrid_rrf()

  → 每次查询要手动串联，没有对话状态管理
  → 需要 RAGAgent 把全部能力打包成一个 chat() 调用
```

第五章把前四章**所有优点**集成到一起，新加 **Query 改写 + 元数据过滤 + Reranker 精排 + 多轮对话 + 内置评估**，形成一条完整的生产可用的 RAG 管线。

---

## 管线全貌（一次请求走完 6 步）

```
用户输入: "那它的续航呢"
   │                        ← 对话历史: 第 1 轮问了"大空间SUV" → AI 回了"理想L6"
   │
   ├── ① Query 改写 ──────────────────────────────────────────
   │   rewrite_query("那它的续航呢", chat_history)
   │   → "理想L6的续航里程"
   │   LLM 消解指代，把"它"替换成"理想L6"
   │
   ├── ② 混合检索 ────────────────────────────────────────────
   │   FAISS.search("理想L6的续航里程", top_k=9)     → 9 条向量候選
   │   BM25.search("理想L6的续航里程", top_k=9)      → 9 条关键词候選
   │   hybrid_rrf(向量 Top-9, BM25 Top-9, k=60)      → 6 条粗排结果
   │
   ├── ③ 元数据过滤 ──────────────────────────────────────────
   │   extract_filters("那它的续航呢")               → {} (无过滤条件，跳过)
   │
   ├── ④ Reranker 精排 ──────────────────────────────────────
   │   CrossEncoder 对 6 条候选逐个和 query 算真正的相关性
   │   → 精排 Top-3（"海豚"和"海豹"被区分开，后者被踢到后面）
   │
   ├── ⑤ 上下文拼接 ──────────────────────────────────────────
   │   _build_prompt_with_history(rewritten, context)
   │   → system + 对话历史 + 参考资料 + 用户问题
   │
   └── ⑥ LLM 生成 ────────────────────────────────────────────
       ask_llm(messages) → "理想L6 的 CLTC 综合续航为 1390km..."
       回答带 [来源: xxx] 标注
```

---

## 各章贡献对照

| 组件 | 来源 | 第五章的位置 |
|------|------|-------------|
| `load_data()` | 第一章 | `RAGAgent.__init__` ① |
| `car_to_text()` / `review_to_text()` / `chunk_sections()` | 第一章/第二章 | `load_data()` 内部调用 |
| `embed_documents()` | 第三章 | `RAGAgent.__init__` ② |
| `_embed_model` (BGE) | 第三章 | `VectorIndex.search()` 内部、`Reranker` 推理 |
| `VectorIndex` (FAISS) | 第四章 | `RAGAgent.__init__` ③ → `self.vector_index` |
| `BM25` | 第四章 | `RAGAgent.__init__` ④ → `self.bm25` |
| `hybrid_rrf()` | 第四章 | `RAGAgent.retrieve()` ② |
| `build_prompt()` / `ask_llm()` | 第一章 | `_build_prompt_with_history()` 升级版 / `chat()` ③ |
| **Reranker** (CrossEncoder) | **第五章新增** | `RAGAgent.__init__` ⑤ → `retrieve()` ④ |
| **Query 改写** | **第五章新增** | `retrieve()` ① |
| **元数据过滤** | **第五章新增** | `retrieve()` ③ |
| **多轮对话** | **第五章新增** | `chat_history` + `reset()` + `get_history_summary()` |
| **内置评估** | **第五章新增** | `self_eval()` + `TEST_CASES` |

---

## 逐组件详解

### ① Query 改写——消解指代，让检索器看到完整的"真实问题"

**为什么必须放在检索之前？**

```
直接检索"那它的续航呢"：
  embed("那它的续航呢") → 向量里"它""呢"占主导
  FAISS → 搜到各种和"它""呢"沾边的文档
  → 第一步就错，后面 Reranker 救不回来

先改写成"理想L6的续航里程"：
  embed("理想L6的续航里程") → 向量里"理想L6""续航"占主导
  FAISS → 精准命中目标文档
```

**核心逻辑（三种情况短路的思路）：**

```python
def rewrite_query(query, chat_history):
    # 情况①：第一轮对话，无历史 → 短路返回，省一次 LLM 调用
    if not chat_history:
        return query

    # 情况②：有指代词，需要改写
    messages = [
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},     # 改写规则
        *chat_history[-6:],                                      # 最近 3 轮
        {"role": "user", "content": f"改写以下问题：\n{query}"}  # 当前问题
    ]
    rewritten = ask_llm(messages)

    # 情况③：容错——LLM 返回空或太长 → 退回原始 query
    if not rewritten or len(rewritten) > 100:
        return query
    return rewritten.strip()
```

**三种改写场景的真实例子：**

| 对话历史 | 当前问题 | 改写后 | 类型 |
|---------|---------|--------|------|
| (空，第一轮) | "25万以内的SUV" | "25万以内的SUV" | 短路返回，不调 LLM |
| "推荐一款大空间家用SUV" → "理想L6..." | "那它的续航呢" | "理想L6的续航里程" | 指代消解 |
| "比亚迪海豚怎么样" → "小型纯电轿车，9.98万起..." | "同价位还有什么选择" | "9.98万起同价位小型纯电轿车" | 上下文补全 |

---

### ② 混合检索——继承第四章，加了候选池放大

和第四章完全一样的逻辑：FAISS 向量检索 + BM25 关键词检索 → RRF 融合。

唯一的差异：候选池比第四章更大：

```
第四章（独立对比实验）：
  vec_candidates = v_idx.search(query, top_k=3)
  bm25_candidates = bm25_idx.search(query, top_k=3)
  hybrid_rrf(vec, bm25, top_k=3)

第五章（完整管线）：
  vec_candidates = vector_index.search(query, top_k*3)   # 3→9
  bm25_candidates = bm25.search(query, top_k*3)           # 3→9
  hybrid_rrf(vec, bm25, top_k=top_k*2)                    # 3→6

为什么取 top_k×3 再缩到 top_k×2？
  → 元数据过滤可能踢掉一半候选
  → Reranker 精排需要在比 top_k 更大的候选池里挑
  → 如果粗排只取 top_k 条，过滤后可能一条不剩
```

---

### ③ 元数据过滤——硬性约束，不是降权

**为什么检索做不到这件事？**

```
检索的本质是"找最像的"，不是"找满足条件的"：

  query = "25万以内的纯电SUV"
  FAISS 向量 → 找到"高性价比SUV"方向的文档，Model Y (26万) 也在其中
  BM25 关键词 → 匹配到"25万""纯电""SUV"这些词，但不知道是 AND 关系
  RRF 融合 → Model Y 可能排进前三（因为其他维度太像）

元数据过滤：
  硬规则：if price > 25: 踢掉
  → 不需要"理解"，直接判对错，精度 100%
```

**extract_filters —— 纯规则实现，不调 LLM：**

```python
def extract_filters(query):
    filters = {}

    # 正则提取价格："25万以内" → max_price=25
    #                 "15万以上" → min_price=15
    #                 "10-15万"  → min_price=10, max_price=15
    for pattern, key in PRICE_PATTERNS:
        m = re.search(pattern, query)
        if m:
            filters[key] = int(m.group("price"))
            break

    # jieba 分词 + 字典匹配
    words = set(jieba.cut(query))
    for w in words:
        if w in CATEGORY_DICT:   filters["category"] = CATEGORY_DICT[w]
        if w in ENERGY_DICT:     filters["powertrain"] = ENERGY_DICT[w]
        if w in BRAND_DICT:      filters["brand"] = BRAND_DICT[w]

    return filters  # 空 dict 表示无约束，后续直接跳过过滤
```

**apply_filters —— 硬性剔除：**

```python
def apply_filters(candidates, filters):
    # 价格检查：只对 car_spec 类型文档生效
    #   max_price: 起售价 > 上限 → 踢
    #   min_price: 最高价 < 下限 → 踢
    # 类别/动力/品牌检查：doc_text 中不包含 → 踢
    # AND 关系：所有通过才保留
    return [(score, doc) for score, doc in candidates if _passes(doc)]
```

**过滤失败时的降级策略：**

```python
# 如果元数据过滤后一条都不剩，退回不过滤的结果
# 宁可多返回几条不相关的，也不能返回空结果
if not hybrid_results:
    hybrid_results = hybrid_rrf(vec_candidates, bm25_candidates, k=60, top_k=top_k)
```

---

### ④ Reranker 精排——从"分开看"升级到"一起看"

**核心区别：双塔 vs 交叉编码器：**

```
向量检索（双塔 / Bi-Encoder）：
  query 塔 → q_vec ─┐
                     ├─ cos(q, d) = q·d  ← 编码时互不相见
  doc 塔   → d_vec ─┘
  → 精度有损，但 doc 向量可以预计算，检索极快

Reranker（交叉编码器 / Cross-Encoder）：
  [query, doc] → CrossEncoder → 一个标量分数
  → query 和 doc 在 Transformer 每一层都互相对比，精度最高
  → 但每对 (query, doc) 都要完整跑一遍模型，不能预计算，慢

管线的正确用法：
  粗排 64 条 → 取 Top-12（混合检索，快）
  精排 12 条 → 取 Top-3（Reranker，慢但准）
  12 次 CrossEncoder 推理 ~50-200ms，完全可接受
```

**内部做了什么事（不需要你实现，知道就好）：**

```
输入 pair:  ["理想L6的续航里程", "理想L6，中大型SUV，CLTC续航1390km..."]
               ↓ CrossEncoder tokenizer
"[CLS] 理想L6的续航里程 [SEP] 理想L6，中大型SUV，CLTC续航1390km... [SEP]"
               ↓ 12 层 Transformer（query 和 doc 的 token 每一层都互相关注）
               ↓ 取 [CLS] 位置的向量 → 线性层
一个标量分数: 4.8（越大越相关，无固定范围）
```

**降级策略（模型加载失败时）：**

```python
class Reranker:
    def __init__(self):
        if os.path.isdir(self._LOCAL_DIR):
            self.model = CrossEncoder(self._LOCAL_DIR)     # 本地有，直接加载
        else:
            try:
                self.model = CrossEncoder(model_name)      # 从镜像下载
            except Exception:
                self.model = None   # 网络不通 → 降级：精排跳过

    def rerank(self, query, candidates, top_k):
        if self.model is None:
            return candidates[:top_k]   # 模型不可用时，退回粗排结果
        # ... 正常精排逻辑
```

**Reranker 如何纠正混合检索的误召回——一个真实案例：**

```
混合检索 Top-6（RRF 分数）：
  1. 理想L6（车型规格）                    ← 正确，是它
  2. 用户评价：理想L6续航实测               ← 正确，一手信息
  3. 理想L7（车型规格）                    ← 误召回！和 L6 是不同的车
  4. 行业报告：增程技术路线                 ← 有关但不直接
  5. 比亚迪海豚（车型规格）                 ← 完全不相关
  6. 用户评价：充电桩安装                   ← 弱相关

Reranker 精排后 Top-3：
  1. 精排分 5.2 - 理想L6（车型规格）       ↑ 最直接相关
  2. 精排分 4.1 - 用户评价：理想L6续航实测  ↑ 用户实测
  3. 精排分 1.8 - 行业报告：增程技术路线    ↑ 调到第三
  ...
  理想L7 精排分 -1.5 → 被踢出 Top-3        ↓ CrossEncoder 真正理解了 L6≠L7
  比亚迪海豚 精排分 -3.2 → 彻底垫底         ↓ "海豚"和"理想L6"无关
```

---

### ⑤ 上下文拼接——升级 system prompt，加入对话历史和引用格式

和第一章 `build_prompt()` 的核心区别：

```
第一章 build_prompt()：
  system: "你是汽车导购助手，严格根据上下文回答，不要编造"
  user:   "【参考资料】\n{context}\n\n【用户问题】\n{query}"

第五章 _build_prompt_with_history()：
  system: "你是汽车导购助手" + "回答时引用来源：[来源: xxx]" + "结合对话历史理解连续问题"
  user:   "## 对话历史\n{历史}\n\n## 参考资料\n{numbered_context}\n\n## 用户问题\n{query}"
```

多了三个东西：
1. **对话历史注入 user prompt**：LLM 知道用户上一轮问了什么
2. **引用格式强制要求**：`[来源: 文档名]` 格式，方便追溯
3. **参考资料编号**：`[1] 来源：xxx`，LLM 可以引用 `[1]` 而不用复制大段文本

---

### ⑥ RAGAgent 统一入口——一次 `chat()` 调用走完全部 6 步

```python
class RAGAgent:
    def __init__(self, data_dir):
        self.documents     = load_data(data_dir)          # ① 加载（第一章）
        self.documents     = embed_documents(documents)   # ② 向量化（第三章）
        self.vector_index  = VectorIndex(documents)       # ③ FAISS（第四章）
        self.bm25          = BM25(documents)              # ④ BM25（第四章）
        self.reranker      = Reranker()                   # ⑤ Reranker（第五章）
        self.chat_history  = []                           # ⑥ 对话状态

    def chat(self, query) -> dict:
        retrieved, rewritten = self.retrieve(query)       # ①-④ 检索管线
        context = format_context(retrieved)               # ⑤ 上下文拼接
        messages = self._build_prompt_with_history(query, context)
        answer = ask_llm(messages)                        # ⑥ LLM 生成
        self.chat_history.append(...)                     # 更新对话状态
        return {"answer": answer, "sources": ..., "rewritten": rewritten, ...}
```

**返回 dict 而不是纯字符串的原因：**

| 字段 | 用途 |
|------|------|
| `answer` | LLM 生成的回答，前端展示 |
| `sources` | 参考来源列表，前端渲染引用链接 |
| `rewritten` | 改写后的查询，调试用——看 Query 改写是否合理 |
| `retrieval_time_ms` | 检索耗时，性能监控——发现某次查询特别慢可以定位 |

---

## 数据流总览（一张图看懂）

```
                        离线建库（启动时跑一次）
                        ════════════════════
  data/ 目录
    ├── cars_specs.json ──→ car_to_text() ──┐
    ├── industry_reports/*.txt → chunk_sections() ──┤
    └── user_reviews.json ──→ review_to_text() ──┤
                                                   ↓
                                            64 条文档
                                              {"content": "...", "source": "...", "type": "..."}
                                                   │
                                        ┌──────────┼──────────┐
                                        ↓                     ↓
                                 embed_documents()        jieba 分词
                                  BGE 向量化              统计 DF + avgdl
                                        │                     │
                                        ↓                     ↓
                                  FAISS 索引              BM25 索引
                                  (IndexFlatIP)         (k1=1.5, b=0.75)
                                        │                     │
                                        └──────────┬──────────┘
                                                   ↓
                                             Reranker 加载
                                        (BAAI/bge-reranker-base)
                                                   │
                                          [OK] Agent 就绪


                        在线查询（每次 chat() 调用的完整路径）
                        ════════════════════════════════
  query: "那它的续航呢"
     │
     │  chat_history: [上一轮问了"大空间SUV" → 回答了"理想L6"]
     │
     ├──→ rewrite_query()
     │       ↓  LLM 看到完整对话历史，知道"它"=理想L6
     │    "理想L6的续航里程"
     │
     ├──→ self.vector_index.search("理想L6的续航里程", top_k=9)
     │       ↓  FAISS C++ 矩阵乘法
     │    [(0.88, doc_12), (0.82, doc_7), ...]   ← 9 条
     │
     ├──→ self.bm25.search("理想L6的续航里程", top_k=9)
     │       ↓  BM25 IDF×饱和词频×长度归一
     │    [(12.3, doc_12), (9.7, doc_3), ...]    ← 9 条
     │
     ├──→ hybrid_rrf(向量_9条, BM25_9条, k=60, top_k=6)
     │       ↓  1/(60+rank_vec) + 1/(60+rank_bm25)
     │    [(0.032, doc_12), (0.028, doc_7), ...] ← 6 条
     │
     ├──→ extract_filters("那它的续航呢")
     │       ↓  无"25万/纯电/SUV"等关键词
     │    {}  → 跳过过滤
     │
     ├──→ self.reranker.rerank("理想L6的续航里程", 6条候选, top_k=3)
     │       ↓  12 次 CrossEncoder 推理
     │    [(5.2, doc_12), (4.1, doc_7), (1.8, doc_15)]  ← 3 条
     │
     ├──→ _build_prompt_with_history("理想L6的续航里程", context)
     │       ↓  system + 对话历史 + 参考资料 + 问题
     │    [{"role": "system", ...}, {"role": "user", ...}]
     │
     └──→ ask_llm(messages)
             ↓  OpenAI API
          "理想L6 的 CLTC 综合续航为 1390km（[来源: 理想 L6]）..."
```

---

## 内置评估：10 条测试用例 × 两指标快速自检

评估不需要额外的测试框架——在 `__main__` 里输入 `/eval` 即可。

### 10 条测试用例覆盖 4 种场景

| 类型 | 例子 | 测试目标 |
|------|------|---------|
| 精确匹配 | "小米SU7的续航是多少" → 必须命中"小米""SU7" | 关键词+向量能否精准定位 |
| 条件过滤 | "25万以内的大空间SUV" → 必须命中"理想L6" | 元数据过滤是否生效 |
| 语义泛化 | "适合家庭出行的新能源车" → 必须命中"SUV""理想" | 语义理解能力 |
| 边界 | "100万以上的豪华电动车" → must_contain 为空 | 数据里没有时不编造 |

### 两个评估指标

```
Hit@K (Hit Rate)：
  检查 Top-K 条来源里是否至少包含一个期望关键词
  10 条 case，8 条命中 → Hit Rate = 80%

MRR (Mean Reciprocal Rank)：
  第一个命中关键词的来源排第几？
  排第 1 → RR = 1.0
  排第 2 → RR = 0.5
  排第 3 → RR = 0.33
  没命中 → RR = 0
  所有 case 的 RR 取平均
```

---

## 踩坑记录

### 坑 1：HF_ENDPOINT 必须在 import sentence_transformers 之前设置

```
❌ 在 Reranker.__init__ 里设置 HF_ENDPOINT
   → CrossEncoder() 构造时 huggingface_hub 已经解析了 huggingface.co 域名
   → 镜像失效，国内网络连不上

✅ 文件头部 import os 后立刻设置
   os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
   → 所有后续 import 都走镜像
```

### 坑 2：rerank() 输入是 [(score, doc), ...]，不是 [doc, ...]

```
❌ pairs = [[query, doc["content"]] for doc in candidates]
   → candidates 实际是 [(0.032, {doc}), ...]，doc 是 tuple
   → doc["content"] → TypeError: tuple indices must be integers or slices, not str

✅ docs = [doc for _, doc in candidates]
   pairs = [[query, doc["content"]] for doc in docs]
   → 先解包处理，把 (score, doc) 的 score 丢掉
```

### 坑 3：RRF 分数在 Reranker 阶段没有意义

```
❌ zip(scored, candidates) → 返回 [(精排分, (rrf分, doc)), ...]
   → 双层嵌套，后续代码解包全错

✅ docs = [doc for _, doc in candidates]  # 丢掉 RRF 分
   ranked = zip(scored, docs)             # 精排分 + 纯文档
   → 单层 (float, dict)，和其他检索方法格式完全一致
```

### 坑 4：文档的 type 字段在顶层，不在 metadata 下

```
❌ doc.get("metadata", {}).get("type", "")
   → 我们的 load_data() 不生成 metadata 字段
   → 永远返回 ""，价格过滤对 car_spec 失效

✅ doc.get("type", "")
   → load_data() 生成的是 {"content": ..., "source": ..., "type": "car_spec"}
```

### 坑 5：元数据过滤后可能一条不剩

```
query = "10万以内的特斯拉"  ← 矛盾条件：没有 ≤10 万的特斯拉
extract_filters → {"max_price": 10, "brand": "特斯拉"}
apply_filters → []  ← 全部被踢

✅ 过滤后检查是否为空，空则退回混合检索结果
   if not hybrid_results:
       hybrid_results = hybrid_rrf(vec_candidates, bm25_candidates, k=60, top_k=top_k)
```

### 坑 6：Reranker 模型下载失败，整个管线不能崩

```
✅ 三重降级：
   ① 本地有 → 直接加载
   ② 本地没有 → 从镜像下载
   ③ 下载失败 → self.model = None，打印警告，rerank() 短路返回原候选
```

---

## 关键认知

1. **Query 改写是多轮对话的入口关卡。** 这一关失守，FAISS、BM25、Reranker 全部在做无用功——它们在搜"那它呢"的向量，而不是你真正要找的实体。

2. **粗排用速度换范围，精排用精度换质量。** 粗排 64 条取 12 条（~1ms），精排 12 条取 3 条（~100ms），总耗时 ~100ms，精度接近 CrossEncoder 全量算 64 条的 95%+，但快了 5 倍。

3. **元数据过滤是硬规则，不是软降权。** "25 万以内"搜到 32 万的车就是错误，不应该通过降权来"不太可能出现在前面"——应该直接不让它出现。检索解决"像不像"，过滤解决"是不是"。

4. **Reranker 和 Embedding 是同一个团队（BAAI）出品，配合使用效果最好。** BGE Embedding 做粗排，BGE Reranker 做精排——训练数据分布一致，不存在模型不匹配的问题。

5. **返回 dict 比返回 str 好得多。** `answer` 是给用户的，`sources` 是给追责回溯的，`rewritten` 是给调试 Query 改写的，`retrieval_time_ms` 是给性能监控的。生产系统需要可观测性，只返回文本是不够的。

6. **降级优于崩溃。** Reranker 下载失败 → 跳过精排。过滤后结果为空 → 退回不过滤结果。精排返回空 → 退回粗排结果。管线里每一个环节都应该有 fallback，宁可多返回几条不相关的，不能返回"系统错误"。

---

## 各章关系图

```
第一章 Naive RAG
  │  load_data() / search(keyword) / build_prompt() / ask_llm()
  │  问题：关键词匹配太粗糙，搜不到同义词
  │
  ├──→ 第二章 Chunking —— 优化文档切块策略
  │       chunk_fixed_size() / chunk_paragraph() / chunk_sections()
  │
  ├──→ 第三章 Embedding —— 语义搜索
  │       embed_documents(BGE) / semantic_search(cosine)
  │       问题：Python for 循环慢 + 数字约束不敏感
  │
  ├──→ 第四章 Retrieval —— FAISS + BM25 + RRF 混合检索
  │       VectorIndex(FAISS) / BM25 / hybrid_rrf()
  │       问题：多轮对话不支持 + 结构化约束缺失 + 粗排有噪音 + 各章独立
  │
  └──→ 第五章 Full RAG Agent ── 全部集成
          rewrite_query()       → 多轮指代消解
          extract/apply_filters() → 硬约束过滤
          Reranker(CrossEncoder)  → 精排去噪
          RAGAgent.chat()         → 统一接口
          self_eval()             → 内置评估
```

---

## 快速启动

```bash
cd rag-project/chapters

# 1. 确保前四章的依赖已安装
pip install jieba numpy faiss-cpu sentence-transformers openai python-dotenv

# 2. 确保 .env 文件有 LLM 配置
# LLM_API_KEY=sk-xxx
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_MODEL_ID=gpt-4o-mini

# 3. （可选）下载 Reranker 模型到本地
# 下载地址：https://hf-mirror.com/BAAI/bge-reranker-base
# 放到：rag-project/models/bge-reranker-base/BAAI/bge-reranker-base/

# 4. 启动
python full_rag_agent.py
```

## 交互命令

```
>> 25万以内的大空间SUV       # 正常查询
>> 那它的续航呢               # 多轮对话（自动结合上一轮上下文）
>> /reset                     # 重置对话
>> /hist                      # 查看对话历史摘要
>> /eval                      # 运行内置评估
>> q                          # 退出
```

## 依赖清单

```
jieba                       # 中文分词（第一章）
numpy                       # 向量运算（第三章）
faiss-cpu                   # 向量索引（第四章）
sentence-transformers       # BGE Embedding + CrossEncoder Reranker
openai                      # LLM API 调用
python-dotenv               # 环境变量加载
```

## 模型清单

| 模型 | 用途 | 大小 | 来源 |
|------|------|------|------|
| BAAI/bge-base-zh-v1.5 | 文档向量化 + Query 向量化 | ~400MB | 第三章加载 |
| BAAI/bge-reranker-base | 粗排后精排 | ~1GB | 第五章新增 |
| GPT-4o-mini / DeepSeek 等 | Query 改写 + 最终回答生成 | API 调用 | 第一章配置 |
