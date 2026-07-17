# Day 4 — RAG 数据摄入与嵌入基础

> **今日目标**：实现多格式文档加载器 + 分块策略 + 向量化管线。Day 4 是 RAG（检索增强生成）的第一步——没有高质量的分块和向量化，后续的检索和 Agent 推理都是空中楼阁。**特别增加 PDF 和 Word 文档解析，满足真实企业需求。**

---

## 目录

1. [今日任务清单](#1-今日任务清单)
2. [RAG 全链路概览](#2-rag-全链路概览)
3. [多格式文档解析](#3-多格式文档解析)
4. [分块策略设计](#4-分块策略设计)
5. [嵌入模型原理](#5-嵌入模型原理)
6. [数据加载管道](#6-数据加载管道)
7. [核心技术原理](#7-核心技术原理)
8. [初学者常见疑问](#8-初学者常见疑问)
9. [面试模拟问答](#9-面试模拟问答)

---

## 1. 今日任务清单

| 文件 | 行数 | 做什么 |
|------|:----:|------|
| `rag/chunker.py` | ~200 | 4 种格式分块器：JSON(车辆/评价/术语) + Markdown + PDF + Word，统一输出 Document 列表 |
| `rag/embeddings.py` | ~90 | BGE 嵌入模型单例加载 + 批量/单条向量化，本地优先 |
| `scripts/load_data.py` | ~140 | 遍历 raw/ 目录，自动检测文件格式并委托分块，统计汇总 |

---

## 2. RAG 全链路概览

Day 4 做的是 RAG 管道最上游的数据摄入层。先看清全链路，再理解每一步：

```
                          ┌──  Day 4 范围  ──┐
                          │                  │
knowledge_base/raw/       │   chunker.py     │   embeddings.py
  ├── vehicles.json ──────┤→ car_to_text()   │→ embed_documents()
  ├── reviews.json ───────┤→ review_to_text()│        │
  ├── glossary.json ──────┤→ glossary_to_text│        │
  ├── guides/*.md ────────┤→ chunk_markdown()│        │
  ├── faq/*.md ───────────┤→ chunk_markdown()│        │
  ├── industry/*.md ──────┤→ chunk_markdown()│        │
  ├── reports/*.pdf  ─────┤→ chunk_pdf()     │        │  (NEW - 企业需求)
  └── specs/*.docx  ──────┤→ chunk_docx()    │        │  (NEW - 企业需求)
                          │                  │        │
                          └──────────────────┘        ▼
                                              List[Document]
                                              .content = "车辆文本段落"
                                              .source  = "vehicles.json"
                                              .doc_type = "vehicle"
                                              .embedding = ndarray(768,)  ← 向量化后填充
                                              .metadata = {brand, model, ...}
                                                   │
                                                   ▼  Day 5 消费
                                              FAISS 索引 + BM25 + RRF 混合检索
                                                   │
                                                   ▼  Day 6 消费
                                              Agent Tool (search_knowledge)
```

**为什么 Day 4 只做摄入和嵌入？**

三步之间有明确的依赖：Day 4 的输出（带 embedding 的 Document 列表）是 Day 5（建 FAISS 索引）的输入。Day 5 的检索结果是 Day 6（Agent 调用工具）的数据源。这是标准的"先有数据、再能检索、最后能用"顺序。

---

## 3. 多格式文档解析

企业知识库不是整齐划一的——运维手册可能是 PDF、产品规格书是 JSON、方法论白皮书是 Word。Day 4 的 chunker.py 核心价值就是**统一不同格式，输出相同的数据结构**。

### 3.1 统一数据契约：Document dataclass

```python
@dataclass
class Document:
    content: str                    # 分块文本（核心字段，检索时匹配的就是这个）
    source: str                     # 原始文件名 → 回答时溯源标注
    doc_type: str                   # 8 种类型之一 → 支持按类型过滤检索
    chunk_id: str                   # uuid4().hex[:12] → 唯一标识
    metadata: dict                  # {brand, model, page, heading, ...} → 元数据过滤
    embedding: Optional[np.ndarray] = None  # embeddings.py 运行时填充
```

**面试常问**：为什么要有 `doc_type`？不能直接用 `source` 区分吗？

`source` 是文件名（`"vehicles.json"`），你无法用文件名做"只搜车辆数据"这种业务过滤。`doc_type = "vehicle"` 是语义标签，后续 Day 5 的元数据过滤（`filters = {"doc_type": "vehicle"}`）直接依赖它。

### 3.2 JSON 解析：结构化数据展平

知识库中有三种 JSON，结构完全不同：

```
vehicles.json  → [{brand, model, full_name, powertrain: {...}, performance: {...}, ...}]
reviews.json   → [{model, rating, pros: [...], cons: [...], owner: {...}}]
glossary.json  → {terms: [{term, full_name, category, explanation}]}
```

**问题**：同样都是 `.json` 后缀，怎么区分？不能靠文件名——文件名可以随时改。

**解法**：结构特征自动检测：

```python
data = json.load(file)

if isinstance(data, list) and data:
    sample = data[0]
    if "brand" in sample:
        # → 车辆数据：每个元素有 brand/model 键
        for item in data:
            text, meta = car_to_text(item)
    elif "pros" in sample:
        # → 评价数据：每个元素有 pros/cons 键
        for item in data:
            text, meta = review_to_text(item)
elif isinstance(data, dict) and "terms" in data:
    # → 术语表：顶层有 terms 键
    for item in data["terms"]:
        text, meta = glossary_to_text(item)
```

**这不是"魔术字符串"——是实际数据结构的指纹。** 一个包含 `"brand"` 键的 JSON 数组几乎一定是汽车数据，包含 `"pros"` 键的几乎一定是用户评价。比文件名校验可靠得多。

---

### 3.3 `car_to_text()` 深度解析

这是整个分块器中最复杂的函数。输入是 911 行 JSON 中的一个车辆对象，输出是一段自然语言文本。

**输入示例（简化）：**
```json
{
  "full_name": "比亚迪 秦PLUS DM-i 2026款",
  "brand": "比亚迪",
  "model": "秦PLUS DM-i",
  "category": "插电混动轿车",
  "price_range": "9.98-14.58万",
  "powertrain": {
    "type": "插电式混合动力",
    "battery_type": "磷酸铁锂刀片电池",
    "battery_capacity_kwh": 18.3,
    "pure_electric_range_km": 120,
    "combined_range_km": 1245,
    "fuel_consumption_l_per_100km": 3.8
  },
  "performance": {
    "system_power_kw": 226,
    "zero_to_hundred_seconds": 7.3,
    "drive_type": "前置前驱"
  }
}
```

**输出文本：**
```
【比亚迪 秦PLUS DM-i 2026款】。
品牌：比亚迪，车型：秦PLUS DM-i，类别：插电混动轿车。
价格区间：9.98-14.58万，目标用户：家庭用户、通勤族。
动力类型：插电式混合动力，电池类型：磷酸铁锂刀片电池，电池容量：18.3kWh，纯电续航：120km，综合续航：1245km，馈电油耗：3.8L/100km。
系统功率：226kW，0-100km/h：7.3s，驱动方式：前置前驱。
```

**为什么这样设计？（面试考点）**

| 设计点 | 原因 |
|--------|------|
| `【车名】` 用【】包裹 | 中文分词器（jieba）会把【】作为边界标记，提高车名命中率 |
| 数字+单位紧挨着写 | "120km" 是一个 token，"CLTC续航：120km" 比 "续航 120 公里" 更精确 |
| 键名作为中文标签 | 如果直接塞 JSON 原文 `{"pure_electric_range_km": 120}`，向量检索永远匹配不到 "纯电续航" |
| 层级结构展平为一段话 | 嵌入模型处理纯文本，不是 JSON 树。展平后 embedding 能捕捉跨字段的语义关系 |

**坑点：PHEV/增程车辆字段不统一**

```python
# 纯电车 v_001 (特斯拉 Model 3):
#   cltc_range_km: 606

# PHEV车 v_003 (秦PLUS DM-i):
#   pure_electric_range_km: 120    ← 没有 cltc_range_km!
#   combined_range_km: 1245

# 如果写成 car["powertrain"]["cltc_range_km"]，PHEV 直接 KeyError
```

**解法**：全部用 `.get()` 加默认值，逐字段独立判断：

```python
if pt.get("cltc_range_km"):
    parts.append(f"CLTC续航：{pt['cltc_range_km']}km")
if pt.get("pure_electric_range_km"):
    parts.append(f"纯电续航：{pt['pure_electric_range_km']}km")
```

每个字段独立检查——不存在的字段自动跳过，不存在的车系也不会缺信息。

---

### 3.4 Markdown 标题分块

```markdown
## 一、确定预算与用车场景
（一大段文字...）

## 二、动力形式选择
（一大段文字...）

## 三、配置清单核查
（一大段文字...）
```

**切分策略**：用正则 `r"(?=^## )"` 在 H2 标题前切分。`(?=...)` 是零宽先行断言——它匹配"即将出现 H2 的位置"，但不消耗字符。结果每个 chunk 都从 `## 标题` 开头。

```
chunk 0: "## 一、确定预算与用车场景\n（一大段文字...）"
chunk 1: "## 二、动力形式选择\n（一大段文字...）"
chunk 2: "## 三、配置清单核查\n（一大段文字...）"
```

**FAQ 特殊处理**：

FAQ 是用 `##` 分大类（预算与时机、油车 vs 新能源...），用 `###` 分具体问题。如果按 `##` 切分，14 个 Q&A 被归入 6 个大 chunk。当用户问"Model 3 保值率如何"，检索结果可能是整个"预算与时机"大段落，包含 4 个不相关问答。

```markdown
## 预算与购车时机
### Q1: 什么时候买车最便宜？
（回答...）
### Q2: 贷款还是全款？
（回答...）

## 油车 vs 新能源
### Q3: 开得少是不是买油车更划算？
（回答...）
```

**解法**：FAQ 文件用 `###` 切分，每个 Q&A 独立为 chunk：

```python
if "faq" in path_lower:
    heading_level = "###"   # 切分到 Q&A 粒度
else:
    heading_level = "##"    # 切分到 H2 节粒度
```

### 3.5 PDF 解析（企业需求 — pdfplumber）

```
真实场景：
  - 厂商技术白皮书是 PDF
  - 第三方评测报告是 PDF
  - 行业年度报告是 PDF
  - 有些 PDF 甚至是扫描件（图片型）
```

**为什么选 pdfplumber 而不是 PyPDF2？**

| 维度 | PyPDF2 | pdfplumber |
|------|--------|------------|
| 中文支持 | 偶有乱码 | 较好 |
| 表格提取 | ❌ 不支持 | ✅ `page.extract_table()` |
| 复杂布局（多栏） | 文字可能错位 | `page.extract_text()` 有布局感知 |
| 速度 | 快 | 中等 |

pdfplumber 在中文场景下比 PyPDF2 稳健得多，特别是包含表格的报告。

**分块流程：**

```
PDF 文件
  → pdfplumber.open() 逐页 extract_text()
  → 拼接所有页面，每页前加 [第N页] 标记（用于溯源）
  → 尝试按中文序号标题 "一、二、三、" 切分
  → 若切不出来 → 尝试阿拉伯数字 "1. 2. 3." 
  → 还切不出来 → 按双换行切，每块 max 1000 字符（带 100 字重叠防截断）
  → 每个 chunk 含 metadata.page 标记（回答时可以告诉用户"参考第3页"）
```

**关键细节：ImportError 保护**

pdfplumber 有 3MB+，不是所有环境都会安装。如果开发者在没有 PDF 文件的环境下运行，不应强制安装：

```python
def chunk_pdf(file_path):
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber 未安装，跳过 PDF 文件。安装: pip install pdfplumber")
        return []   # ← 静默跳过，不崩溃
```

### 3.6 Word 解析（企业需求 — python-docx）

```
真实场景：
  - 产品配置表（.docx 表格）
  - 内部技术文档（.docx 带标题层级）
  - 员工培训手册（.docx 含图片+表格+文字混合）
```

**分块策略：按 Word 标题样式切分**

Word 文档和 Markdown 的关键区别：Word 没有 `##` 标记，但 Word 有**样式系统**：

```python
doc = Document("spec.docx")
for para in doc.paragraphs:
    style = para.style.name   # → "Heading 1" / "Heading 2" / "Normal"
    text = para.text

    if "heading" in style.lower():
        # 这是章节标题 → 新 chunk 的起点
        start_new_chunk(text)
    else:
        # 这是正文 → 追加到当前 chunk
        append_to_current_chunk(text)
```

**表格提取**：Word 文档中表格是独立对象，需要专门处理：

```python
for table in doc.tables:
    for row in table.rows:
        cells = [cell.text for cell in row.cells]
        print(" | ".join(cells))   # → "参数 | 数值 | 单位"
```

表格文本追加到最近一个标题对应的 chunk 中，保持上下文完整性。

---

## 4. 分块策略设计

### 4.1 结构化分块 vs 语义分块 vs 固定窗口分块

这是面试中关于 RAG 的经典问题：

| 策略 | 方法 | 优点 | 缺点 | 本项目使用场景 |
|------|------|------|------|-------------|
| **结构化分块** | 按 JSON 实体 / Markdown 标题 / Word 样式边界切分 | 语义完整，不截断逻辑单元 | 依赖文档结构，纯文本文件无效 | 所有 JSON + Markdown + Word |
| **固定窗口分块** | 每 N 字符切一段，带 overlap | 通用，无格式假设 | 可能在句子中间截断 | PDF 退化为段落切分 |
| **语义分块** | 用 LLM 判断语义边界 | 最精确 | 成本高，速度慢 | (Phase 3 可选优化) |

**Day 4 的策略是"结构优先，窗口降级"**：先尝试按文档原生结构切分（JSON 实体、H2/H3 标题、Word 标题样式），结构检测失败时退化为固定窗口。

### 4.2 为什么"结构优先"对企业场景重要？

假设一篇购车指南被固定窗口（500字符）切分：

```
chunk_0: "...智能驾驶方面，推荐选择带有L2级别辅助驾驶的车型。比如比
亚迪秦PLUS搭载的DiPilot系统，支持自适应巡航和车道保持功能。特斯拉的
Autopilot则是行业标杆，但需要额外付费选装。小鹏的XNGP在城"
chunk_1: "市道路场景下表现优秀，特别是无图方案。对于经常跑高速的用户，
L2辅助驾驶的疲劳减轻效果确实明显。但考虑到预算，不一定需要上到..."

← "城市道路" 被切成两半，"小鹏的XNGP" 在 chunk_0、后面半句在 chunk_1
```

用户搜"小鹏城市智驾"时，两个 chunk 各包含一半信息，都匹配不强。

结构化分块后：整个"智能驾驶"一节是一个完整 chunk，所有相关信息原子化地集中在一起，检索命中率显著提高。

---

## 5. 嵌入模型原理

### 5.1 什么是 Embedding？

Embedding（嵌入/向量化）是把一段文本映射到高维空间中的一个点。语义相似的文本在空间中距离近：

```
高维空间 (768维) 的可视化降维示意：

           "纯电续航" ●
                      \
    "电池容量" ●───────● "续航里程"    ← 语义相近，距离近
                         \
                          \
                          "加速性能" ●──────● "0-100km/h"

    "座椅加热" ●                              ← 语义远，距离远
```

**关键**：距离不是靠关键词匹配（"加速" vs "0-100" 没有公共词），而是靠模型理解语义。

### 5.2 Bi-Encoder（双编码器）原理

BGE (BAAI General Embedding) 使用双编码器架构：

```
Query: "25万家用SUV推荐"
           │
           ▼
    [BERT-like Encoder]    ← 同一个模型，共享权重
           │
           ▼
      q_vec (768,)

Document: "【比亚迪 宋PLUS DM-i】价格区间：15.98-20.98万..."
           │
           ▼
    [BERT-like Encoder]    ← 同一个模型，共享权重
           │
           ▼
      d_vec (768,)

相似度 = cos(q_vec, d_vec) = (q_vec · d_vec) / (|q_vec| × |d_vec|)
```

如果所有向量都 normalize 到单位长度（`normalize_embeddings=True`），分母恒为 1，余弦相似度退化为点积：

```
相似度 = q_vec · d_vec   （单次点积运算，几十微秒）
```

这就是为什么 Day 5 用 `faiss.IndexFlatIP`（IP = Inner Product 内积）而不是 `IndexFlatL2`。

### 5.3 BGE 模型的中文适配

BAAI/bge-base-zh-v1.5 是专门为中文检索优化的：

| 特性 | 说明 |
|------|------|
| 维度 | 768（base 级别，平衡速度与精度） |
| 训练语料 | 中文维基 + 百度百科 + 新闻 + 社区问答，约 2.2 亿句子对 |
| 训练目标 | 对比学习（contrastive learning）：正例对靠近，负例对远离 |
| 特殊 token | `[CLS]` 向量作为整句表示（标准做法） |
| 最大长度 | 512 tokens |

### 5.4 本地加载策略

```python
# 优先级 1: 本地已下载的模型
local_path = project_root / "models" / "bge-base-zh-v1.5"
if (local_path / "pytorch_model.bin").exists():
    model = SentenceTransformer(str(local_path))

# 优先级 2: HuggingFace 镜像（国内网络友好）
else:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
```

**为什么本地优先？**

- Docker 构建时不需要网络访问 HuggingFace（内网部署场景）
- 首次启动不需要等待模型下载（~400MB）
- hf-mirror.com 镜像可能偶尔不稳定，本地模型作为兜底

**为什么用单例？**

```python
_embed_model: Optional[SentenceTransformer] = None  # 模块级私有变量

def get_embedding_model():
    global _embed_model
    if _embed_model is not None:      # 第二次调用直接返回
        return _embed_model
    # 第一次调用才加载模型
    _embed_model = SentenceTransformer(...)
    return _embed_model
```

一个 SentenceTransformer 实例占用 ~400MB 显存/内存。如果 `embed_documents()` 每次调用都创建新实例，处理 91 个文档会 OOM。单例保证全局只加载一次。

### 5.5 Bi-Encoder vs Cross-Encoder（面试高频）

| 维度 | Bi-Encoder | Cross-Encoder |
|------|-----------|---------------|
| 编码方式 | Query 和 Doc 独立编码，再算相似度 | Query 和 Doc 拼接后联合编码 |
| 速度 | **快**（Doc 向量可预计算、索引化） | 慢（每次检索都要重新编码所有 pairs） |
| 精度 | 中等 | **高**（Query-Doc 交互更充分） |
| 使用位置 | **粗排**（从 1000+ 文档中召回 Top-20） | **精排**（Day 5 Reranker，对 Top-20 重新排序） |
| 本项目对应 | `embed_documents()` / `embed_query()` | Day 5 `reranker.rerank()` |

**为什么不能只用 Cross-Encoder 替代 Bi-Encoder？**

假设 1000 个文档，每个 Query-Doc pair 经 Cross-Encoder 推理 100ms → 100 秒。Bi-Encoder 预计算所有文档向量后，单次检索 ~1ms。实际架构是**粗排（Bi-Encoder）× 精排（Cross-Encoder）**：Fast pass 过滤 95% 无关文档，Slow pass 只在候选池上精细排序。

---

## 6. 数据加载管道

### 6.1 load_data() 的设计哲学

```python
def load_data(data_dir=None) -> list[Document]:
    for file_path in sorted(raw_dir.rglob("*")):  # 递归遍历
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue                               # 跳过不支持的格式

        try:
            chunks = chunk_file(str(file_path))    # 委派给 chunker
            documents.extend(chunks)
        except Exception as e:
            logger.error(f"加载失败: {e}")          # 记录但继续
            continue                               # 不中断整体流程
```

**三个设计原则：**

1. **每个文件是独立工作单元**：一个损坏的 PDF 不应该阻止 10 个正常 JSON 文件加载。`try/except` 按文件粒度的隔离，而不是按批次。

2. **静默跳过不支持的格式**：`.jpg`、`.xlsx` 出现在 raw/ 目录时不会报错，只是跳过。避免"一个脏文件阻塞整个管道"。

3. **统计信息透明化**：加载完成后打印 `"{loaded}/{total} 文件, {total_chunks} 分块, {skipped} 跳过"`，运维人员一眼能看到有没有预期外的零分块。

### 6.2 路径推导

```python
# load_data.py 位于 knowledge_base/scripts/load_data.py
# 项目根 = knowledge_base/../ = car-ai-advisor/

_project_root = Path(__file__).resolve().parent.parent.parent
# __file__          → .../knowledge_base/scripts/load_data.py
# .parent           → .../knowledge_base/scripts/
# .parent.parent    → .../knowledge_base/
# .parent.parent.parent → .../car-ai-advisor/   ← 项目根
```

用 `Path(__file__).resolve()` 而不是 `os.getcwd()` 或相对路径 `"../.."`，因为：
- `os.getcwd()` 取决于 uvicorn 从哪里启动
- 相对路径依赖工作目录
- `__file__` 永远指向当前文件的实际位置 → 无论从哪里启动都能正确推导

---

## 7. 核心技术原理

### 7.1 为什么文档分块是 RAG 最关键的一步？

```
垃圾进 → 垃圾出 (Garbage In → Garbage Out)

如果分块把 "小鹏XNGP在城|市道路表现优秀" 切成两半：
  - 检索时两个半截 chunk 都匹配不到 "小鹏城市智驾"
  - LLM 收到的上下文不完整 → 生成"小鹏的城市智驾能力我没有找到具体信息"
  - 事后追查：向量检索其实命中了，但分块破坏了语义完整性
```

RAG 系统的质量上限由分块质量决定。检索算法（FAISS/BM25/RRF）只是在这个上限内逼近最优解。如果分块先天不足，检索无论如何优化都无济于事。

### 7.2 normalize_embeddings 的数学原理

```
未归一化时：
  余弦相似度 = (A·B) / (|A| × |B|)
  需要：点积（一次）+ 两次模长计算（两次开方，较慢）

归一化后（|A| = |B| = 1）：
  余弦相似度 = A·B
  只需要：点积（一次）

FAISS IndexFlatIP = 内积搜索，在高维空间做最近邻
在归一化向量上 = 余弦相似度搜索
```

这是为什么 `model.encode(texts, normalize_embeddings=True)` 和 `faiss.IndexFlatIP` 天然配对。

### 7.3 SentenceTransformer vs Transformers

```python
# 原生 Transformers 方式（低层 API）
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-zh-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-base-zh-v1.5")
inputs = tokenizer("你好", return_tensors="pt")
outputs = model(**inputs)
# 还需要手动做 mean pooling + normalize...约 15 行代码

# SentenceTransformer 方式（高层 API）
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
vec = model.encode("你好", normalize_embeddings=True)
# 1 行代码，自动处理 tokenization/pooling/normalization
```

`sentence-transformers` 库封装了 `transformers` 的低层细节，对文本嵌入场景做了专门优化（正确的 pooling 策略、自动 batch、进度条等）。除非有特殊的嵌入计算需求（如自定义 pooling），否则直接用 `sentence-transformers`。

### 7.4 Python 可选依赖模式

```python
def chunk_pdf(file_path):
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber 未安装")
        return []    # ← 关键：静默降级，不崩溃
```

这是一个**重要的企业级模式**：

- pdfplumber 有 3MB+，python-docx 有 5MB+
- 如果没有 PDF/Word 文件，装这些库纯浪费
- 函数内 import + ImportError 降级 → 核心功能（JSON/Markdown）不受影响
- 日志明确提示安装命令 → 运维友好

等价于 Node.js 的 `try { require('xxx') } catch` 或 Java 的 `Class.forName` 反射加载。

---

## 8. 初学者常见疑问

**Q: 为什么 16 辆车在 vehicles.json 中是一个大数组，不拆成 16 个 JSON 文件？**

单文件的好处：① 维护方便（增加一款车只改一行 JSON）；② 版本管理清晰（git diff 能看到新增了哪款车）；③ 加载效率（一次 `json.load()` 比 16 次文件 I/O 快得多）。知识库文件是"发布单元"，不是"存储单元"。

**Q: `car_to_text()` 中为什么要逐字段拼字符串，不直接把 JSON dump 成 text？**

```python
# 错误做法
text = json.dumps(car, ensure_ascii=False)
# → '{"full_name": "比亚迪 秦PLUS", "powertrain": {"cltc_range_km": 606}}'

# 正确做法
text = "【比亚迪 秦PLUS】CLTC续航：606km。"
```

前者的嵌入向量主要表示"这是一个 JSON 对象"，后者表示"这是一款续航 606km 的车"。检索目标是匹配用户意图（"续航长的车"），不是匹配 JSON 键名。

**Q: `embedding` 为什么存在 Document 字段而不是 metadata 字典里？**

`metadata` 是文档固有属性（品牌、车型、页码），存入后不应改变。`embedding` 是运行时计算结果——今天用 BGE-base 模型，明天可能换成 BGE-large，embedding 全部重建。语义上它们是不同层的东西。另外 `metadata` 可能被序列化（存磁盘），numpy 数组不适合频繁序列化。

**Q: 为什么 Markdown 切分用零宽断言 `(?=^## )` 而不是 `re.split(r'^## ', text)`？**

```python
# re.split(r'^## ', text)  → 分隔符被消费掉，不在结果中
# "## 一、标题\n正文" → ["一、标题\n正文"]  ← 丢失了 "##"

# re.split(r'(?=^## )', text)  → 分隔符保留在前面
# "## 一、标题\n正文" → ["## 一、标题\n正文"]  ← 完整保留
```

保留 `## ` 前缀对后续处理很重要——标题是 chunk 的天然摘要，检索命中时标题本身也是匹配信号。

**Q: FAQ 为什么用 `###` 而不是 `##` 切分？**

FAQ 的结构是 `## 大类 → ### Q: 具体问题 → A: 回答`。如果按 `##` 切，一个 chunk 包含 3-4 个 Q&A。用户搜 "Model 3 保值率"，命中的 chunk 是整段"保值率与二手车"大类，还包含"比亚迪保值率""蔚来保值率"等无关 Q&A。用 `###` 切分后每个 Q&A 独立为 chunk，检索精度从"大类级别"提升到"单个问答级别"。

**Q: `sorted(data_dir.rglob("*"))` 为什么排序？**

`rglob` 返回的文件顺序依赖于操作系统文件系统（Linux 通常是 inode 顺序，Windows 通常是字母序但不保证）。排序后保证每次运行的 Document 顺序一致 → 向量索引顺序一致 → 检索结果可复现。对开发调试至关重要。

---

## 9. 面试模拟问答

> **Q: 你们的 RAG 系统怎么处理多种格式的知识库文档？**

我们设计了一个统一的 Document 数据契约，所有格式的文档通过格式特定的分块器（JSON 展平、Markdown 标题切分、PDF 逐页提取、Word 标题样式切分）转换为相同的 Document 列表。关键设计是"结构优先，窗口降级"——优先利用文档原生结构边界（JSON 实体、H2 标题、Word 样式），结构缺失时退化为固定窗口切分。这样既保证了结构化文档的语义完整性，也兼容了格式混乱的文档。

> **Q: 为什么选 BGE 而不选 OpenAI text-embedding-3？**

两个原因：① 企业部署场景——Docker 内网部署时不需要调外部 API，没有网络延迟和安全顾虑；② 中文优化——BGE-base-zh 是专门为中文检索训练的，在 C-MTEB（中文嵌入评测基准）上的检索任务得分优于多语言通用模型。维度 768 在精度和索引性能之间达到工程上可接受的平衡。

> **Q: 如果 PDF 是扫描件（图片型），你们怎么处理？**

当前 pdfplumber 提取文本时，扫描件返回空字符串。代码会检测到空文本并记录警告日志 `"PDF 可能为扫描件/图片型"`。后续可以集成 OCR 管线（如 PaddleOCR）作为图片型 PDF 的前置处理步骤。我们的设计预留了这个扩展点——chunk_pdf 只返回 [] 而不会崩溃，后续可以在函数内部增加 OCR 检测和调用逻辑。

> **Q: 嵌入模型为什么用单例模式？每次请求创建新实例不行吗？**

一个 BGE-base 模型约 400MB 内存/显存。如果每次 `embed_documents()` 或 `embed_query()` 都 `SentenceTransformer(...)` 加载新实例，100 次请求 = 40GB 内存 → OOM 崩溃。单例保证全局只加载一次，所有请求共享。实现上用模块级私有变量 `_embed_model` + `get_embedding_model()` 懒加载——第一次调用时加载，后续直接返回缓存。

> **Q: 你们的 chunk 大小是怎么决定的？**

不走固定 token 数，走文档结构。JSON 实体（一辆车、一条评价、一个术语）天然是完整的语义单元，大小从 50 到 500 字不等。Markdown 按 H2 章节拆分，一个章节从 100 到 2000 字不等。PDF 和 Word 在有结构时同理。这种"语义驱动"的分块比固定窗口在检索场景下表现更好——用户提问通常对应一个完整的知识点，而不是 512 个 token 的文本切片。

> **Q: PyPDF2 和 pdfplumber 你是怎么选的？**

PyPDF2 的优势是轻量（无外部依赖），但中文文本提取偶有乱码，且完全不支持表格提取。pdfplumber 对中文 PDF 支持更好、表格提取是内置功能、复杂布局的文字排序更准。对于汽车行业知识库——产品规格表往往包含参数表格——pdfplumber 的表格提取能力是硬需求。代价是额外依赖（pdfplumber + pdfminer.six 约 5MB），我们通过函数内 import + ImportError 降级来消化这个代价。

> **Q: 你们的嵌入向量为什么要 normalize？**

数学上，归一化后向量长度为 1，余弦相似度退化为点积。工程上：① `faiss.IndexFlatIP` 做内积搜索比 `IndexFlatL2` 做欧氏距离搜索更快（少一次开方）；② 归一化消除了文本长度对相似度的偏向——长文本的向量模长天然更大，`normalize` 后长文档和短查询在同一个尺度下比较。FAISS 用内积（IP）替代 L2 距离是 RAG 系统的标准做法。

---

## 附：今日文件依赖关系

```
config.py (settings) ──→ embeddings.py (模型名称)
                        chunker.py    ← 独立模块，无内部依赖
                            ↑
                    load_data.py ──→ chunk_file() + Document
                            │
                            ▼
                      List[Document]
                            │
                            ▼  Day 5 消费
                      retriever.py (FAISS + BM25)
                            │
                            ▼  Day 6 消费
                      advisor.py (Agent 流式执行)
```

Day 4 的三个文件是 RAG 管道的**上游基础**。chunker 定义数据结构，embeddings 做向量化，load_data 做编排。三者的输出（带 embedding 的 Document 列表）是 Day 5 建 FAISS 索引的唯一输入。
