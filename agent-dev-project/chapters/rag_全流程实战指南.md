# RAG 全流程实战指南

## 总览：RAG 是什么

**RAG（Retrieval-Augmented Generation）** 让 LLM 回答"它没见过"的问题——不是靠训练记忆，而是先把相关资料找出来，把资料和问题一起喂给模型，让模型照着资料回答。

全流程分两大阶段：

```
┌──────────────────────────────────────────────────────────────────┐
│                      离线建库（Indexing Pipeline）                 │
│                                                                  │
│  PDF/Word/PPT/视频 ──→ 文档解析 ──→ Chunk分块 ──→ Embedding向量化 │
│                                                                  │
│       ┌──────────────────────────────────────────────┐           │
│       │  入库：FAISS / ChromaDB / Milvus / ES        │           │
│       │  每条记录 = {chunk文本, 向量, 来源, 元数据}    │           │
│       └──────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                      在线问答（Query Pipeline）                    │
│                                                                  │
│  用户提问 ──→ Query Embedding ──→ 向量库检索                      │
│                                       │                          │
│                                       ├──→ 向量检索（语义）        │
│                                       └──→ BM25检索（关键词）      │
│                                            │                      │
│                                            ▼                      │
│                              混合排序（加权融合 / RRF）            │
│                                            │                      │
│                                            ▼                      │
│                              拼接上下文 ──→ LLM 生成回答           │
└──────────────────────────────────────────────────────────────────┘
```

**核心原则：LLM 不是数据库，RAG 用"检索"补上 LLM 的知识盲区。**

---

# 第一部分：离线建库（Indexing Pipeline）

离线建库跑一次，把文档转成可被检索的（文本 + 向量）对。

---

## 一、文档解析：把任何格式变成纯文本

### 1.1 为什么文档解析是第一个坑

真实项目的知识库不是 5 个 txt 文件。用户会往里面扔 PDF、Word 合同、PPT 汇报、甚至会议录音。**如果不能正确解析，后面的分块、向量化、检索全是白做。**

### 1.2 PDF 解析

PDF 是最复杂的格式——它存的是"画布上的文字位置"，不是流式文本。

```python
import fitz  # PyMuPDF，速度最快，适合中文

def parse_pdf(file_path: str) -> str:
    """解析 PDF，返回纯文本"""
    doc = fitz.open(file_path)
    full_text = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            full_text.append(text)
    return "\n\n".join(full_text)
```

**三种 PDF 的类型与策略**：

| PDF 类型 | 特征 | 工具 | 陷阱 |
|----------|------|------|------|
| 原生 PDF（Word/网页生成） | 文字可选中、复制正常 | `PyMuPDF` (fitz) | 无 |
| 扫描版 PDF（纸质扫描） | 文字不可选中、图片感 | `pytesseract` + OCR | OCR 准确率受扫描质量影响，中文需要中文字体包 |
| 混合 PDF | 大部分文字 + 部分图片表格 | `PyMuPDF` + `pdfplumber` 互补 | 表格提取需要专门处理 |

**生产注意事项**：

```
1. 分页保留页码信息 → 方便定位原文
2. 表格不要用 get_text() 硬取 → 用 pdfplumber.extract_tables()
3. 双栏排版会打乱阅读顺序 → PyMuPDF 的 get_text("blocks") 按阅读顺序排序
4. 页眉页脚/水印 → 用正则过滤掉重复出现的行
5. 超大 PDF（>100MB）→ 流式读取，不要一次性加载到内存
```

### 1.3 Word 解析

```python
from docx import Document

def parse_docx(file_path: str) -> str:
    """解析 .docx，提取段落文本"""
    doc = Document(file_path)
    paragraphs = []
    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append(para.text)
    return "\n\n".join(paragraphs)
```

**生产注意事项**：

```
1. 表格内容 → 遍历 doc.tables，每个 cell 取 text
2. 页眉页脚 → doc.sections[0].header.paragraphs
3. 嵌入图片 → 单独导出，用 OCR 或视觉模型描述
4. 修订痕迹（Track Changes）→ 默认解析的是当前版本，旧内容会被忽略
5. .doc（旧格式）→ 先转 .docx 或用 python-pptx 的姊妹库无法处理，需 LibreOffice 命令行转换
```

### 1.4 PPT 解析

```python
from pptx import Presentation

def parse_pptx(file_path: str) -> str:
    """解析 PPT，提取所有幻灯片的文本"""
    prs = Presentation(file_path)
    slides_text = []
    for i, slide in enumerate(prs.slides):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    if paragraph.text.strip():
                        texts.append(paragraph.text)
        if texts:
            slides_text.append(f"【幻灯片 {i+1}】\n" + "\n".join(texts))
    return "\n\n".join(slides_text)
```

**生产注意事项**：

```
1. SmartArt / 图表 → has_text_frame=False，取不到文字 → 单独处理图表对象
2. 演讲者备注 → slide.notes_slide.notes_text_frame.text
3. PPT 的文字是碎片化的（每个文本框独立）→ 上下文丢失，需要按幻灯片聚合
4. 嵌入视频/音频 → 用 ffmpeg 提取后转文字（见 1.5 视频解析）
```

### 1.5 视频/音频解析（Whisper 转文字）

```python
# 安装：pip install openai-whisper
import whisper

def transcribe_video(video_path: str, model_size: str = "medium") -> str:
    """把视频/音频转成文字（需要先提取音频）"""
    model = whisper.load_model(model_size)
    result = model.transcribe(video_path, language="zh")
    return result["text"]
```

**生产注意事项**：

```
1. 长视频（>30分钟）→ Whisper 按 30 秒分段处理，不用担心 OOM
2. 多说话人 → Whisper 不会标注"A说""B说"，需要额外的说话人分离（diarization）
3. 背景噪音 → 先用 ffmpeg 降噪或提取人声轨道
4. 模型选择：
   - tiny（39M）：速度快但准确率低，适合英文
   - medium（1.5G）：中文推荐起步档
   - large-v3（3G）：最高准确率，但需要 GPU
5. 成本：本地 Whisper 免费；云端用 OpenAI Whisper API $0.006/分钟
6. 音频提取：ffmpeg -i video.mp4 -vn -ar 16000 audio.wav
```

### 1.6 文档解析的统一接口

生产环境中你不会预先知道用户上传的是什么格式。需要一个统一入口：

```python
import os

PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".pptx": parse_pptx,
    ".txt": lambda p: open(p, encoding="utf-8").read(),
    ".mp4": transcribe_video,
    ".wav": transcribe_video,
}

def parse_document(file_path: str) -> dict:
    """统一文档解析入口，返回 {text, metadata}"""
    ext = os.path.splitext(file_path)[1].lower()
    parser = PARSERS.get(ext)
    if parser is None:
        raise ValueError(f"不支持的文件格式: {ext}")

    text = parser(file_path)
    return {
        "text": text,
        "source": os.path.basename(file_path),
        "file_type": ext,
        "char_count": len(text),
    }
```

**一句话总结**：文档解析的目标是把任何格式变成干净的纯文本。格式千奇百怪，但下游（分块、向量化）只认文本。

---

## 二、文本分块（Chunking）

> 详见第二章完整文档。此处只讲生产级注意事项。

### 2.1 Chunk 大小的黄金法则

```
太小（<100字）：丢失上下文，LLM 看不懂孤立的碎片
太大（>1000字）：检索精度下降，噪音多，超过 Embedding 模型的 max_seq_length
合适（200-500字）：一个能独立回答一个问题的语义单元

BGE-base-zh-v1.5 的 max_seq_length = 512 tokens ≈ 约 800-1000 中文字
```

### 2.2 Metadata 不要丢

```python
# 错误做法：只存 content
{"content": "2026年新能源汽车销量突破...", "chunk_id": "c_0"}

# 正确做法：保留来源和层级信息
{
    "content": "2026年新能源汽车销量突破...",
    "chunk_id": "report_01_sec_3",
    "source": "2026新能源市场报告.pdf",
    "page": 3,
    "section": "三、市场规模与销量",
    "char_start": 1200,
    "char_end": 1550,
}
```

Metadata 的价值在于：（1）回答时可以引用出处 （2）用户可以点击跳转原文 （3）调试时可以快速定位问题 chunk。

### 2.3 Chunking 策略选择

| 数据类型 | 推荐策略 | 工具 |
|----------|---------|------|
| 结构化报告（有"一二三"标题） | 按章节标题切分 | 正则 `re.split()` |
| Markdown/技术文档 | 按 `##/###` 切分 | `langchain.text_splitter.MarkdownHeaderTextSplitter` |
| 对话记录/客服日志 | 按发言轮次切分 | 正则按 `用户：/客服：` 分割 |
| 代码文件 | 按函数/类切分 | AST 解析（`ast` 模块） |
| 无结构长文本 | 固定大小 + overlap | `RecursiveCharacterTextSplitter` |

### 2.4 Overlap（重叠）的正确用法

```
chunk_size=400, overlap=50

block 1: [0-400]
block 2: [350-750]   ← 50 字重叠
block 3: [700-1100]

作用：防止关键信息卡在两个块边界上，检索时至少有一个块能完整包含它
代价：存储膨胀（64 块变 70 块）

经验值：overlap = chunk_size 的 10%-15%
```

---

## 三、向量化（Embedding）

> 详见第三章完整文档。此处聚焦生产级选型和优化。

### 3.1 模型选型

| 模型 | 维度 | 中文效果 | 速度 | 适用场景 |
|------|------|---------|------|---------|
| `BAAI/bge-base-zh-v1.5` | 768 | ★★★★★ | 中 | 中文 RAG 首选 |
| `BAAI/bge-large-zh-v1.5` | 1024 | ★★★★★ | 慢 | 对精度要求极高时 |
| `BAAI/bge-small-zh-v1.5` | 512 | ★★★★ | 快 | 文档量 >10 万条 |
| `text2vec-large-chinese` | 1024 | ★★★★ | 中 | 替代方案 |
| `moka-ai/m3e-base` | 768 | ★★★★ | 中 | 社区热门 |

**选型原则**：
- 10 万条以内 → `bge-base`，性价比最高
- 100 万条以上 → `bge-small` 或上 GPU
- 中英混合 → `bge-m3`（多语言版本）

### 3.2 批量编码 vs 逐条编码

```python
# 错误：逐条编码（慢 10-50 倍）
for doc in documents:
    doc["embedding"] = model.encode(doc["content"])

# 正确：批量编码
texts = [doc["content"] for doc in documents]
embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True)
```

批量编码利用 GPU/CPU 的矩阵运算并行能力。`batch_size=32` 是一个安全默认值。

### 3.3 模型部署策略

| 方案 | 延迟 | 成本 | 适用 |
|------|------|------|------|
| 本地 CPU | ~50ms/条 | 0 | 文档 <1 万，开发测试 |
| 本地 GPU | ~5ms/条 | 显卡成本 | 文档 1-100 万 |
| 云端 API（如硅基流动） | ~20ms/条 | ¥0.001/条 | 不想管运维 |
| 模型服务化（Triton/vLLM） | ~2ms/条 | GPU 服务器 | 高并发在线服务 |

### 3.4 向量"保鲜"

知识库会更新，embedding 不会自动同步。需要：
```
1. 新增文档 → 增量 embed + 追加到索引
2. 修改文档 → 找到旧 chunk → 重新 embed → 更新索引
3. 删除文档 → 从索引中移除对应 chunk 的向量
4. 模型升级（如 bge-base → bge-large）→ 全量重建，不能混用
```

---

## 四、建立索引（向量入库）

### 4.1 为什么需要索引

第三章的 `semantic_search()` 每次查询遍历全部 64 条文档做点积——O(n)。

64 条：0.1ms，完全够用
64,000 条：60ms，还能忍
6,400,000 条：6 秒，用户已经关页面了

**索引的目标：O(log n) 甚至 O(1) 的检索速度，与文档量几乎无关。**

### 4.2 FAISS 索引（适合单机、中小规模）

```python
import faiss
import numpy as np

# ① 把所有文档向量堆成矩阵
vectors = np.array([doc["embedding"] for doc in documents], dtype=np.float32)
# shape: (64, 768)

# ② 建索引
dimension = 768
index = faiss.IndexFlatIP(dimension)    # IP = Inner Product = 余弦相似度
index.add(vectors)                       # 入库

# ③ 查询一次
q_vec = model.encode([query], normalize_embeddings=True)[0]
q_vec_faiss = q_vec.reshape(1, -1).astype(np.float32)
distances, ids = index.search(q_vec_faiss, k=3)
# distances[0] = [0.67, 0.65, 0.63]  ← 相似度，越大越相关
# ids[0] = [3, 17, 42]              ← 对应的文档序号
```

**FAISS 索引类型对比**：

| 索引类型 | 速度 | 精度 | 内存 | 适用 |
|---------|------|------|------|------|
| `IndexFlatIP` | 基准 | 100%（精确） | 高 | <10 万条，暴力搜索 |
| `IndexIVFFlat` | 快 10-100x | 95-99%（近似） | 中 | 10-1000 万条 |
| `IndexHNSWFlat` | 极快 | 98-99%（近似） | 高 | 图索引，内存换速度 |
| `IndexIVFPQ` | 极快 | 90-95%（有损压缩） | 低 | >1 亿条，内存受限时 |

```python
# IndexIVFFlat 示例：先聚类再搜索，大幅加速
import faiss
quantizer = faiss.IndexFlatIP(768)
index = faiss.IndexIVFFlat(quantizer, 768, nlist=10)  # nlist=聚类中心数，一般取 sqrt(N)
index.train(vectors)   # 必须先 train 再 add
index.add(vectors)
index.nprobe = 3       # 搜索时探测的聚类数，越大越准越慢
```

### 4.3 FAISS 索引的保存与加载

```python
# 保存
faiss.write_index(index, "knowledge_base.index")

# 加载
index = faiss.read_index("knowledge_base.index")
```

**注意**：FAISS 索引只存向量和 ID，不存文档内容。你需要另外维护一个 `id → document` 的映射表：

```python
# 入库时
id_to_doc = {}
for i, doc in enumerate(documents):
    id_to_doc[i] = doc

# 查询后
ids = index.search(q_vec, k=3)[1][0]
results = [id_to_doc[i] for i in ids]
```

### 4.4 向量数据库（适合大规模、多用户）

当文档量 >100 万或需要多用户隔离时，用专业向量数据库：

| 数据库 | 定位 | 特点 |
|--------|------|------|
| **ChromaDB** | 轻量级，适合原型 | 安装简单，Python 原生，内置 metadata 过滤 |
| **Milvus** | 企业级，分布式 | 支持十亿级向量，GPU 加速，云原生 |
| **Weaviate** | 全功能 | 自带向量化 + 搜索 + 问答，GraphQL 接口 |
| **Elasticsearch** | 搜索引擎 + 向量 | 如果你的技术栈已有 ES，直接用 kNN 插件 |

```python
# ChromaDB 示例：代码最少
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("knowledge_base")

# 入库
collection.add(
    documents=[doc["content"] for doc in documents],
    embeddings=[doc["embedding"].tolist() for doc in documents],
    metadatas=[{"source": doc["source"]} for doc in documents],
    ids=[f"doc_{i}" for i in range(len(documents))],
)

# 查询
results = collection.query(query_embeddings=[q_vec.tolist()], n_results=3)
```

**一句话总结**：文档 <10 万且离线用 → FAISS。文档多、要增删改查、多用户 → ChromaDB 或 Milvus。技术栈已有 ES → 直接加 kNN 向量插件。

---

# 第二部分：在线问答（Query Pipeline）

离线建库完成后，每次用户提问都走这个流程。

---

## 五、Query Embedding：将问题向量化

```python
def encode_query(query: str, model) -> np.ndarray:
    """将用户问题转成向量"""
    return model.encode([query], normalize_embeddings=True)[0]
```

**注意事项**：

```
1. 同一个模型！query 和 document 必须用同一个 embedding 模型编码
   → 用 bge-base 编码文档，用 bge-large 编码 query → 向量空间不兼容，检索全乱

2. BGE 模型的特殊处理：BGE 官方建议 query 前加 "为这个句子生成表示以用于检索相关文章：" 前缀
   但实际测试中不加前缀通常影响不大，取决于你的数据

3. 短 query（<5字）的向量质量差 → 考虑先做 Query Expansion（用 LLM 把短 query 扩写成完整句）
   "小米续航" → expanded → "小米SU7的续航里程和电池容量是多少"
```

---

## 六、混合检索：向量 + BM25

### 6.1 为什么单靠向量不够

第三章的实测结果已经说明了：

| 查询 | 向量检索 | BM25 检索 | 谁更好 |
|------|---------|-----------|:---:|
| "大空间家用SUV" | 理想 L8/L6/AITO M7 | 购车指南（匹配到"空间""家用"） | 向量 |
| "小米SU7续航多少" | 小米SU7、小鹏 P8、海豹08 | 小米SU7 | 两者都好 |
| **"25万以内的纯电轿车"** | 25-32万的海豹08（超预算！） | 13-18万的领克 Z20（在预算内） | **BM25** |

向量不理解数字大小，BM25 不理解同义词。**两者互补。**

### 6.2 BM25 实现

```python
import math
import jieba

class BM25:
    """
    BM25 是 TF-IDF 的升级版，考虑了词频饱和和文档长度归一化。
    比第一章的"命中次数"打分方式精确得多。
    """
    def __init__(self, documents: list[str], k1=1.5, b=0.75):
        self.k1 = k1      # 词频饱和度参数
        self.b = b         # 长度归一化强度
        self.N = len(documents)
        self.docs = [list(jieba.cut(doc)) for doc in documents]
        self.avgdl = sum(len(d) for d in self.docs) / self.N
        self.df = {}       # df[词] = 出现过该词的文档数
        for doc in self.docs:
            for word in set(doc):
                self.df[word] = self.df.get(word, 0) + 1

    def _idf(self, word: str) -> float:
        n = self.df.get(word, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query: str, doc_tokens: list[str]) -> float:
        score = 0.0
        dl = len(doc_tokens)
        for word in jieba.cut(query):
            if word not in doc_tokens:
                continue
            f = doc_tokens.count(word)
            # TF 饱和 + 长度归一
            tf = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            score += self._idf(word) * tf
        return score

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, int]]:
        scores = [(self.score(query, doc), i) for i, doc in enumerate(self.docs)]
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]
```

**BM25 vs 第一章的简单关键词打分**：

| 维度 | 第一章（命中计数） | BM25 |
|------|------------------|------|
| 词频处理 | 出现 1 次和 100 次得分一样 | 词频饱和，出现越多分越高但有上限 |
| 文档长度 | 不处理，长文档占优势 | 长度归一化，长短文档公平竞争 |
| 稀有词权重 | 所有词等权 | IDF 给稀有词更高权重 |
| 区分度 | 低（多数文档得分相近） | 高 |

### 6.3 混合检索的两种融合方式

#### 方式一：分数归一化 + 加权求和

```python
import numpy as np

def normalize(scores: list[float]) -> np.ndarray:
    """Min-Max 归一化到 [0, 1]"""
    s = np.array(scores)
    return (s - s.min()) / (s.max() - s.min() + 1e-8)

def hybrid_search_weighted(query, documents, faiss_index, embed_model, bm25, alpha=0.7, top_k=3):
    """alpha=0.7 表示向量权重 70%，BM25 权重 30%"""
    # ① 向量检索（对所有文档）
    q_vec = embed_model.encode([query], normalize_embeddings=True)[0]
    q_vec_faiss = q_vec.reshape(1, -1).astype(np.float32)
    vec_scores_all = faiss_index.search(q_vec_faiss, len(documents))[0][0]

    # ② BM25 检索（对所有文档）
    bm25_scores_all = bm25.search(query, len(documents))

    # ③ 归一化 + 加权
    vec_norm = normalize(vec_scores_all)
    bm25_norm = normalize([s for s, _ in bm25_scores_all])
    final_scores = alpha * vec_norm + (1 - alpha) * bm25_norm

    # ④ 取 top_k
    top_indices = np.argsort(final_scores)[::-1][:top_k]
    return [documents[i] for i in top_indices]
```

#### 方式二：RRF（Reciprocal Rank Fusion）—— 不需要归一化

```python
def rrf_fusion(rank_list_a: list[int], rank_list_b: list[int], k=60) -> dict[int, float]:
    """
    RRF：按排名融合。谁在两个列表里都排得高，谁的最终分就高。
    优点：不需要归一化，对排名的绝对分数不敏感。
    """
    scores = {}
    for rank, doc_id in enumerate(rank_list_a):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(rank_list_b):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
```

**选哪种？**

| 场景 | 推荐 |
|------|------|
| 你能拿到每个文档的分数（相似度、BM25 分） | 方式一（加权求和），信息更丰富 |
| 你只有排名（如调用第三方 API 只返回排序列表） | 方式二（RRF），最鲁棒 |
| alpha 调了好几次都不对 | 换 RRF，免调参 |

### 6.4 alpha 怎么调

```
alpha = 0.0  → 纯 BM25（精确匹配优先）
alpha = 0.5  → 等权融合
alpha = 0.7  → 偏语义（推荐默认值）
alpha = 1.0  → 纯向量（同义词语义优先）

经验：
- 用户查询偏事实性（人名、地名、数字）→ alpha 调低，多信赖 BM25
- 用户查询偏描述性（"适合家用的""性价比高的"）→ alpha 调高，多信赖向量
- 更好的做法：不要固定 alpha，用一个小模型判断 query 是精确型还是模糊型，动态调整
```

---

## 七、上下文拼接：组装 Prompt

### 7.1 拼接策略

```python
def build_rag_prompt(query: str, retrieved_docs: list[dict]) -> list[dict]:
    """将检索结果组装成 LLM 对话格式"""
    # 拼上下文
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        context_parts.append(
            f"【参考片段 {i+1}】来源：{doc['source']}\n{doc['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "你是一个专业的知识问答助手。请严格根据下面提供的参考片段回答问题。"
        "规则：\n"
        "1. 如果参考片段中有足够信息，请据实回答并引用来源\n"
        "2. 如果信息不充分，请明确说'目前的资料不足以回答这个问题'\n"
        "3. 不要编造任何参考片段中没有的数据、日期或事实\n"
        "4. 如果参考片段之间存在矛盾，请指出矛盾并列出两方说法"
    )

    user_prompt = (
        f"【参考资料】\n{context}\n\n"
        f"【用户问题】\n{query}\n\n"
        f"请根据以上参考资料回答。"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
```

### 7.2 Prompt 设计的关键细节

```
1. 编号每个参考片段 → LLM 回答时可以引用"根据参考片段 3..."，用户可追溯
2. System Prompt 明确"不知道"的边界 → 防幻觉的第一道防线
3. 处理信息矛盾 → 让 LLM 列出两方说法，而不是自己"仲裁"出一个假答案
4. Token 预算控制：
   - 单次检索：top_k=3~5 条，每条 300-500 字 → 约 1500-2500 字上下文
   - 加上 system + user prompt → 总共 2000-3500 tokens
   - 留够生成空间（LLM 输出可能 500-2000 tokens）
```

### 7.3 检索结果为空时怎么办

```python
if not retrieved_docs or all(doc["score"] < threshold for doc in retrieved_docs):
    return "抱歉，知识库中暂时没有找到与您问题相关的信息。建议您换一种问法，或者联系管理员补充相关知识。"
```

**不要硬编答案。** 用户宁愿看到"不知道"，也不愿看到一个貌似合理但完全错误的回答。

---

## 八、LLM 生成：调用大模型

### 8.1 生成配置

```python
from openai import OpenAI

client = OpenAI(api_key="...", base_url="...")

response = client.chat.completions.create(
    model="gpt-4o",        # 或 deepseek-chat / glm-4 / qwen-max
    messages=messages,
    temperature=0,         # RAG 场景用 0，稳定、不编造
    max_tokens=2000,       # 限制生成长度
    stream=True,           # 流式输出，用户体验更好
)
```

### 8.2 RAG 场景下的模型选型

| 模型 | 适用 | 原因 |
|------|------|------|
| GPT-4o / Claude Sonnet 4 | 高质量企业应用 | 指令遵循最强，不容易幻觉 |
| DeepSeek-V3 / Qwen-Max | 中文场景 | 中文理解好，性价比高 |
| GPT-4o-mini / DeepSeek-V2-Lite | 高并发低成本 | 简单问答够用 |
| 本地部署（Qwen2.5-7B） | 数据不出域 | 安全和隐私优先 |

### 8.3 防幻觉的三道防线

```
第一道：检索阶段
  → 只检索真正相关的文档（相似度 > 阈值，如 0.4）
  → 检索结果太少或相似度太低 → 直接说"不知道"

第二道：Prompt 阶段
  → System Prompt 强制"据实回答"
  → 要求引用来源（不引用 = 可能编造）
  → 明确告诉模型"信息不足时请承认"

第三道：后处理阶段（可选，生产级）
  → NLI（自然语言推理）模型检查：生成的回答是否被检索文档支持
  → 不支持的部分 → 标注或删除
```

---

# 第三部分：生产环境注意事项

## 九、架构设计

### 9.1 生产级 RAG 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Gateway                              │
│                    /upload（上传文档）  /chat（问答）               │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
     ┌─────────▼─────────┐      ┌─────────▼─────────┐
     │   Indexing Worker  │      │   Query Service    │
     │   (异步任务队列)    │      │   (同步/流式)      │
     │                    │      │                    │
     │  1. 解析文档       │      │  1. Query Embed    │
     │  2. Chunk 分块     │      │  2. 向量检索       │
     │  3. Embedding      │      │  3. BM25 检索      │
     │  4. 写入向量库     │      │  4. 混合排序       │
     │  5. 更新元数据库   │      │  5. 组 Prompt      │
     └────────────────────┘      │  6. 调 LLM         │
                                 │  7. 流式返回       │
                                 └────────────────────┘
                                          │
                                 ┌────────▼────────┐
                                 │  Vector DB       │
                                 │  (Milvus/FAISS)  │
                                 └────────┬────────┘
                                 ┌────────▼────────┐
                                 │  Metadata DB     │
                                 │  (PostgreSQL)    │
                                 └─────────────────┘
```

### 9.2 异步处理

文档上传后不要同步等待解析完成（大文件可能需要几分钟）：

```python
# 用户上传 → 返回任务 ID
task_id = submit_indexing_task(file_path)

# 后台 Celery / Redis Queue Worker 处理
@celery.task
def index_document(file_path):
    text = parse_document(file_path)
    chunks = chunk_document(text)
    embeddings = embed_chunks(chunks)
    store_to_vectordb(chunks, embeddings)
    update_status(task_id, "completed")

# 前端轮询任务状态
GET /task/{task_id} → {"status": "processing", "progress": "65%"}
```

### 9.3 增量更新（不要每次重建）

```python
# 新增文档：只 embed 新 chunk，追加到索引
new_chunks = chunk_document(new_doc)
new_vectors = embed_model.encode([c["content"] for c in new_chunks])
index.add(new_vectors)
id_to_doc.update({new_id: chunk for new_id, chunk in enumerate(new_chunks)})

# 删除文档：标记删除，定期清理
deleted_ids = find_chunks_by_source("旧文档.pdf")
for did in deleted_ids:
    index.remove_ids(np.array([did]))   # FAISS 支持
```

---

## 十、评估体系

### 10.1 RAG 需要评估什么

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  检索质量      │    │  生成质量      │    │  端到端质量    │
│              │    │              │    │              │
│ - Hit Rate   │    │ - 忠实度      │    │ - 用户满意度   │
│ - MRR        │    │ - 相关性      │    │ - 回答准确率   │
│ - NDCG       │    │ - 无害性      │    │ - 响应时间    │
└──────────────┘    └──────────────┘    └──────────────┘
```

### 10.2 检索质量指标

| 指标 | 含义 | 计算 |
|------|------|------|
| **Hit Rate** | Top-K 里有没有正确答案 | 命中数 / 总查询数 |
| **MRR** (Mean Reciprocal Rank) | 第一个正确答案排在第几位 | (1/正确答案排位) 的平均值 |
| **NDCG** | 排名质量（越靠前越好） | 考虑位置权重的相关性平均 |

```python
# 评估示例
test_queries = [
    {"query": "小米SU7续航多少", "relevant_doc_id": "xiaomi_su7"},
    {"query": "购置税政策", "relevant_doc_id": "report_01#sec_6"},
]

hit_count = 0
for test in test_queries:
    results = hybrid_search(test["query"], ...)
    if any(test["relevant_doc_id"] in doc["source"] for doc in results):
        hit_count += 1

print(f"Hit Rate: {hit_count}/{len(test_queries)} = {hit_count/len(test_queries):.1%}")
```

### 10.3 生成质量评估

| 指标 | 含义 | 评估方式 |
|------|------|---------|
| **忠实度（Faithfulness）** | 回答是否基于检索文档 | LLM-as-Judge 或 NLI 模型 |
| **答案相关性** | 回答是否切题 | LLM-as-Judge |
| **上下文召回率** | 检索到的文档是否覆盖所有需要的信息 | 人工标注 |

---

## 十一、常见问题与解决

| 问题 | 症状 | 根因 | 解法 |
|------|------|------|------|
| 检索结果全是同一来源 | 用户问什么都返回同一篇文档 | 该文档包含大量通用词 | 增加 diversity rerank（MMR 算法） |
| 检索结果不相关 | 相似度都很低或结果胡扯 | Chunk 粒度不对或模型不匹配 | 换 chunk 策略 + 验证模型和语言是否匹配 |
| LLM 编造信息 | 回答里有检索文档里没有的数字 | System Prompt 约束不够 | 加强约束 + 加 NLI 后验证 |
| 数字/日期经常错 | "25万以内"搜出 30 万的车 | 向量不理解数值约束 | 混合检索 + metadata 过滤 |
| 大文件上传超时 | >100MB PDF 上传后前端一直等待 | 同步处理大文件太慢 | 异步任务队列 |
| 索引膨胀 | 向量库占用磁盘远大于原始文档 | 文档多 + 维度高 | 用 PQ（乘积量化）压缩向量 |
| 召回太多无关内容 | Top-5 里 3 条不相关 | 相似度阈值太低 | 设 `threshold=0.4`，低于此值直接过滤 |
| 多轮对话上下文丢失 | 第二问"那它的价格呢？" LLM 不知道"它"指什么 | 没有对话历史 | 第五章要解决的：多轮对话 + Query 改写 |

---

## 十二、全流程技术栈速查

```
文档解析：PyMuPDF + python-docx + python-pptx + openai-whisper
文本分块：正则/RecursiveCharacterTextSplitter
向量化：  BAAI/bge-base-zh-v1.5 + sentence-transformers
向量索引：FAISS (<10万条) / Milvus (>10万条) / ChromaDB (原型)
BM25检索：自实现或 rank-bm25 库
LLM调用：openai SDK（兼容任何 OpenAI 接口的模型）
异步任务：Celery + Redis / FastAPI BackgroundTasks
元数据库：PostgreSQL / SQLite
监控：    LangSmith / 自建日志 + Prometheus
```

---

## 总结

RAG 的本质是一句话：

> **不要让 LLM 凭记忆回答，先把相关资料找出来，让 LLM 照着资料回答。**

全流程 8 个环节，每个环节都有坑，但核心原则只有一个：**检索质量决定回答质量。** 检索不出好的上下文，再强的 LLM 也答不好。

四个最重要的生产决策：

```
1. 文档解析 → 花 30% 的精力，这是地基
2. Chunk 大小 → 200-500 字，一个能独立回答问题的语义单元
3. 混合检索 → 向量 + BM25，不要只用一种
4. 防幻觉 → 三道防线：检索阈值 + Prompt 约束 + NLI 后验证
```
