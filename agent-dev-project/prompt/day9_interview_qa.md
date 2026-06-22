# Day 9 面试题：图片识别 + RAG（Vision RAG）

> 对应文件：`prompt/vision_rag.py`
> 核心能力：Base64 图片编码、多模态模型识别、图片描述转 Embedding、Vision RAG 全链路、与纯文本 RAG 对比

---

## 为什么需要 Vision RAG？

Day 2 的 RAG 管道有一个隐含前提：用户输入是文本。但真实场景中，用户可能发一张汽车照片问"这是什么车？参数怎么样？"——纯文本 RAG 无法处理这个场景。

Vision RAG 的核心思路：**图片不能直接进向量库，先让多模态模型把图片转成文本描述，再用文本描述做 Embedding 检索**。

---

## 端到端信息流（5 步全链路）

```
Step 1: 多模态识别 — 图片 → Base64 → vision_llm → 结构化描述
Step 2: 实体提取 — 描述 → text_llm → '品牌 型号'（用于 BM25）
Step 3: 混合检索 — dense(语义) + sparse(关键词) → RRF 融合
Step 4: Reranker 精排 — hybrid top-6 → reranker → top-3
Step 5: LLM 生成 — context + description + question → 最终回答
```

**架构原则**：检索+生成全链路复用纯文本 RAG，差异仅在检索入口（image→text→Embedding），这叫"最小侵入性原则"。

---

## Vision 作为 Agent Tool 的 ReAct 协作

```
用户上传车照 + "这车多少钱？"
    │
    ▼
Agent 第 1 轮 ReAct:
  Thought: "用户上传图片，先识别车型"
  Action: analyze_car_image(path, "identify")
  Observation: {result: "小鹏G6 深灰SUV..."}
    │
    ▼
Agent 第 2 轮 ReAct:
  Thought: "已确认小鹏G6，查价格和参数"
  Action: get_car_price(brand="小鹏", model="G6")
  Action: search_car_knowledge(query="小鹏G6续航智驾")
    │
    ▼
Agent 第 3 轮 ReAct:
  Thought: "数据齐全，可以回答" → 最终推荐
```

**三种分析类型**：

| analysis_type | 输入 | 输出 → 下游动作 |
|---------------|------|-----------------|
| `identify` | 外观照片 | 品牌型号 → 查价格/参数 |
| `dashboard` | 仪表盘照片 | 故障灯含义 → 维修建议 |
| `damage` | 事故照片 | 损伤评估 → 维修费用 |

---

## 纯文本 RAG vs Vision RAG 对比

### 场景差异

| 场景 | 纯文本 RAG | Vision RAG | 推荐 |
|------|:---:|:---:|:---:|
| 用户知道品牌型号："小鹏G6续航" | 直接搜→精准 | 多一步识别 | 纯文本 |
| 用户不知道型号：只拍了张车照 | 描述困难→检索差 | 图片识别→精准 | Vision |
| 用户发来仪表盘："这灯什么意思" | 无法用文字描述 | 识别故障灯→维修 | Vision |
| 预算+需求："20万买什么SUV" | 直接描述需求 | 图片无预算信息 | 纯文本 |

### 量化对比

| 维度 | 纯文本 RAG | Vision RAG |
|------|:---:|:---:|
| image→text 延迟 | 0ms | ~500-1500ms |
| 额外 token 消耗 | 0 | 85-500/张 |
| 对无文字输入的鲁棒性 | ❌ | ✅ |
| 检索精度（有图片时） | 低 | 高 |

### 选型决策树

```
用户输入类型？
├── 纯文本（"小米SU7续航"）→ 纯文本 RAG（零额外成本）
├── 文本 + 图片（"这车怎么样" + 照片）
│   └── Vision RAG — 图片用于实体识别，文本用于意图理解
├── 纯图片（无文字咨询）
│   └── Vision RAG — 先识别再检索
└── 多张图片（事故照片 + 仪表盘）
    └── Vision RAG — 每张分别识别，信息汇总后检索
```

---

## Q1：多模态模型是怎么"看"图片的？和人类看图片有什么本质区别？

**一句话**：多模态模型把图片切成小块（patches），每个 patch 用视觉编码器（ViT）转成向量，然后和文本 token 一起输入 Transformer。它"看"到的不是图片，是向量序列。

**技术细节**：

```
原始图片 (1024×1024, RGB)
  → 切成 64×64 个 16×16 的 patch
  → 每个 patch 通过 ViT 转成一个 768/1024 维向量
  → 得到 4096 个视觉 token
  → 和文本 token 一起送入 LLM 的 Transformer
  → Self-Attention 在视觉 token 和文本 token 之间计算关联
```

**和人类视觉的对比**：

| 维度 | 人类 | 多模态模型 |
|------|------|-----------|
| 输入表示 | 视网膜光信号 | patch 向量序列 |
| 理解方式 | 分层抽象（边缘→纹理→物体→语义） | Self-Attention 全局关联 |
| 细节敏感度 | 关注焦点区域 | 取决于 `detail` 参数 |
| 文字识别 | OCR 能力 | 需要明确训练才会读文字 |
| 偏见来源 | 个人经验 | 训练数据分布 |

**面试话术**："多模态模型的'看'本质上是文本 token 和视觉 token 在同一个 Self-Attention 空间中做关联计算。这意味着它对'图中文字'的识别不是 OCR 引擎做的，而是 Attention 机制学会了文字 token 和图像 patch 之间的对应关系。这是一个非常重要的工程理解——它解释了为什么多模态模型有时会'读错'图中的文字。"

---

## Q2：多模态模型的 API 消息格式和纯文本有什么不同？content block 数组的设计好在哪里？

**一句话**：纯文本的 `content` 是字符串，多模态的 `content` 是内容块数组——每个块可以是 `{"type": "text", "text": ...}` 或 `{"type": "image_url", "image_url": {"url": "data:image/..."}}`。

**API 消息格式对比**：

```python
# 纯文本消息
{"role": "user", "content": "这是什么车？"}

# 多模态消息（文本 + 图片混合）
{"role": "user", "content": [
    {"type": "text", "text": "请识别这辆车的品牌和型号"},
    {"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    }}
]}
```

**设计好处**：
- 文本和图片可以任意组合——一条消息可以包含多张图片、多处文字说明
- 可以精确控制图片和文字的先后顺序（先给图让 LLM 观察，再给文字指令）
- 扩展性好——未来新增音频块（`{"type": "audio_url", ...}`）不需要改 API 结构

**面试话术**："content block 数组是 OpenAI 引领的多模态 API 标准，Google Gemini、Anthropic Claude 都采用了类似设计。它的核心洞察是：图片和文字在消息中的地位是平等的——都是'一个消息块'，只是类型不同。"

---

## Q3：为什么图片不能直接做 Embedding 进向量库？如果一定要做，需要什么？

**一句话**：文本 Embedding 模型（BGE/Sentence-BERT/OpenAI Embedding）的输入是 token ID 序列，不接受像素矩阵。如果一定要做图片 Embedding，需要 CLIP 等多模态 Embedding 模型。

**语义空间不一致的直观理解**：
- 文本模型："小鹏G6" → [0.7, -0.2, 0.3, ...]（在"汽车型号"语义空间中）
- 图片模型：G6照片 → [0.1, 0.8, -0.5, ...]（在"视觉特征"语义空间中）
- 两者不在同一坐标系中，点积无意义

**如果一定要走图片→向量（CLIP 方案）**：

```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 图片 → 向量
image_embedding = model.get_image_features(**processor(images=image, return_tensors="pt"))
# 文本 → 向量（同一空间！）
text_embedding = model.get_text_features(**processor(text=["小鹏G6"], return_tensors="pt", padding=True))
# 可以直接算相似度
similarity = (image_embedding @ text_embedding.T).item()
```

**为什么 Day 9 不推荐直接走 CLIP**：虽然技术上可行，但引入 CLIP 意味着你的向量库需要存两种不同类型的向量（文本 Embedding + 图片 Embedding），检索时需要分两条路径，维护复杂度翻倍。image→text→Embedding 方案只需一种向量类型。

**面试话术**："是否直接用 CLIP 做多模态 Embedding 取决于场景。如果你的应用每天要处理 > 10 万张图片且延迟要求 < 100ms，CLIP 的端到端向量检索值得投资。但对于大多数 Agent 场景，图片少且图片→文本描述不仅能用于检索，还能展示给用户——一举两得。"

---

## Q4：图片转文本描述后，描述的质量对后续检索有多大影响？如何优化描述质量？

**一句话**：描述质量直接决定检索召回率——如果描述错了车型名，后续检索就全偏了。优化关键是结构化 Prompt 设计。

**描述质量→检索质量的因果链**：

```
描述正确："小鹏G6 深灰SUV 贯穿式灯带"
  → 检索: "小鹏G6" → ✅ 命中小鹏G6参数文档 → 精排 Top-3 全相关

描述错误："比亚迪宋 白色SUV"（实际是小鹏G6）
  → 检索: "比亚迪宋" → ❌ 全错，所有检索结果都不相关

描述模糊："一辆灰色新能源SUV"
  → 检索: "灰色 新能源 SUV" → ⚠️ 召回范围太广，精度低
```

**优化描述质量的 Prompt 设计**：

```python
# ❌ 弱 Prompt
"请描述这张图片"

# ✅ 结构化 Prompt
"""
请按以下结构描述这张汽车图片：
1. 【品牌】识别车标、尾标，不确定则列可能品牌
2. 【车型】识别具体型号（如G6、SU7、Model Y）
3. 【类型】轿车/SUV/MPV/跑车/皮卡
4. 【颜色】车身颜色
5. 【外观特征】（5-8条）：灯组、格栅、轮毂、门把手、腰线
6. 【置信度】品牌和型号识别置信度（高/中/低）
"""
```

**三级优化手段**：

| 级别 | 手段 | 代价 |
|------|------|------|
| L1 — Prompt 设计 | 结构化输出指令 | 无额外成本 |
| L2 — 多角度识别 | 多张图片（正面/侧面/车尾/LOGO特写）→ 合并描述 | N× API 调用 |
| L3 — 实体提取二次查询 | 描述中提取品牌型号 → 关键词查询 → 与语义查询并行 | 1× 额外 LLM 调用 |

**面试话术**："Vision RAG 的最大瓶颈不是多模态模型的识别能力，而是识别结果到检索 query 的转换质量。一个'蓝色SUV'描述到达不了'小鹏G6'的文档。所以一定要在管道中加一个实体提取步骤——从自然语言描述中提取品牌型号，作为关键词查询的输入。这实际是 Query Rewrite 在 Vision 场景下的应用。"

---

## Q5：多模态模型的 Token 计费模型是怎样的？图片怎么计费？

**一句话**：图片按分辨率和数量计费而非 token——通常切成固定大小的 patch（如 512×512），每个 tile 计固定 token 数。一张普通照片约 500-1000 token。

**高分辨率 tiles 计算**：

```
1. 缩放：使图片适应 2048×2048 的框
2. 切 tile：将缩放后的图片切成 512×512 的 tile
3. 每个 tile 消耗 170 tokens
4. 总消耗 = 85（基础） + 170 × n_tiles

例子：
  1024×1024 → 4 个 512 tile → 85 + 170×4 = 765 tokens
  2048×2048 → 16 个 512 tile → 85 + 170×16 = 2805 tokens
```

**三种 detail 模式**：

| detail | 计算方式 | 典型消耗 |
|--------|---------|:---:|
| `"low"` | 固定 85 tokens | 85 tokens |
| `"high"` | 按 tiles 数量动态计算 | 85 + 170 × tiles |
| `"auto"` | 模型自动选择 | 取决于图片尺寸 |

**图片预处理优化**：

```python
def preprocess_image_for_vision(image_path: str, max_size: int = 1024, quality: int = 85) -> str:
    """缩小+压缩图片以减少 tile 消耗。
    4000×3000 原图 ~2800 tokens → 1024×768 ~765 tokens。"""
    from PIL import Image
    import io
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
```

**面试话术**："图片计费是 Vision RAG 成本的主要来源。生产环境必须在编码前做图片预处理：把 4000×3000 的原始照片缩放到 1024×768 再 quality 降到 85%，token 消耗从 ~3000 降到 ~500，视觉质量差异人眼几乎不可察觉。这个优化每个请求省 ~2500 tokens。"

---

## Q6：什么时候用 Vision RAG，什么时候纯文本 RAG 就够？决策框架是什么？

**一句话**：看用户输入中是否包含文本无法表达的视觉信息——如果有，走 Vision 路径；如果文本已经足够，走更便宜的纯文本路径。

**决策框架**：

```
用户输入
├── 只有文字
│   └── 纯文本 RAG（更便宜，更精准）
├── 有图片 + 用户知道这是什么
│   └── 检测图片是否提供额外信息
│       ├── 图片只是配图（如表情包）→ 忽略图片，走纯文本
│       └── 图片包含关键信息（如车损照片）→ Vision RAG
└── 有图片 + 用户不知道这是什么（"帮我看看这是什么车"）
    └── 必须走 Vision RAG，纯文本无法回答
```

**面试话术**："Vision RAG 的成本是纯文本的 2-5 倍。生产环境中的正确做法是前端做个简单判断——如果用户没上传图片，直接跳过 Vision 分支。多模态模型只在需要时才调用，这是成本意识。Vision RAG 不是纯文本 RAG 的替代品，而是补充——它解决的是'用户无法用文字准确描述'这个痛点。"

---

## Q7：多模态 LLM 能做的不只是"描述图片"——它在 Agent 中还能做什么？

**一句话**：在 Agent 场景中，多模态 LLM 可以识别仪表盘故障灯、读取图表数据、对比两张图片的差异、分析事故照片——远不止"这是什么车"。

**汽车导购场景的扩展应用**：

| 场景 | 输入 | 多模态 LLM 的能力 | 下游动作 |
|------|------|------------------|---------|
| 车型识别 | 外观照片 | 品牌+型号识别 | 查询价格/参数 |
| 故障诊断 | 仪表盘照片 | 识别故障灯 → 描述含义 | 建议维修方案 |
| 损伤评估 | 事故照片 | 识别受损部位和程度 | 估算维修费用 |
| 对比选车 | 两张车的内饰照片 | 对比内饰风格和配置 | 推荐更合适的车 |
| 说明书解读 | 说明书/按钮面板照片 | 识别按钮功能 | 解释操作步骤 |
| 里程估算 | 仪表盘续航+电量照片 | 读取数字+估算 | 建议充电方案 |

**Agent 集成方式**：多模态 LLM 作为 Agent 的一个"感知工具"

```python
@tool
def analyze_car_image(image_path: str, analysis_type: str = "identify") -> str:
    """分析汽车图片：identify(车型识别) / dashboard(故障灯) / damage(损伤评估)"""
    ...
```

**面试话术**："多模态 LLM 在 Agent 中的角色是'感知层'——它把非结构化的视觉信息转成结构化的文本数据，让 Agent 的逻辑层（工具调用、状态流转、决策判断）能正常消费。视觉理解在 Agent 中的位置应该在'输入预处理层'，而非'核心推理层'。这样视觉能力的增强不影响 Agent 的主体逻辑。"

---

## Q8：多模态模型有哪些常见陷阱？如何在实际项目中规避？

**一句话**：四大陷阱——幻觉（认错车）、文字 OCR 不可靠、对罕见车型识别差、图片角度/光线敏感。

**四大陷阱与对策**：

| 陷阱 | 表现 | 对策 |
|------|------|------|
| 视觉幻觉 | 把普通 SUV 认成库里南 | 要求输出置信度；低置信度时追问用户 |
| OCR 不可靠 | 车尾标"G6"读成"G8" | 不依赖 OCR 单点判断；结合外观特征交叉验证 |
| 罕见车型盲区 | 对冷门品牌完全识别不出 | 返回"可能"列表而非强行指定；引导用户补充文字 |
| 角度/光线敏感 | 同一辆车不同角度识别结果不同 | 建议用户拍正面+尾部；多角度融合判断 |

**代码中的防御设计**：

```python
def identify_car_with_confidence(image_path: str) -> dict:
    """带置信度的车型识别 + 不确定性处理"""
    description = describe_car_image(image_path)
    
    if result.confidence == "低":
        return {"status": "need_confirmation", "candidates": [...],
                "message": "无法从图片确定车型，请问是以下哪款？"}
    elif result.confidence == "中":
        return {"status": "uncertain", "best_guess": result.model_name,
                "message": f"可能是{result.model_name}，如有误请告知"}
    else:
        return {"status": "confident", "model": result.model_name}
```

**面试话术**："多模态模型最大的坑是'看起来能识别 = 实际识别准确'的错觉。生产环境必须建立置信度机制——让模型输出不确定性估计，低置信度时降级到追问用户或人工审核。这和传统 OCR 的'识别置信度'是同一个工程思路。"

---

## Q9：Vision RAG 和 Day 3 Tool Calling + Day 5 ReAct + Day 8 Structured Output 如何串联？

**一句话**：Vision 作为 Agent 的输入通道（Tool），识别结果用 Day 8 的结构化输出格式化，然后参与 Agent 的 ReAct 推理循环。

**三个 Day 的协同**：

| Day | 能力 | 在 Vision Agent 中的角色 |
|-----|------|------------------------|
| Day 3 | Tool Calling 循环 | 编排图片识别→价格查询→推荐的多步推理 |
| Day 5 | ReAct Agent | Thought→Action→Observation 循环框架 |
| Day 8 | Structured Output | 图片识别结果 + 最终推荐 都用 Pydantic 校验 |
| Day 9 | Vision 感知 | 新增 analyze_car_image 工具，扩展 Agent 感知维度 |

**面试话术**："Vision RAG 不是一个孤立的能力——它通过 Tool Calling 机制融入 Agent 的推理循环。图片识别只是 Agent 的一步 Action，识别结果通过 Structured Output 格式化后作为 Observation 回传给 LLM，LLM 基于它决定下一步。这就是多模态 Agent 的标准架构。"

---

## Q10：多模态 RAG 的下一步演进方向是什么？

**一句话**：从"图片→文本→检索"的串行管道演进为"多模态并行理解 + 交叉验证"——图文各自独立编码，在决策层融合。

**三个演进方向**：

1. **Multi-Vector 检索**：同一文档同时存储文本向量和图片向量（CLIP），查询时两张向量并行检索，结果融合
2. **多模态 LLM 原生检索**：未来可能不需要"图片→文本"的转换——支持直接输入图片的 LLM 内部做图文对齐
3. **视频 RAG**：视频帧抽样 + 时序分析 → 理解"发生了什么"而不只是"看到了什么"

**面试话术**："当前的 Vision RAG 是过渡方案——把图片降解为文本再检索。未来多模态模型会直接支持图文混合检索，但这个过渡期还要持续 1-2 年。理解当前方案的局限性本身就是一种工程判断力。"

---

### Day 9 面试自检清单

| # | 问题 | 能答出吗？ |
|---|------|:---:|
| 1 | 多模态模型如何"看"图片？ViT patch + Transformer 的流程？ | □ |
| 2 | 多模态 API 消息格式和纯文本的区别？content block 数组的设计好在哪里？ | □ |
| 3 | 为什么图片不直接 Embedding？如果一定要做需要什么（CLIP）？ | □ |
| 4 | 图片描述质量对检索的影响？三级优化手段？ | □ |
| 5 | 多模态模型的 token 计费方式？图片预处理优化策略？ | □ |
| 6 | Vision RAG vs 纯文本 RAG 的决策框架？ | □ |
| 7 | 多模态 LLM 在 Agent 中除了"描述图片"还能做什么（6 个场景）？ | □ |
| 8 | 多模态模型四大陷阱（幻觉/OCR/罕见车型/角度敏感）及对策？ | □ |
| 9 | Day 9 + Day 3 + Day 5 + Day 8 如何串联成多模态 Agent？ | □ |
| 10 | 多模态 RAG 的三个演进方向？当前方案的局限性？ | □ |
