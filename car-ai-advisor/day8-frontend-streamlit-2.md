# Day 8 — Streamlit 前端（下）：对话组件 + 侧边栏 + 工具可视化

> **今日目标**：实现可复用的 UI 组件层——对话气泡、会话管理侧边栏、Agent 推理过程可视化。Day 8 是 Phase 4 的前端下半场——将 Day 7 的单体 app.py 拆分为模块化组件，为后续扩展和维护打好基础。

---

## 目录

1. [今日任务清单](#1-今日任务清单)
2. [组件化架构设计](#2-组件化架构设计)
3. [对话气泡渲染组件](#3-对话气泡渲染组件)
4. [会话管理侧边栏](#4-会话管理侧边栏)
5. [工具调用可视化](#5-工具调用可视化)
6. [组件间数据流](#6-组件间数据流)
7. [核心技术原理](#7-核心技术原理)
8. [初学者常见疑问](#8-初学者常见疑问)
9. [面试模拟问答](#9-面试模拟问答)

---

## 1. 今日任务清单

| 文件 | 行数 | 做什么 |
|------|:----:|------|
| `frontend/components/chat.py` | 56 | `render_message()` 对话气泡 + `render_sources()` 来源引用标签（分数颜色标记） |
| `frontend/components/sidebar.py` | 169 | `render_sidebar()` 会话 CRUD：新建/切换(加载历史)/重命名(行内编辑)/删除(确认弹窗) |
| `frontend/components/tools.py` | 80 | `render_tool_calls()` Agent 推理过程折叠面板 + 5 工具 icon 映射 + 中文名 |

---

## 2. 组件化架构设计

### 2.1 从单体到组件的演进

Day 7 的 app.py 把所有 UI 逻辑写在一个文件里。Day 8 拆分为三层组件：

```
app.py (编排层)
  │  · 初始化 session_state
  │  · 编排渲染顺序
  │  · 处理用户输入 → API 调用 → 更新状态
  │
  ├── components/sidebar.py ── 侧边栏
  │     · render_sidebar(client)
  │     · 会话列表 + CRUD
  │     · 历史消息加载
  │
  ├── components/chat.py ── 对话区
  │     · render_message(role, content, sources)
  │     · render_sources(sources)
  │     · 气泡样式 + 来源标签
  │
  └── components/tools.py ── 工具面板
        · render_tool_calls(events)
        · render_tool_status(msg)
        · Agent 推理链可视化
```

**为什么这样拆分？**
- **单一职责**：每个组件只负责一类 UI 元素的渲染
- **可复用**：`render_message()` 可以被历史消息回显和实时对话复用
- **可测试**：组件是纯函数（接收数据 → 渲染 UI），不依赖全局状态
- **可替换**：想换一种会话管理方式？只替换 sidebar.py，chat.py 和 tools.py 不受影响

### 2.2 组件的数据契约

组件不持有状态，通过参数接收数据：

```python
# ✅ 好的设计：纯函数，数据从参数传入
def render_message(role: str, content: str, sources: list[dict] | None = None):
    with st.chat_message(role):
        st.markdown(content)
        if sources:
            render_sources(sources)

# ❌ 坏的设计：组件内部直接读 session_state
def render_message():
    role = st.session_state.role      # 隐式依赖，换上下文就崩溃
    content = st.session_state.content
```

这种模式就是 React 社区说的"受控组件"——组件的行为完全由 props（参数）控制。

---

## 3. 对话气泡渲染组件

### 3.1 render_message()

```python
def render_message(role: str, content: str, sources: list[dict] | None = None):
    avatar = "🧑" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if sources:
            render_sources(sources)
```

**为什么用 `st.chat_message()`？** Streamlit 1.35+ 内置了 `st.chat_message()` 和 `st.chat_input()`，原生支持对话 UI。它会自动处理气泡样式（用户靠右/蓝色，AI 靠左/灰色），比手动用 `st.container()` + CSS 简单得多。

**avatar emoji 的作用**：给用户和 AI 不同的头像图标，视觉上区分消息角色。`st.chat_message("user")` 默认用用户头像，`st.chat_message("assistant")` 默认用机器人头像，我们显式指定 emoji 方便国际化。

### 3.2 render_sources()

```python
def render_sources(sources: list[dict]):
    with st.expander(f"📚 参考来源 ({len(sources)})", expanded=False):
        for doc in sources:
            score = doc.get("score", 0.0)
            source_name = doc.get("source", "未知来源")

            # 分数颜色标记
            if score > 0.7:
                score_color = "green"     # 高相关
            elif score > 0.4:
                score_color = "orange"    # 中等相关
            else:
                score_color = "gray"      # 低相关

            st.markdown(f"**#{doc['rank']}** [{source_name}] :{score_color}[相关性: {score:.3f}]")
            st.caption(doc.get("content", "")[:200])
```

**设计要点**：
- **st.expander 默认折叠**（`expanded=False`）：来源信息是辅助参考，不应抢占主回答的视觉焦点
- **分数颜色三档**：绿色(>0.7) = 高度相关，橙色(0.4-0.7) = 中等，灰色(<0.4) = 低。用户扫一眼就能判断来源可信度
- **内容截断 200 字**：避免来源文本撑满屏幕。用户感兴趣可以展开看完整内容
- **排名序号 `#N`**：展示检索排序，排名越高越相关

### 3.3 来源数据格式

```json
{
  "rank": 1,
  "source": "vehicles.json",
  "content": "【比亚迪 宋PLUS DM-i】品牌：比亚迪...",
  "score": 0.856,
  "type": "vehicle"
}
```

这个格式与后端 `schemas/chat.py` 的 `SourceDoc` 模型完全对齐。前后端用同一套数据契约，不需要格式转换。

---

## 4. 会话管理侧边栏

### 4.1 功能矩阵

| 操作 | 触发 | 流程 |
|------|------|------|
| 新建会话 | 点击"➕ 新建会话" | `POST /sessions` → 获取 session_id → 设为当前会话 → 清空消息 → rerun |
| 切换会话 | 点击会话按钮 | 设为当前会话 → `GET /sessions/{id}/history` → 加载历史到 messages → rerun |
| 重命名 | 点击 ✏️ → 输入新名 → 确认 | 显示 text_input → `PATCH /sessions/{id}` → rerun |
| 删除 | 点击 🗑️ → 确认 | 显示 warning → `DELETE /sessions/{id}` → 如果删的是当前会话则切回 default → rerun |

### 4.2 状态管理设计

每个会话的交互状态（重命名中、删除确认中）用独立的 `session_state` key 管理：

```python
# 重命名状态
st.session_state[f"renaming_{sid}"] = True   # 进入重命名模式
st.session_state.pop(f"renaming_{sid}")      # 退出重命名模式

# 删除确认状态
st.session_state[f"confirm_delete_{sid}"] = True  # 显示确认弹窗
st.session_state.pop(f"confirm_delete_{sid}")     # 关闭确认弹窗
```

**为什么用 `f"renaming_{sid}"` 而不是 `st.session_state.renaming_sid`？** 因为每个会话的交互状态是互相独立的。用 `{sid}` 后缀可以同时打开多个会话的重命名/删除弹窗而互不干扰。

### 4.3 历史消息加载

切换会话时需要从 Redis 加载历史消息并恢复到当前界面：

```python
# 获取历史
hist = await client.get_history(sid)

# 解析消息
for msg in hist.get("messages", []):
    data = json.loads(msg) if isinstance(msg, str) else msg
    role = data.get("role", "user")
    content = data.get("content", "")
    if role in ("user", "assistant"):
        st.session_state.messages.append({"role": role, "content": content})
```

**后端存储格式**：消息在 Redis List 中存为 JSON 字符串 `'{"role":"user","content":"你好"}'`。前端加载时需要 `json.loads()` 反序列化。同时兼容已经是 dict 的情况（`isinstance(msg, str)` 判断），防止后端返回格式变化时前端崩溃。

### 4.4 重命名行内编辑

```
┌──────────────────────────────┐
│ 对话 1 (12)     [✏️] [🗑️]   │  ← 正常模式
├──────────────────────────────┤
│ ┌──────────────────────┐     │
│ │ 新名称_______________ │     │  ← 点击 ✏️ 后
│ └──────────────────────┘     │
│  [确认]  [取消]              │
└──────────────────────────────┘
```

不用弹窗而用行内编辑，减少用户的操作步数。两个按钮（确认/取消）提供明确的退出路径。

---

## 5. 工具调用可视化

### 5.1 Agent 推理链展示

ReAct 循环中 Agent 调用工具的完整过程通过 `st.expander` 折叠面板展示：

```python
def render_tool_calls(events: list[dict]):
    with st.expander(f"🔧 Agent 推理过程 ({len(events)} 步)", expanded=True):
        for i, evt in enumerate(events, 1):
            # Step N: 🔍 知识库检索
            st.markdown(f"**Step {i}**: {icon} {name_cn}")

            # 📥 参数
            with st.expander("📥 参数"):
                st.code(args_str, language="json")

            # 📤 结果
            with st.expander("📤 结果"):
                st.caption(result_str)
```

**为什么默认展开（`expanded=True`）？** 工具调用是 Agent 推理的关键过程。用户看到 Agent 做了什么，才能理解最终回答的依据。开源 AI 应用的一个核心竞争力就是**可解释性**——让用户看到"AI 不是凭空编造，而是先查了知识库再回答"。

### 5.2 工具映射表

```python
TOOL_ICONS: dict[str, str] = {
    "search_car_knowledge": "🔍",     # 知识库检索
    "get_car_price": "💰",           # 查询价格
    "compare_cars": "⚖️",             # 车型对比
    "recommend_cars": "🎯",          # 智能推荐
    "calculate_ownership_cost": "🧮", # 用车成本
}

TOOL_NAMES_CN: dict[str, str] = {
    "search_car_knowledge": "知识库检索",
    "get_car_price": "查询价格",
    "compare_cars": "车型对比",
    "recommend_cars": "智能推荐",
    "calculate_ownership_cost": "用车成本",
}
```

**为什么不用 LLM 返回的 `tool_name` 直接展示？** 工具名 `search_car_knowledge` 是面向机器的（英文、蛇形命名），前端应该转换为面向用户的中文名。emoji 图标让工具类型一目了然（🔍搜索、💰价格、⚖️对比），比纯文字更快识别。

### 5.3 工具调用事件流

SSE 事件流中携带工具调用信息：

```
data: {"type":"source","documents":[{"source":"tool:search_car_knowledge","content":"{\"query\":\"25万SUV\"}"}]}
data: {"type":"source","documents":[{"source":"tool:get_car_price","content":"{\"brand\":\"比亚迪\"}"}]}
```

前端从 `source` 字段提取工具信息：
```python
if tool_source.startswith("tool:"):
    tool_name = tool_source[len("tool:"):]
    tool_events.append({
        "tool": tool_name,
        "args": doc.get("content", ""),
    })
```

**设计权衡**：当前工具结果（Observation）也通过同一个 `source` 事件返回。理想情况下后端应该区分 `tool_start` / `tool_result` 两个事件类型，让前端更精确地展示。这留到后续优化（Phase 5）。

---

## 6. 组件间数据流

### 6.1 单向数据流

```
app.py (状态管理者)
  │
  │  st.session_state.messages     ← 消息数据源
  │  st.session_state.session_id   ← 当前会话
  │  st.session_state.tool_events  ← 工具事件
  │
  ├──→ sidebar.render_sidebar(client)
  │      └─ 读取: session_id, messages
  │      └─ 修改: session_id, messages (切换/删除时)
  │
  ├──→ chat.render_message(role, content, sources)
  │      └─ 纯展示: 只读参数，不修改状态
  │
  └──→ tools.render_tool_calls(events)
         └─ 纯展示: 只读参数，不修改状态
```

**关键约束**：`chat.py` 和 `tools.py` 是纯展示组件——接收参数渲染 UI，不修改 `st.session_state`。只有 `sidebar.py`（会话管理）和 `app.py`（对话流程）有权修改状态。这避免了"状态被谁改了"的调试噩梦。

### 6.2 并行执行与 rerun

Streamlit 中所有 `st.*` 调用是顺序执行的非阻塞操作。一个关键细节：

```python
# app.py 中的对话流程
with st.chat_message("user"):
    st.markdown(prompt)          # ① 先渲染用户消息

with st.chat_message("assistant"):
    placeholder = st.empty()     # ② 创建占位符
    async for event in stream:
        if event_type == "token":
            placeholder.markdown(full_text + "▌")  # ③ 流式更新
```

在 `async for` 循环中，`placeholder.markdown()` 每次调用都会触发 Streamlit 向前端推送更新。整个过程是同一个脚本上下文中的顺序执行——Streamlit 等 `asyncio.run()` 返回后才完成本次渲染周期。

---

## 7. 核心技术原理

### 7.1 st.expander 的 DOM 实现

`st.expander` 在前端渲染为一个 `<details>` HTML 元素：

```html
<details>
  <summary>📚 参考来源 (3)</summary>
  <div class="expander-content">
    <!-- 内部内容 -->
  </div>
</details>
```

`<details>` 元素是 HTML5 原生的折叠组件，不需要 JavaScript。点击 `<summary>` 切换 `open` 属性。Streamlit 的后端 diff 算法会跟踪 expander 的展开/折叠状态，重跑时恢复之前的 UI 状态。

### 7.2 Streamlit 的组件生命周期

每个 `st.*` 调用在 Streamlit 内部创建一个 "delta"（变化记录）：

```
st.markdown("Hello")
  → Delta(type="markdown", content="Hello", id="abc123")

st.button("Click")
  → Delta(type="button", label="Click", id="def456")
```

重跑时，Streamlit 对比新旧 delta 列表：
- 同 id、同参数 → 跳过（不重新渲染）
- 同 id、不同参数 → 更新 DOM 节点
- 新 id → 创建新 DOM 节点
- 旧 id 消失 → 删除 DOM 节点

这就是为什么 `placeholder.markdown(new_text)` 每次只更新一个节点——占位符的 id 不变，只是 content 变了。

### 7.3 历史消息加载的性能考虑

当前实现每次切换会话都从 Redis 全量加载历史消息。102 条消息约 50KB，加载耗时 <10ms（本地 Redis）。但生产环境需要考虑：

1. **消息量大的会话**（1000+ 条）：改为分页加载（`GET /sessions/{id}/history?limit=50&offset=0`）
2. **频繁切换会话**：前端缓存最近访问的会话历史
3. **消息压缩**：超过 100 条的消息自动摘要合并

当前阶段这些优化不必要——会话消息量小、切换频率低。

---

## 8. 初学者常见疑问

**Q: 为什么不把 sidebar.py 的逻辑写在 app.py 里？**

app.py 已经 161 行了，加上侧边栏的 169 行会变成 300+ 行的上帝文件。拆分为组件后：① 每个文件职责单一，改侧边栏只改 sidebar.py；② 不同组件可以并行开发和测试；③ 新成员加入时，看文件名就知道功能在哪。

**Q: 工具调用面板什么时候更新？**

当前实现在收到 `source` 事件时更新面板——收集所有带 `tool:` 前缀的 source 事件，然后调用 `render_tool_calls()`。后端的工具执行和 LLM 推理是顺序的（ReAct 循环），所以工具调用总是先于 token 到达。面板在 assistant 气泡内部渲染，用户看到打字结果的同时也能展开看到推理过程。

**Q: 会话重命名和删除的确认流程安全吗？**

确认流程用 `st.session_state[f"confirm_delete_{sid}"]` 控制，需要用户点击两次（🗑️ → 确认删除）。这不是服务端的安全机制——服务端已经做了鉴权（只有会话创建者能删除）。前端的确认弹窗只是防止用户误操作。

**Q: 为什么工具调用用 st.expander 而不是弹窗？**

弹窗（`st.dialog`）会打断用户浏览对话的流程。工具调用是对话的**附加上下文**，用户想看就展开，不想看就折叠。`st.expander` 默认展开让用户感知到"AI 确实做了推理"，但不会遮盖主回答。

**Q: sidebar.py 的 asyncio.run() 会影响性能吗？**

每次 CRUD 操作都创建一个新事件循环。操作频率低（用户手工点击，每秒 <1 次），单次延迟微秒级，感知不到。高频场景（如搜索框实时补全）需要复用事件循环，但会话管理不需要。

---

## 9. 面试模拟问答

> **Q: 你们前端是怎么展示 Agent 推理过程的？**

我们用 `st.expander` 折叠面板展示 ReAct 循环的每一步。Agent 调用工具时，后端通过 SSE `source` 事件推送工具名和参数，前端收集这些事件后用工具映射表（icon + 中文名）渲染为"Step N: 🔍 知识库检索"格式。每个步骤的入参用 `st.code()` JSON 高亮展示，出参用 `st.caption()` 展示。我们默认展开面板，因为可解释性是 AI 应用的核心竞争力——让用户看到 AI 不是凭空编造，而是先查了知识库再回答。

> **Q: 会话管理的 CRUD 状态是怎么维护的？**

每个会话的交互状态（重命名中、删除确认中）用独立的 `session_state` key 管理——`f"renaming_{sid}"` 和 `f"confirm_delete_{sid}"`。不同会话的交互状态互不干扰。重命名用行内编辑（点击 ✏️ → text_input → 确认/取消），删除用二次确认（🗑️ → warning → 确认删除/取消）。所有 CRUD 操作通过 `asyncio.run()` 调用异步 APIClient 方法。

> **Q: 对话历史是怎么在不同会话之间切换的？**

切换会话时调用 `GET /sessions/{id}/history` 从 Redis 加载历史消息，解析 JSON 后放入 `st.session_state.messages` 列表，然后 `st.rerun()` 重绘界面。后端消息在 Redis List 中存为 JSON 字符串，前端用 `json.loads()` 反序列化，同时兼容已经是 dict 的情况。当前全量加载（不翻页），因为会话消息量小、性能不是瓶颈。

> **Q: 你们的组件之间是怎么通信的？**

单向数据流——app.py 是状态管理者，持有 `st.session_state` 中的所有状态。chat.py 和 tools.py 是纯展示组件，只通过参数接收数据，不修改状态。sidebar.py 有权修改状态（会话 CRUD），通过 `st.session_state` 直接写入。这避免了"状态被谁改了"的调试噩梦。组件设计遵循"受控组件"模式——行为完全由参数决定。

> **Q: 检索来源的展示是怎么设计的？**

`render_sources()` 用 `st.expander` 折叠展示检索结果。每条来源显示：排名序号、源文件名、相关性分数（颜色标记三档：绿>0.7/橙>0.4/灰<0.4）、内容摘要（截断 200 字）。默认折叠不抢占主回答视觉焦点。分数颜色让用户一眼判断可信度——绿色高相关可以信赖，灰色低相关需要谨慎。数据格式与后端 `SourceDoc` 模型完全对齐。

> **Q: 如果后端返回了一个新的工具类型，前端会崩溃吗？**

不会。工具映射表用 `.get()` 安全访问：`TOOL_ICONS.get(tool_name, "🔌")` 和 `TOOL_NAMES_CN.get(tool_name, tool_name)`。未知工具名会显示通用图标 🔌 和原始英文名作为 fallback。这体现了"优雅降级"原则——新功能后端先上线，前端后续补充映射。

---

## 附：今日文件依赖关系

```
app.py
  ├── api_client.py ─────────────→ FastAPI 后端
  │     POST /chat   (SSE 流式)
  │     POST /sessions (CRUD)
  │
  ├── components/chat.py ──────── 对话区渲染
  │     render_message()   气泡 + 来源引用
  │     render_sources()   分数颜色 + 内容摘要
  │
  ├── components/sidebar.py ──── 侧边栏管理
  │     render_sidebar()   会话列表 + 新建/切换/重命名/删除
  │     asyncio.run()      桥接异步 API
  │
  └── components/tools.py ────── 工具可视化
        render_tool_calls()  推理过程折叠面板
        render_tool_status() 状态提示
        TOOL_ICONS / TOOL_NAMES_CN  工具映射表
```
