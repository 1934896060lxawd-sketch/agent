"""
第一章：Naive RAG —— 最简单的检索增强生成

管线：加载文档 → 关键词检索 → 拼接 prompt → 调用 LLM

你要实现的核心功能：
  1. 从 JSON/TXT 文件加载数据，转成统一格式的文档列表
  2. 用 jieba 分词做关键词匹配检索
  3. 构造 system prompt + user prompt（防幻觉指令）
  4. 调用 OpenAI 兼容 API 获取回答

环境变量（.env 文件）：LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_ID
"""

import json
import glob
import os
import re
import jieba
from openai import OpenAI
from dotenv import load_dotenv


# TODO: 1. load_dotenv() 加载 .env 文件
load_dotenv()
# TODO: 2. 创建 OpenAI client，api_key 和 base_url 从环境变量读取
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)
# TODO: 3. 从环境变量读取 model id
model = os.getenv("LLM_MODEL_ID")


# ============================================================
# 1. 数据转换 —— 把原始数据转成纯文本
# ============================================================

def car_to_text(car: dict) -> str:
    """
    把一辆车的结构化字典转成一段自然语言描述。

    输入 car 结构示例：
      {
        "brand": "比亚迪", "model": "海豚",
        "full_name": "比亚迪海豚",
        "category": "小型纯电轿车",
        "price_range": "9.98-12.98万元",
        "powertrain": {"type": "纯电", "battery_capacity_kwh": 45, ...},
        "performance": {"motor_power_kw": 70, "zero_to_hundred_seconds": 10.5},
        "smart_driving": {"level": "L2", "chip": "征程3", "lidar": false},
        "key_features": ["刀片电池", "12.8寸旋转屏", ...]
      }

    输出示例：
      "比亚迪海豚，小型纯电轿车，售价9.98-12.98万元。\n动力方面：纯电，45kWh电池..."

    提示：
      - 用 .get(key, default) 安全取值，防止 KeyError
      - 嵌套字段用变量接一下再取，例如 p = car.get("powertrain", {})
      - 列表用 "、".join() 拼接
      - 布尔值转中文：{'有' if sd.get('lidar') else '无'}激光雷达
    """
    # TODO: 实现
    p = car.get("powertrain", {})
    perf = car.get("performance", {})
    sd = car.get("smart_driving", {})

    features = "、".join(car.get("key_features", []))

    text = (
        f"{car.get('full_name', '未知车型')}，{car.get('category', '')}，"
        f"售价{car.get('price_range', '暂无')}。\n"
        f"动力方面：{p.get('type', '')}，{p.get('battery_capacity_kwh', '')}kWh电池，"
        f"CLTC续航{p.get('cltc_range_km', '')}km，"
        f"充电：{p.get('fast_charge', '')}。\n"
        f"性能：{perf.get('motor_power_kw', '')}kW电机，"
        f"零百加速{perf.get('zero_to_hundred_seconds', '')}秒。\n"
        f"智驾：{sd.get('level', '')}，{sd.get('chip', '')}，"
        f"{'有' if sd.get('lidar') else '无'}激光雷达。\n"
        f"亮点：{features}。"
    )
    return text
    
    


def review_to_text(review: dict) -> str:
    """
    把一条用户评价拍平成文本。

    输入 review 结构：{"model": "...", "owner": "...", "rating": 4,
                      "pros": [...], "cons": [...], "advice": "..."}

    输出示例：
      "车型：比亚迪海豚，车主：张先生，评分：4/5。优点：续航扎实、空间不错。缺点：内饰塑料感。建议：适合城市通勤。"
    """
    # TODO: 实现
    pros = "、".join(review.get("pros", []))
    cons = "、".join(review.get("cons", []))
    return (
        f"车型：{review.get('model', '')}，车主：{review.get('owner', '')}，"
        f"评分：{review.get('rating', '')}/5。"
        f"优点：{pros}。缺点：{cons}。"
        f"建议：{review.get('advice', '')}"
    )


def chunk_sections(text: str) -> list[dict]:
    """
    按「一、」「二、」...中文序号标题切块，每块 = 标题 + 正文。

    输入：一篇行业报告的完整文本
    输出：[{"content": "一、市场概况\n...", "chunk_id": "sec_0"}, ...]

    实现思路：
      1. 用 re.split(r'([一二三四五六七八九十]+、)', text) 切分
      2. 第一个元素如果是正文（不以序号开头），当作 preamble 单独一块
      3. 剩下的两两配对：序号 + 正文
    """
    # TODO: 实现
    pattern = r"([一二三四五六七八九十]+、)"
    parts = re.split(pattern, text)
    chunks = []
    i = 0
    if parts and not re.match(pattern, parts[0]):
        preamble = parts[0].strip()
        if preamble:
            chunks.append({"content": preamble, "chunk_id": "sec_preamble"})
        i = 1
    while i < len(parts):
        if re.match(pattern, parts[i]):
            title = parts[i]
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            chunks.append({
                "content": title + "\n" + body,
                "chunk_id": f"sec_{len(chunks)}"
            })
            i += 2
        else:
            i += 1
    return chunks


# ============================================================
# 2. 数据加载 —— 统一入口
# ============================================================

def load_data(data_dir: str) -> list[dict]:
    """
    加载所有数据源，返回格式统一的文档列表。

    每篇文档格式：{"content": "文本内容", "source": "来源标识", "type": "car_spec|industry_report_chunk|user_review"}

    数据源（在 data_dir 目录下）：
      ① cars_specs.json  — 车型规格，用 car_to_text() 转文本
      ② industry_reports/*.txt  — 行业报告，用 chunk_sections() 切块
      ③ user_reviews.json  — 用户评价，用 review_to_text() 转文本

    提示：glob.glob(os.path.join(data_dir, "industry_reports", "*.txt")) 遍历报告文件
    """
    documents = []

    # TODO: ① 加载 cars_specs.json，逐条用 car_to_text() 转换
    with open(os.path.join(data_dir, "cars_specs.json"), "r", encoding="utf-8") as f:
        cars = json.load(f)
    for car in cars:
        documents.append({
            "content": car_to_text(car),
            "source": f"{car['brand']} {car['model']}",
            "type": "car_spec"
        })
    # TODO: ② 遍历 industry_reports/*.txt，逐文件读取 → chunk_sections() 切块
    for report_path in glob.glob(os.path.join(data_dir, "industry_reports", "*.txt")):
        with open(report_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        for chunk in chunk_sections(raw_text):
            documents.append({
                "content": chunk["content"],
                "source": f"{os.path.basename(report_path)}#{chunk['chunk_id']}",
                "type": "industry_report_chunk"
            })

    # TODO: ③ 加载 user_reviews.json，逐条用 review_to_text() 转换
    with open(os.path.join(data_dir, "user_reviews.json"), "r", encoding="utf-8") as f:
        reviews = json.load(f)
    for review in reviews:
        documents.append({
            "content": review_to_text(review),
            "source": review.get("model", ""),
            "type": "user_review"
        })
        
    return documents


# ============================================================
# 3. 检索 —— 关键词匹配
# ============================================================

def search(query: str, documents: list[dict], top_k: int = 3) -> str:
    """
    最简单的关键词匹配检索。

    流程：
      ① jieba 分词 query
      ② 过掉停用词（"的""了""是""在""和""我""有"）
      ③ 对每篇文档：统计有多少个 query 关键词出现在 content 里 → 得分
      ④ 按得分降序，取 top_k
      ⑤ 把 top_k 文档拼成一个上下文字符串

    返回格式：
      【来源：xxx】
      文档内容...
      ---
      【来源：yyy】
      文档内容...

    提示：jieba.cut(query) 返回生成器，用 list() 包一下 / set 去重
    """
    # TODO: 实现
    # ① 把 query 切成关键词
    keywords = list(set(jieba.cut(query)))
    stop_words = {"的", "了", "是", "在", "和", "我", "有"}
    keywords = [w for w in keywords if w not in stop_words]

    # ② 对每篇文档打分：统计多少个关键词出现在 content 里
    scored = []
    for doc in documents:
        content = doc["content"]
        score = 0
        for kw in keywords:
            if kw in content:
                score += 1
        scored.append((score, doc))

    # ③ 按分数降序排列，取 top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for score, doc in scored[:top_k] if score > 0]

    # ④ 把 top_k 文档拼成一段上下文字符串，供 LLM 阅读
    context_parts = []
    for doc in top_docs:
        context_parts.append(f"【来源：{doc['source']}】\n{doc['content']}")

    return "\n\n---\n\n".join(context_parts)


# ============================================================
# 4. Prompt 构造 + LLM 调用
# ============================================================

def build_prompt(query: str, context: str) -> list[dict]:
    """
    把用户问题 + 检索到的上下文拼成 LLM 对话格式。

    System prompt 要点：
      - 角色：专业汽车导购助手
      - 核心约束：严格根据上下文回答，不要编造数据；不确定就告知用户
      - 格式要求：引用具体的数据来源和数字

    User prompt 格式：
      【参考资料】\n{context}\n\n【用户问题】\n{query}\n\n请根据以上参考资料回答用户的问题。

    返回 list of {"role": ..., "content": ...}
    """
    # TODO: 实现
    system_prompt = (
        "你是一个专业的汽车导购助手。请严格根据下面提供的上下文信息来回答用户问题。"
        "如果上下文中没有足够的信息，请明确告知用户，不要编造任何数据。"
        "回答时请引用具体的数据来源和数字。"
    )

    user_prompt = (
        f"【参考资料】\n{context}\n\n"
        f"【用户问题】\n{query}\n\n"
        f"请根据以上参考资料回答用户的问题。"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def ask_llm(messages: list[dict]) -> str:
    """
    调用 LLM，返回回答文本。

    参数：
      model=从环境变量读取的 model id
      messages=build_prompt() 返回的消息列表
      temperature=0  # 稳定输出，不乱编
    """
    # TODO: 实现（用模块级的 _llm_client）
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

# ============================================================
# 5. 交互入口
# ============================================================

if __name__ == "__main__":
    """
    启动后的交互流程：
      ① 确定 data_dir 路径（chapters/../data）
      ② 调用 load_data(data_dir) 加载文档
      ③ 打印加载数量
      ④ while 循环：
          - 读取用户输入
          - q 退出，空输入跳过
          - search(query, documents) 检索
          - 无结果时提示用户
          - build_prompt(query, context) 构造 prompt
          - ask_llm(messages) 获取回答
          - 打印回答
    """
    # TODO: 实现
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")

    # 加载
    documents = load_data(DATA_DIR)
    print(f"[OK] 加载 {len(documents)} 条文档")

    # 交互循环
    while True:
        query = input("\n>> 请输入你的问题（输入 q 退出）：").strip()
        if query.lower() == "q":
            break
        if not query:
            continue

        # RAG 管线：检索 → 组 prompt → 调 LLM
        context = search(query, documents)
        if not context:
            print("[!] 未找到相关信息，请换一种问法。")
            continue

        messages = build_prompt(query, context)
        answer = ask_llm(messages)

        print(f"\n[AI] {answer}")
