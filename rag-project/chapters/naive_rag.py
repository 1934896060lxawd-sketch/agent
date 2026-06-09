import json
import glob
import os
import re
import jieba
from openai import OpenAI
from dotenv import load_dotenv

# ============================================================
# 模块级初始化：只做一次
# ============================================================
load_dotenv()

_llm_client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)
_llm_model = os.getenv("LLM_MODEL_ID")


def car_to_text(car: dict) -> str:
    """把一辆车的结构化字典转成一段自然语言描述"""
    p = car.get("powertrain", {})
    perf = car.get("performance", {})
    dims = car.get("dimensions", {})
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
    """把一条用户评价拍平成文本"""
    pros = "、".join(review.get("pros", []))
    cons = "、".join(review.get("cons", []))
    return (
        f"车型：{review.get('model', '')}，车主：{review.get('owner', '')}，"
        f"评分：{review.get('rating', '')}/5。"
        f"优点：{pros}。缺点：{cons}。"
        f"建议：{review.get('advice', '')}"
    )


def chunk_sections(text: str) -> list[dict]:
    """按「一、」「二、」...标题切块，每块=标题+正文"""
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


def load_data(data: str) -> list[dict]:
    """加载所有数据源，返回格式统一的文档列表"""
    documents = []
    # ① 车型规格
    with open(os.path.join(data, "cars_specs.json"), "r", encoding="utf-8") as f:
        cars = json.load(f)
    for car in cars:
        documents.append({
            "content": car_to_text(car),
            "source": f"{car['brand']} {car['model']}",
            "type": "car_spec"
        })

    # ② 行业报告（分块版）
    for report_path in glob.glob(os.path.join(data, "industry_reports", "*.txt")):
        with open(report_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        for chunk in chunk_sections(raw_text):
            documents.append({
                "content": chunk["content"],
                "source": f"{os.path.basename(report_path)}#{chunk['chunk_id']}",
                "type": "industry_report_chunk"
            })

    # ③ 用户评价
    with open(os.path.join(data, "user_reviews.json"), "r", encoding="utf-8") as f:
        reviews = json.load(f)
    for review in reviews:
        documents.append({
            "content": review_to_text(review),
            "source": review.get("model", ""),
            "type": "user_review"
        })

    return documents


def search(query: str, documents: list[dict], top_k: int = 3) -> str:
    """
    最简单的关键词匹配检索。
    对 query 做分词 → 在每篇文档的 content 中统计命中次数 → 取 top_k 拼成上下文
    """
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


def build_prompt(query: str, context: str) -> list[dict]:
    """
    把用户问题 + 检索到的上下文拼成 LLM 对话格式。
    返回 messages 列表，可直接喂给 openai chat.completions.create()
    """
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
    """调用 LLM，返回回答文本"""
    response = _llm_client.chat.completions.create(
        model=_llm_model,
        messages=messages,
        temperature=0,   # 稳定输出，不乱编
    )
    return response.choices[0].message.content


if __name__ == "__main__":
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
