# ================================================================
# Day 9：图片识别 + RAG（Vision RAG）
# 核心：图片 → Base64 → 多模态模型识别 → 文本描述 → Embedding → 检索 → 回答
# 关键洞察：图片不直接进向量库，而是先转文本再走标准 RAG 管道（最小侵入性）
# 面试题：见 agent/day9_interview_qa.md
# ================================================================

import base64
import io
import json
import os
import sys
from typing import Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

# ---- 路径 & 环境 ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "chapters"))  # retrieval_test / naive_rag
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ---- 模型配置（智谱 GLM-4V） ----
VISION_MODEL = "glm-4v"         # 多模态模型，支持图片输入
TEXT_MODEL = "glm-4-flash"      # 文本模型
API_KEY = os.getenv("ZAI_API_KEY")
BASE_URL = os.getenv("ZAI_BASE_URL")


# ================================================================
# 练习 1：Base64 图片编码 + 多模态消息格式
# ================================================================

def encode_image(image_path: str) -> Tuple[str, str]:
    """将图片编码为 (base64字符串, MIME类型)。Base64 增大约 33%，生产需先压缩。"""
    with open(image_path, "rb") as f:
        raw_bytes = f.read()
    b64_str = base64.b64encode(raw_bytes).decode("utf-8")
    ext = image_path.lower().rsplit(".", 1)[-1]
    mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}
    return b64_str, mime_map.get(ext, "jpeg")


def preprocess_image_for_vision(image_path: str, max_size: int = 1024, quality: int = 85) -> str:
    """缩小+压缩图片以减少 tile 消耗。4000×3000 ~2800tokens → 1024×768 ~765tokens。"""
    try:
        from PIL import Image
        img = Image.open(image_path)
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        print(f"  图片预处理: {os.path.getsize(image_path)/1024:.0f}KB → {buffer.tell()/1024:.0f}KB")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except ImportError:
        print("  [!] Pillow 未安装，跳过图片预处理")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def build_multimodal_message(prompt: str, image_path: str) -> HumanMessage:
    """构造多模态消息：content 从 str 变为 content block 数组（文本+图片混合）。"""
    b64_str, mime_type = encode_image(image_path)
    return HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/{mime_type};base64,{b64_str}"}},
    ])


# ================================================================
# 练习 2：图片描述 → Embedding → 向量检索
# ================================================================

def describe_car_image(image_path: str, vision_llm: ChatOpenAI) -> str:
    """多模态模型生成结构化图片描述。描述质量 = 检索召回率上限。"""
    msg = build_multimodal_message(
        "请按以下结构描述这张汽车图片：\n"
        "1. 【品牌】识别车标、尾标，不确定则列可能品牌\n"
        "2. 【车型】识别具体型号（如G6、SU7、Model Y）\n"
        "3. 【类型】轿车/SUV/MPV/跑车/皮卡\n"
        "4. 【颜色】车身颜色\n"
        "5. 【外观特征】灯组、格栅、轮毂、门把手、腰线（5-8条）\n"
        "6. 【置信度】品牌和型号识别置信度（高/中/低）\n"
        "直接输出描述，不附加解释。",
        image_path,
    )
    response = vision_llm.invoke([msg])
    return response.content


def extract_brand_model(description: str, llm: ChatOpenAI) -> str:
    """从描述提取'品牌 型号'，用于 BM25 关键词检索，提升召回率。"""
    return llm.invoke(
        f"从以下描述提取汽车品牌和型号，只输出'品牌 型号'格式（如'小鹏 G6'）：\n\n{description}"
    ).content.strip()


# ================================================================
# 练习 3：端到端 Vision RAG 管线
# ================================================================

def vision_rag_pipeline(
    image_path: str,
    user_question: str,
    vector_index,
    bm25_idx,
    reranker,
    vision_llm: ChatOpenAI,
    text_llm: ChatOpenAI,
    encoder: "SentenceTransformer" = None,
) -> dict:
    """端到端 Vision RAG：图片→识别→混合检索→Reranker→LLM 生成。
    检索+生成全链路复用纯文本 RAG，差异仅在检索入口（image→text→Embedding）。"""

    # ① 多模态识别
    car_description = describe_car_image(image_path, vision_llm)

    # ② 提取品牌型号 → 关键词检索
    brand_model = extract_brand_model(car_description, text_llm)

    # ③ 混合检索
    search_query = f"{car_description}\n{user_question}"
    dense_results = vector_index.search(search_query, top_k=10)
    sparse_results = bm25_idx.search(brand_model, top_k=10)

    from retrieval_test import hybrid_rrf
    hybrid = hybrid_rrf(dense_results, sparse_results, k=60, top_k=6)

    # ④ Reranker 精排
    if reranker and reranker.model:
        hybrid = reranker.rerank(search_query, hybrid, top_k=3)
    else:
        hybrid = hybrid[:3]

    # ⑤ 组装 context + LLM 生成
    context_parts = []
    for i, (score, doc) in enumerate(hybrid, 1):
        content = doc.get("content", "")[:300]
        source = doc.get("source", "未知")
        context_parts.append(f"[{i}] (来源: {source}) {content}")
    context = "\n\n".join(context_parts)

    final_prompt = (
        f"## 图片识别结果\n{car_description}\n\n"
        f"## 知识库参考资料\n{context}\n\n"
        f"## 用户问题\n{user_question}\n\n"
        f"请根据图片识别结果和参考资料回答。必须引用具体数据，标注来源。"
    )
    answer = text_llm.invoke(final_prompt).content

    return {
        "car_description": car_description,
        "extracted_entity": brand_model,
        "retrieved_docs": [(score, doc.get("source", "")) for score, doc in hybrid],
        "answer": answer,
    }


# ================================================================
# 练习 5：Vision 作为 Agent Tool
# ================================================================

@tool
def analyze_car_image(image_path: str, analysis_type: str = "identify") -> str:
    """分析汽车图片：identify(车型识别) / dashboard(故障灯) / damage(损伤评估)。
    Vision 能力通过 Tool 融入 Agent 的 ReAct 循环，是 Day 3+5+9 协同的关键。"""
    vision_llm = ChatOpenAI(
        model=VISION_MODEL, api_key=API_KEY, base_url=BASE_URL, temperature=0,
    )

    prompts = {
        "identify": "请识别这辆车：1)品牌 2)型号 3)类型 4)颜色 5)外观特征(5条) 6)置信度。直接输出。",
        "dashboard": "请识别仪表盘所有故障灯，解释含义和严重程度，标注需立即停车的项目。",
        "damage": "评估事故损伤：1)受损部位 2)程度(轻微/中等/严重) 3)是否影响行驶 4)估算维修费。",
    }
    msg = build_multimodal_message(prompts.get(analysis_type, prompts["identify"]), image_path)
    response = vision_llm.invoke([msg])
    return json.dumps({"analysis_type": analysis_type, "result": response.content}, ensure_ascii=False)


# ================================================================
# LLM 工厂
# ================================================================

def get_vision_llm():
    """智谱 GLM-4V 多模态模型 — 支持图片+文本混合输入"""
    return ChatOpenAI(model=VISION_MODEL, api_key=API_KEY, base_url=BASE_URL, temperature=0)


def get_text_llm():
    """智谱 GLM-4-Flash 文本模型 — 纯文本任务"""
    return ChatOpenAI(model=TEXT_MODEL, api_key=API_KEY, base_url=BASE_URL, temperature=0)


# ================================================================
# 运行入口
# ================================================================

if __name__ == "__main__":
    # 自动探测默认图片（优先 images/su7.jpg）
    image_path = None
    for name in ["images/su7.jpg", "car.jpg", "car.png"]:
        p = os.path.join(BASE_DIR, name)
        if os.path.exists(p):
            image_path = p
            break

    if image_path is None:
        print("未找到测试图片。请放置 images/su7.jpg 或传 --image 参数。")
        print("测试纯文本流程...")
        # 即使无图片也演示消息格式
        print("\n纯文本消息: HumanMessage(content='这是什么车？')")
        print("content 类型: str")

        print("\n多模态消息: HumanMessage(content=[")
        print('    {"type": "text", "text": "请识别..."},')
        print('    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}')
        print("  ])")
        print("content 类型: list[dict]（content block 数组）")
    else:
        print(f"测试图片: {image_path}")
        vision_llm = get_vision_llm()

        # 练习 1：多模态识别
        print("\n[练习1] 多模态识别...")
        msg = build_multimodal_message("请识别图片中的汽车品牌和型号。直接输出事实。", image_path)
        response = vision_llm.invoke([msg])
        print(f"结果: {response.content}")

        # 练习 2：描述 → 提取品牌型号
        print("\n[练习2] 结构化描述 → 提取品牌型号...")
        car_desc = describe_car_image(image_path, vision_llm)
        print(f"描述: {car_desc[:200]}")

        text_llm = get_text_llm()
        brand_model = extract_brand_model(car_desc, text_llm)
        print(f"品牌型号: {brand_model}")
