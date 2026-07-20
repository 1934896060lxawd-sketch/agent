"""Agent 工具集 — 5 个 @tool 函数，供 ReAct Agent 调用。

工具注册方式: OpenAI function-calling 格式的 JSON Schema。
执行器: ToolExecutor 通过 getattr 反射将工具名分发到 _tool_xxx 方法。
"""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

from backend.rag.chunker import Document
from backend.rag.retriever import hybrid_rrf
from backend.rag.reranker import rerank

if TYPE_CHECKING:
    from backend.rag.retriever import VectorIndex, BM25

logger = logging.getLogger(__name__)

# 车型价格数据库（模拟，Phase 4 可替换为真实 API）
CAR_PRICE_DB: dict[str, str] = {
    "比亚迪 秦PLUS DM-i": "9.98-14.58 万",
    "比亚迪 海豚": "9.98-13.98 万",
    "比亚迪 海豹": "17.98-24.98 万",
    "比亚迪 宋PLUS DM-i": "15.98-20.98 万",
    "特斯拉 Model 3": "23.19-33.59 万",
    "特斯拉 Model Y": "24.99-35.49 万",
    "蔚来 ET5": "29.80-35.60 万",
    "蔚来 ES6": "33.80-39.60 万",
    "小鹏 G6": "20.99-27.69 万",
    "小鹏 P7": "22.99-33.99 万",
    "理想 L6": "24.98-27.98 万",
    "理想 L7": "30.98-37.98 万",
    "小米 SU7": "21.59-29.99 万",
    "极氪 001": "26.90-32.90 万",
    "问界 M7": "24.98-32.98 万",
    "吉利 银河 L7": "13.87-17.37 万",
    # ── 与知识库 vehicles.json 对齐的补充车型 ──
    "比亚迪 宋L": "18.98-24.98 万",
    "比亚迪 海豹08": "25-32 万",
    "比亚迪 汉L": "20.98-27.98 万",
    "比亚迪 海豹06 DM-i": "9.98-14.98 万",
    "小鹏 P7i": "22.39-28.99 万",
    "小鹏 G9": "26.39-35.99 万",
    "理想 L9": "42.98-45.98 万",
    "极氪 007": "20.99-29.99 万",
    "吉利 银河E8": "17.58-22.88 万",
}

# 车型简要规格（关键词检索用）
CAR_SPEC_BRIEF: dict[str, str] = {
    "小米 SU7": "CLTC续航 700-830km，零百加速 2.78-5.28s，智驾芯片 Orin-X，含激光雷达",
    "比亚迪 海豚": "CLTC续航 301-405km，零百加速 ~10s，L2 基础智驾",
    "比亚迪 海豹": "CLTC续航 550-700km，零百加速 3.8s，DiPilot 智驾",
    "特斯拉 Model Y": "CLTC续航 545-688km，零百加速 3.7-5.0s，FSD 智驾",
    "特斯拉 Model 3": "CLTC续航 556-713km，零百加速 3.3-6.1s，FSD 智驾",
    "理想 L6": "增程，CLTC综合续航 1390km，零百加速 5.4s，AD Max 智驾",
    "理想 L7": "增程，CLTC综合续航 1315km，零百加速 5.3s，AD Max 智驾",
    "问界 M7": "增程/纯电，CLTC续航 1200km，零百加速 4.8s，ADS 2.0 智驾",
    "小鹏 G6": "CLTC续航 580-755km，零百加速 3.9-6.2s，XNGP 智驾，双激光雷达",
    "蔚来 ET5": "CLTC续航 560-710km，零百加速 4.0s，NAD 智驾，含激光雷达",
    "极氪 001": "CLTC续航 546-741km，零百加速 3.8s，NZP 智驾",
    "比亚迪 秦PLUS DM-i": "插混，纯电续航 120km，综合续航 1245km，零百加速 7.3s",
    "比亚迪 宋PLUS DM-i": "插混，纯电续航 150km，综合续航 1200km，零百加速 7.9s",
    "蔚来 ES6": "CLTC续航 490-625km，零百加速 4.5s，NAD 智驾，含激光雷达",
    "小鹏 P7": "CLTC续航 550-702km，零百加速 3.9-6.2s，XNGP 智驾",
    "吉利 银河 L7": "插混，综合续航 1370km，零百加速 6.9s，L2 智驾",
    # ── 与知识库 vehicles.json 对齐的补充车型 ──
    "比亚迪 宋L": "中型纯电SUV（猎装），CLTC续航 662km，零百加速 6.8s，800V快充30%-80%需25分钟，DiPilot L2智驾",
    "比亚迪 海豹08": "中大型纯电轿车，CLTC续航 720km，零百加速 3.8s，兆瓦级闪充5分钟补能400km",
    "比亚迪 汉L": "中大型轿车，纯电/插混，CLTC续航 715km，零百加速 3.9s，800V高压快充",
    "比亚迪 海豹06 DM-i": "中型插混轿车，DM-i混动，零百加速 7.2s，40kW直流快充",
    "小鹏 P7i": "中型纯电轿车，CLTC续航 702km，零百加速 3.9s，800V SiC快充，XNGP智驾",
    "小鹏 G9": "中大型纯电SUV，CLTC续航 702km，零百加速 3.9s，800V SiC快充，XNGP智驾",
    "理想 L9": "全尺寸增程SUV，六座，零百加速 5.3s，75kW直流快充，AD Max智驾",
    "极氪 007": "中型纯电轿车，CLTC续航 870km，零百加速 2.84s，800V快充15分钟补能610km",
    "吉利 银河E8": "中大型纯电轿车，CLTC续航 665km，零百加速 6.5s，400V快充",
}

# 车型类别映射（recommend_cars 的 category 过滤用）
# 历史教训：车型名本身不含"SUV/轿车"字样，直接子串匹配会把所有车过滤光，
# 导致"25万预算推荐SUV"返回 0 款、模型只能回答"没有符合要求的车型"。
CAR_CATEGORY: dict[str, str] = {
    "比亚迪 秦PLUS DM-i": "轿车",
    "比亚迪 海豚": "轿车",
    "比亚迪 海豹": "轿车",
    "比亚迪 宋PLUS DM-i": "SUV",
    "特斯拉 Model 3": "轿车",
    "特斯拉 Model Y": "SUV",
    "蔚来 ET5": "轿车",
    "蔚来 ES6": "SUV",
    "小鹏 G6": "SUV",
    "小鹏 P7": "轿车",
    "理想 L6": "SUV",
    "理想 L7": "SUV",
    "小米 SU7": "轿车",
    "极氪 001": "轿车",
    "问界 M7": "SUV",
    "吉利 银河 L7": "SUV",
    "比亚迪 宋L": "SUV",
    "比亚迪 海豹08": "轿车",
    "比亚迪 汉L": "轿车",
    "比亚迪 海豹06 DM-i": "轿车",
    "小鹏 P7i": "轿车",
    "小鹏 G9": "SUV",
    "理想 L9": "SUV",
    "极氪 007": "轿车",
    "吉利 银河E8": "轿车",
}


# ============================================================
# 工具 Schema 定义（OpenAI function-calling 格式）
# ============================================================
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_car_knowledge",
            "description": "搜索汽车知识库，获取车型的详细参数、续航、智驾、空间、车主评价等信息。当用户询问车型的具体配置、性能参数、或是需要了解某款车时调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户查询的问题，如'小米SU7的续航和智驾配置'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_car_price",
            "description": "查询指定车型的最新市场指导价。参数 brand(品牌)+model(车型) 或直接用 model_name(全名)。当用户询问价格、预算、多少钱时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "brand": {"type": "string", "description": "品牌，如'比亚迪'、'特斯拉'"},
                    "model": {"type": "string", "description": "车型名，如'海豚'、'Model Y'"},
                    "model_name": {"type": "string", "description": "车型全名，如'比亚迪 海豚'、'特斯拉 Model Y'（与brand+model二选一）"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_cars",
            "description": "对比两款车的核心参数（价格、续航、加速、智驾）。当用户要求对比两款车、或问'A和B哪个好'时调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "car1": {"type": "string", "description": "第一款车全名，如'小米 SU7'"},
                    "car2": {"type": "string", "description": "第二款车全名，如'特斯拉 Model 3'"},
                },
                "required": ["car1", "car2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_cars",
            "description": "根据预算和偏好推荐车型。当用户询问'XX万买什么车'、'推荐一款'时调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "budget_min": {"type": "number", "description": "预算下限（万元）。用户只给一个预算数时传 0"},
                    "budget_max": {"type": "number", "description": "预算上限（万元）。如'25万预算'传 25"},
                    "category": {
                        "type": "string",
                        "description": "车型类别，如SUV/轿车/MPV，默认全部",
                    },
                    "preferred_brand": {
                        "type": "string",
                        "description": "偏好品牌，如比亚迪/特斯拉，默认不限",
                    },
                },
                "required": ["budget_min", "budget_max"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_ownership_cost",
            "description": "计算购车落地价和年均用车成本。当用户询问'落地多少钱'、'养车贵不贵'、'一年花多少钱'时调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "车型全名，如'小米 SU7'"},
                    "years": {
                        "type": "integer",
                        "description": "用车年限，默认3年",
                    },
                },
                "required": ["model"],
            },
        },
    },
]


# ============================================================
# 工具执行器（由 Advisor 注入 retriever/reranker 后调用）
# ============================================================
class ToolExecutor:
    """工具执行器 — 持有检索器引用，通过 getattr 反射分发工具调用。"""

    def __init__(self):
        self.retriever: VectorIndex | None = None
        self.bm25: BM25 | None = None

    def set_retrievers(self, retriever: VectorIndex, bm25: BM25) -> None:
        """注入检索器实例（由 get_agent() 在启动时调用）。"""
        self.retriever = retriever
        self.bm25 = bm25

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """根据工具名反射分发到 _tool_xxx 方法。"""
        method = getattr(self, f"_tool_{name}", None)
        if method is None:
            return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
        try:
            result = method(**arguments)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"工具执行失败 {name}({arguments}): {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    # ── 知识库检索 ──
    def _tool_search_car_knowledge(self, query: str = "", keyword: str = "") -> str:
        """RAG 全管线。接受 query 或 keyword 参数（LLM 可能用任一名称调用）。"""
        search_text = query or keyword or ""
        if self.retriever is None or self.bm25 is None:
            return json.dumps({"error": "知识库索引未就绪"}, ensure_ascii=False)

        dense = self.retriever.search(search_text, top_k=6)
        sparse = self.bm25.search(search_text, top_k=6)
        hybrid = hybrid_rrf(dense, sparse, k=60, top_k=5)
        reranked = rerank(search_text, hybrid, top_k=3)

        results = []
        for i, (score, doc) in enumerate(reranked, 1):
            results.append({
                "rank": i,
                "source": doc.source,
                "content": doc.content[:300],
                "score": round(score, 4),
                "type": doc.doc_type,
            })
        return json.dumps({"results": results, "count": len(results)}, ensure_ascii=False)

    # ── 车型价格查询 ──
    @staticmethod
    def _tool_get_car_price(brand: str = "", model: str = "",
                            model_name: str = "") -> str:
        """接受两种参数格式：brand+model 或 model_name 全名。"""
        # model_name 优先：按已知品牌拆分匹配
        if model_name and not (brand and model):
            if model_name in CAR_PRICE_DB:
                return json.dumps({"car": model_name, "price": CAR_PRICE_DB[model_name],
                                   "status": "found"}, ensure_ascii=False)
            # 用 model_name 作为子串搜索
            for k, v in CAR_PRICE_DB.items():
                # "比亚迪宋L" vs "比亚迪 宋PLUS DM-i" → 比亚迪 和 宋 都匹配
                if model_name.replace(" ", "") in k.replace(" ", "") or \
                   any(word in k for word in model_name.split() if len(word) >= 2):
                    return json.dumps({"car": k, "price": v, "status": "found"},
                                      ensure_ascii=False)
            # 反向：DB key 是 model_name 的子串
            for k, v in CAR_PRICE_DB.items():
                if k.replace(" ", "") in model_name.replace(" ", ""):
                    return json.dumps({"car": k, "price": v, "status": "found"},
                                      ensure_ascii=False)

        key = f"{brand} {model}".strip()
        if key in CAR_PRICE_DB:
            return json.dumps({"car": key, "price": CAR_PRICE_DB[key], "status": "found"},
                              ensure_ascii=False)

        # 模糊匹配：品牌和车型名各自至少 2 个字符
        if (brand and len(brand) >= 2) or (model and len(model) >= 2):
            for k, v in CAR_PRICE_DB.items():
                if (not brand or brand in k) and (not model or model in k):
                    return json.dumps({"car": k, "price": v, "status": "found"}, ensure_ascii=False)

        return json.dumps({"car": key or model_name, "price": None, "status": "not_found",
                           "message": f"未找到价格信息"}, ensure_ascii=False)

    # ── 车型名模糊解析 ──
    @staticmethod
    def _resolve_car_name(name: str) -> str | None:
        """把模型传入的车型名（空格/大小写/简称差异）映射到 CAR_PRICE_DB 的 key。

        历史教训：LLM 常传 "小米SU7"（无空格）而 DB key 是 "小米 SU7"，
        精确 .get() 查不到 → 返回"暂无"→ 模型告诉用户"没有数据"（其实是有的）。
        """
        if not name:
            return None
        if name in CAR_PRICE_DB:
            return name
        norm = name.replace(" ", "").lower()
        # 1) 去空格、忽略大小写后完全相等
        for k in CAR_PRICE_DB:
            if k.replace(" ", "").lower() == norm:
                return k
        # 2) 子串互相包含（如 "Model 3" → "特斯拉 Model 3"）
        for k in CAR_PRICE_DB:
            kk = k.replace(" ", "").lower()
            if len(norm) >= 2 and (norm in kk or kk in norm):
                return k
        # 3) 分词全部命中（如 "小米 SU7" → 小米、SU7 都在 key 中）
        words = [w for w in name.split() if len(w) >= 2]
        if words:
            for k in CAR_PRICE_DB:
                if all(w in k for w in words):
                    return k
        return None

    # ── 车型对比 ──
    @staticmethod
    def _tool_compare_cars(car1: str, car2: str) -> str:
        k1 = ToolExecutor._resolve_car_name(car1)
        k2 = ToolExecutor._resolve_car_name(car2)
        return json.dumps({
            "car1": {"name": k1 or car1,
                     "price": CAR_PRICE_DB.get(k1 or "", "暂无"),
                     "spec": CAR_SPEC_BRIEF.get(k1 or "", "暂无参数"),
                     "matched": k1 is not None},
            "car2": {"name": k2 or car2,
                     "price": CAR_PRICE_DB.get(k2 or "", "暂无"),
                     "spec": CAR_SPEC_BRIEF.get(k2 or "", "暂无参数"),
                     "matched": k2 is not None},
        }, ensure_ascii=False)

    # ── 车型推荐 ──
    @staticmethod
    def _tool_recommend_cars(budget_min: float, budget_max: float,
                             category: str = "全部", preferred_brand: str = "") -> str:
        # 健壮性①：用户只给一个预算数（"25万预算"）时，模型常传 min==max，
        # 精确区间几乎不可能命中任何车。按"预算上限"理解，下限放宽到 0。
        if budget_min >= budget_max:
            budget_min, budget_max = 0.0, max(budget_min, budget_max)

        # 健壮性②：类别归一化，方言/大小写/近义词都映射到标准类别
        cat = category.strip() if category else "全部"
        if any(k in cat for k in ("SUV", "suv", "越野")):
            cat = "SUV"
        elif any(k in cat for k in ("轿", "三厢", "两厢")):
            cat = "轿车"
        elif any(k in cat for k in ("MPV", "mpv", "商务")):
            cat = "MPV"
        else:
            cat = "全部"

        results = []
        for name, price_str in CAR_PRICE_DB.items():
            try:
                parts = price_str.replace(" 万", "").split("-")
                low_price = float(parts[0])
                high_price = float(parts[-1])
            except (ValueError, IndexError):
                continue
            # 价格区间与预算区间有重叠即视为预算内
            # （起步价不超预算上限，顶配不低于预算下限）
            if low_price > budget_max or high_price < budget_min:
                continue
            if cat != "全部" and CAR_CATEGORY.get(name, "") != cat:
                continue
            if preferred_brand and preferred_brand not in name:
                continue
            results.append({"name": name, "price": price_str,
                            "spec": CAR_SPEC_BRIEF.get(name, "暂无参数")})
        # 起步价从高到低排序：贴近预算上限的车型优先（25万预算先看到24.99万的
        # Model Y，而不是13万的入门车），符合导购"贴着预算推荐"的习惯
        results.sort(key=lambda x: float(x["price"].split("-")[0]), reverse=True)
        return json.dumps({"count": len(results), "cars": results}, ensure_ascii=False)

    # ── 用车成本计算 ──
    @staticmethod
    def _tool_calculate_ownership_cost(model: str, years: int = 3) -> str:
        key = ToolExecutor._resolve_car_name(model)
        price_str = CAR_PRICE_DB.get(key) if key else None
        if not price_str:
            return json.dumps({"error": f"未找到 {model} 的价格"}, ensure_ascii=False)
        parts = price_str.replace(" 万", "").split("-")
        mid_price = (float(parts[0]) + float(parts[-1])) / 2
        insurance = round(mid_price * 0.03, 2)    # 年均保险 ≈ 车价 × 3%
        maintenance = round(mid_price * 0.01, 2)  # 年均保养 ≈ 车价 × 1%
        energy = round(0.3 * 20000 / 10000, 2)    # 电费 0.3 元/km × 年 2 万 km → 万元
        annual = round(insurance + maintenance + energy, 2)
        total = round(mid_price + annual * years, 2)
        return json.dumps({
            "model": key,
            "mid_price_wan": round(mid_price, 2),
            "annual_cost_wan": annual,
            "years": years,
            "total_cost_wan": total,
        }, ensure_ascii=False)
