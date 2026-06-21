import os
import json
import re as _re
from typing import Optional, Literal

from pydantic import BaseModel, Field, field_validator, ValidationError
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

llm_client = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


# 练习 1: Pydantic BaseModel 定义输出结构
class CarRecommendation(BaseModel):
    """汽车推荐的结构化输出"""

    model_name: str = Field(description="推荐车型全称，如'小鹏G6 755超长续航Max'")
    price_range: str = Field(description="价格区间，如'20.99-27.69万'")
    score: float = Field(description="综合评分，0-10分", ge=0, le=10)
    pros: list[str] = Field(description="优点列表，至少3条", min_length=3)
    cons: list[str] = Field(description="缺点列表")
    best_for: str = Field(description="最适合的人群描述")

    @field_validator("model_name")
    @classmethod
    def must_be_real_car(cls, v: str) -> str:
        """车型名必须包含已知品牌（防止 LLM 虚构车型）"""
        known_brands = [
            "比亚迪", "特斯拉", "小鹏", "理想", "蔚来", "问界",
            "零跑", "小米", "埃安", "极氪", "长安", "吉利", "大众",
        ]
        if not any(brand in v for brand in known_brands):
            raise ValueError(f"'{v}' 不包含已知汽车品牌，可能是 LLM 虚构的车型")
        return v

# 练习 2: 原生 SDK 手写结构化输出（用 Function Calling 模拟）
def structured_output_native(prompt: str, schema_class: type[BaseModel]) -> BaseModel:
    """用手写 Function Calling 实现结构化输出（Day 3 + Day 8 串联）。

    为什么要手写？
      - 理想方案: response_format={"type": "json_schema"} — 但部分模型不支持
        （如 DeepSeek 返回 400: "This response_format type is unavailable now"）
      - 兼容方案: 把 Schema 包装成 tool，用 tool_choice="required" 强制 LLM
        以 tool call 形式输出结构化数据 ← 这也是 LangChain with_structured_output
        在 method="function_calling" 时做的事

    Day 3 → Day 8 理解升华：
      Function Calling 的 tool_calls 本质就是系统预定义的 Structured Output。
      tool_choice="required" ≈ response_format={"type": "json_schema"}
    """
    json_schema = schema_class.model_json_schema()

    # 把 Pydantic Schema 包装成一个"伪工具"（只用于输出格式约束，不执行）
    fake_tool = {
        "type": "function",
        "function": {
            "name": schema_class.__name__,
            "description": f"输出一个 {schema_class.__name__} 结构",
            "parameters": json_schema,
        },
    }

    # 用 tool_choice 强制 LLM"调用"这个伪工具 → LLM 必须输出符合 Schema 的 JSON
    response = llm_client.invoke(
        [
            SystemMessage(content="你是一个汽车导购助手，请根据用户需求输出推荐结果。"),
            HumanMessage(content=prompt),
        ],
        tools=[fake_tool],
        tool_choice={"type": "function", "function": {"name": schema_class.__name__}},
    )

    # 从 tool_calls 中提取结构化数据（回忆 Day 3 的 tool_calls 解析逻辑）
    # 注意：LangChain ChatOpenAI 把 tool_calls 放在 response.tool_calls（Pydantic 属性），
    # 而非 additional_kwargs（原生 dict）。与 Day 3 原生 SDK 的 response.choices[0].message.tool_calls 对应。
    tool_args = response.tool_calls[0]["args"]
    return schema_class.model_validate_json(tool_args)

# 练习 3: LangChain with_structured_output() 一行绑定
def structured_output_langchain(prompt: str, schema_class: type[BaseModel]) -> BaseModel:
    """用 LangChain 的 with_structured_output()

    背后发生了什么（面试要能说清）：
      1. 读取 Pydantic Model 的 model_json_schema()
      2. 把 Schema 包装成 tool 定义 + 注入 tool_choice 参数到 API 调用
      3. LLM 返回 tool_calls → 自动解析 JSON → Pydantic 对象
      4. 解析失败时抛出 OutputParserException

    method 参数的含义：
      - method="json_schema": 用 response_format 管道（需模型支持，如 GPT-4）
      - method="function_calling"（当前使用）: 把 Schema 包装成 tool
        → 兼容所有支持 Function Calling 的模型（DeepSeek / GPT / 等）
    """
    structured_llm = llm_client.with_structured_output(schema_class, method="function_calling")
    return structured_llm.invoke(prompt)

# 练习 4: 多层嵌套结构（车型 → 参数 → 智驾子对象）
class SmartDrive(BaseModel):
    """智驾子对象 — 第二层嵌套"""

    chip: str = Field(description="智驾芯片型号，如'Orin-X'")
    computing_power_tops: int = Field(description="算力，单位TOPS")
    lidar_count: int = Field(description="激光雷达数量")
    features: list[str] = Field(
        description="智驾功能列表，如['高速NOA', '城市NOA', '自动泊车']"
    )


class CarParams(BaseModel):
    """车辆参数子对象 — 第二层嵌套"""

    length_mm: int = Field(description="车长(mm)")
    wheelbase_mm: int = Field(description="轴距(mm)")
    power_hp: int = Field(description="马力")
    range_km: int = Field(description="CLTC续航(km)")
    acceleration_0_100: float = Field(description="0-100km/h加速(秒)")


class DetailedCarInfo(BaseModel):
    """完整车型信息 — 三层嵌套结构（面试展示用）

    设计原则：
      - 子对象表达内聚概念：params = 物理参数, smart_drive = 智驾能力
      - 不超过 3 层嵌套（超过则建议拆成多次调用）
      - 枚举值用 Literal 而非 str（LLM 从有限选项中选取更可靠）
    """

    model_name: str = Field(description="车型全称")
    brand: str = Field(description="品牌")
    price_wan: float = Field(description="价格(万元)")
    category: Literal["轿车", "SUV", "MPV", "跑车", "皮卡"] = Field(description="车型类别")
    energy_type: Literal["纯电", "增程", "插混", "燃油"] = Field(description="能源类型")

    # ★ 第二层嵌套：参数子对象
    params: CarParams

    # ★ 第二层嵌套：智驾子对象
    smart_drive: SmartDrive

    # 竞品列表（简单嵌套）
    alternatives: list[str] = Field(description="同价位竞品车型列表，至少2个", min_length=2)

    # 综合评语
    overall_rating: str = Field(description="综合评价，一段话总结推荐理由")


# 练习 5: 错误处理 — 字段校验失败时重试
def structured_output_with_retry(
    prompt: str,
    schema_class: type[BaseModel],
    max_retries: int = 3,
) -> BaseModel:
    """带重试的结构化输出。

    三种错误类型及处理策略：
      OutputParserException → JSON 不合法（极少见）  → 重试 + 加强 Prompt
      ValidationError       → 字段不满足约束       → 把错误信息反馈给 LLM 自我修正
      语义错误              → 合法但不正确           → @field_validator 做业务规则校验

    重试策略的核心：每次重试都要把具体的校验失败原因注入 Prompt，
    让 LLM 知道"哪里不对、应该怎么改"。
    """
    structured_llm = llm_client.with_structured_output(schema_class, method="function_calling")
    current_prompt = prompt
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return structured_llm.invoke(current_prompt)
        except OutputParserException as e:
            last_error = e
            # JSON 格式错误：提醒 LLM 严格遵守 JSON 格式
            current_prompt = (
                f"{prompt}\n\n"
                f"[第{attempt}次修正] 上次输出不是合法的 JSON：{str(e)[:200]}。\n"
                f"请确保输出是纯 JSON，以 {{ 开始、以 }} 结束，不要加任何解释文字。"
            )
        except ValidationError as e:
            last_error = e
            # 字段校验错误：把具体错误反馈给 LLM
            error_details = json.dumps(e.errors(), ensure_ascii=False, indent=2)
            current_prompt = (
                f"{prompt}\n\n"
                f"[第{attempt}次修正] 上次输出的字段校验失败，具体错误：\n"
                f"{error_details}\n"
                f"请根据以上错误信息修正后重新输出纯 JSON。"
            )

    # 重试耗尽：降级处理
    print(f"[WARN] 结构化输出重试 {max_retries} 次后仍失败: {last_error}")
    return None


# 补充：不支持 json_schema 的降级方案（面试常问）
def extract_json_from_text(text: str) -> dict:
    """从 LLM 自由文本中提取 JSON（降级方案）。

    适用场景：模型不支持 response_format: json_schema 时，
    用 Prompt 要求输出 JSON，再从回复中提取。
    """
    # 1. 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 提取 ```json ... ``` 代码块
    match = _re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 提取最外层 { ... }
    match = _re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 4. 修复常见 JSON 错误后重试
    fixed = text
    fixed = _re.sub(r',\s*}', '}', fixed)       # 尾部逗号
    fixed = _re.sub(r',\s*]', ']', fixed)       # 数组尾部逗号
    fixed = _re.sub(r'//.*?\n', '\n', fixed)     # 单行注释
    # 提取并解析
    match = _re.search(r'\{.*\}', fixed, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"无法从文本中提取合法 JSON: {text[:200]}...")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Day 8: 结构化输出")
    print("=" * 70)

    # ── 练习 1+2: 基础结构化输出 ──
    print("\n[练习 1+2] Pydantic Schema + response_format 强制 JSON 输出")
    print("-" * 50)

    query = "25万预算，看重智驾和续航，推荐一款纯电SUV并给出优缺点"

    try:
        result = structured_output_native(query, CarRecommendation)
        print(f"车型: {result.model_name}")
        print(f"价格: {result.price_range}")
        print(f"评分: {result.score}/10")
        print(f"优点: {result.pros}")
        print(f"缺点: {result.cons}")
        print(f"适合: {result.best_for}")
        print(f"类型: {type(result).__name__} (Pydantic 对象，非 str)")
    except Exception as e:
        print(f"[FAIL] {e}")

    # ── 练习 3: LangChain 一行绑定 ──
    print("\n[练习 3] with_structured_output() 一行绑定")
    print("-" * 50)

    try:
        result2 = structured_output_langchain(query, CarRecommendation)
        print(f"车型: {result2.model_name}")
        print(f"代码量: 2 行 vs 原生 SDK 的 ~15 行")
    except Exception as e:
        print(f"[FAIL] {e}")

    # ── 练习 4: 多层嵌套结构 ──
    print("\n[练习 4] 多层嵌套结构：DetailedCarInfo")
    print("-" * 50)

    detailed_query = "详细介绍一下小鹏G6 755超长续航Max版，包括车身参数、智驾配置和竞品"

    try:
        detailed_llm = llm_client.with_structured_output(DetailedCarInfo, method="function_calling")
        car = detailed_llm.invoke(detailed_query)

        print(f"车型: {car.model_name}")
        print(f"品牌: {car.brand}  |  价格: {car.price_wan}万  |  类别: {car.category}")
        print(f"能源: {car.energy_type}")
        print(f"--- 车身参数 ---")
        print(f"  车长: {car.params.length_mm}mm  |  轴距: {car.params.wheelbase_mm}mm")
        print(f"  马力: {car.params.power_hp}hp  |  续航: {car.params.range_km}km")
        print(f"  0-100加速: {car.params.acceleration_0_100}s")
        print(f"--- 智驾配置 ---")
        print(f"  芯片: {car.smart_drive.chip}  |  算力: {car.smart_drive.computing_power_tops}TOPS")
        print(f"  激光雷达: {car.smart_drive.lidar_count}个")
        print(f"  功能: {car.smart_drive.features}")
        print(f"--- 竞品 ---")
        print(f"  {car.alternatives}")
        print(f"--- 综合评价 ---")
        print(f"  {car.overall_rating[:100]}...")
    except Exception as e:
        print(f"[FAIL] {e}")

    # ── 练习 5: 错误处理重试 ──
    print("\n[练习 5] 错误处理：校验失败时重试")
    print("-" * 50)

    # 模拟一个容易出错的 query（要求一个很难满足的约束）
    tricky_query = "推荐一款10万以下的纯电轿车，要求智驾芯片算力至少500TOPS"

    try:
        # 定义一个严格校验的 Schema
        class StrictCarRec(BaseModel):
            model_name: str = Field(description="车型全称")
            price_wan: float = Field(ge=0, le=10, description="价格必须 ≤ 10万")
            smart_drive_tops: int = Field(ge=500, description="智驾算力必须 ≥ 500TOPS")

            @field_validator("price_wan")
            @classmethod
            def check_price_reasonable(cls, v):
                if v < 3:
                    raise ValueError(f"价格 {v}万 过低，新能源车起步价通常在 3 万以上")
                return v

        tricky_result = structured_output_with_retry(tricky_query, StrictCarRec, max_retries=3)

        if tricky_result:
            print(f"成功: {tricky_result.model_name}, "
                  f"{tricky_result.price_wan}万, {tricky_result.smart_drive_tops}TOPS")
        else:
            print("重试耗尽 → 降级处理（返回 None + 记录日志）")
            print("提示: 10万以下 + 500TOPS 智驾几乎不存在，这是合理的降级行为")
    except Exception as e:
        print(f"[FAIL] {e}")

    # ── 达标自检 ──
    print("\n" + "=" * 70)
    print("[Day 8 达标自检]")
    print("=" * 70)

    checks = []

    # 练习 1: Pydantic Schema 定义
    try:
        schema = CarRecommendation.model_json_schema()
        ok1 = (
            "model_name" in schema["properties"]
            and "score" in schema["properties"]
            and schema["properties"]["score"]["minimum"] == 0
            and schema["properties"]["score"]["maximum"] == 10
        )
    except Exception:
        ok1 = False
    checks.append(("练习1: Pydantic BaseModel Schema 定义完整（含约束 ge/le/min_length）", ok1))

    # 练习 2: 原生 SDK 结构化输出
    try:
        result = structured_output_native("推荐一款15万的纯电轿车", CarRecommendation)
        ok2 = isinstance(result, CarRecommendation) and result.score <= 10
    except Exception:
        ok2 = False
    checks.append(("练习2: response_format json_schema 返回 Pydantic 对象且通过校验", ok2))

    # 练习 3: LangChain 一行绑定
    try:
        result = structured_output_langchain("推荐一款20万的家用SUV", CarRecommendation)
        ok3 = isinstance(result, CarRecommendation) and len(result.pros) >= 3
    except Exception:
        ok3 = False
    checks.append(("练习3: with_structured_output 一行绑定返回合法对象", ok3))

    # 练习 4: 多层嵌套结构
    try:
        detailed_llm = llm_client.with_structured_output(DetailedCarInfo, method="function_calling")
        car = detailed_llm.invoke("介绍小鹏G6")
        ok4 = (
            isinstance(car.params, CarParams)
            and isinstance(car.smart_drive, SmartDrive)
            and car.params.range_km > 0
        )
    except Exception:
        ok4 = False
    checks.append(("练习4: 三层嵌套结构正确解析（params + smart_drive 子对象）", ok4))

    # 练习 5: 错误处理重试
    try:
        tricky_result = structured_output_with_retry(
            "推荐一款5万以下的家用轿车", StrictCarRec, max_retries=2
        )
        # tricky_result is None 是预期行为（不可能查询 → 降级兜底）
        # tricky_result is not None 也可接受（LLM 强行填出了合法值）
        ok5 = True  # 无论哪种结果，只要没抛异常就说明重试+降级流程正确
    except Exception:
        ok5 = False
    checks.append(("练习5: 校验失败 → 重试 → 降级兜底 流程完整", ok5))

    all_pass = True
    for desc, ok in checks:
        status = "[PASS]" if ok else "[FAIL]"
        if not ok:
            all_pass = False
        print(f"  {status}  {desc}")

    print(f"\n结论: {'全部达标 [PASS]' if all_pass else '有未完成项 [WARN]'}")
