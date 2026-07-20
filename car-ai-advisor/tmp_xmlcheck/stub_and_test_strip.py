"""对抗性测试 _strip_all_xml / _strip_xml / XML 检测与提取的一致性。

无需 torch：注入 sentence_transformers 桩模块后再 import backend。
"""
import sys
import types

# ── 桩：sentence_transformers（避免安装 torch）──
_stub = types.ModuleType("sentence_transformers")


class _Boom:
    def __init__(self, *a, **k):
        raise RuntimeError("stub: sentence_transformers unavailable")


_stub.SentenceTransformer = _Boom
_stub.CrossEncoder = _Boom
sys.modules["sentence_transformers"] = _stub

from backend.agent.advisor import (  # noqa: E402
    _strip_all_xml, _has_any_xml_or_markup, _extract_xml_tool_calls,
)
from backend.api.routes.chat import _strip_xml  # noqa: E402

CASES = [
    # (用例名, 输入, 期望: 输出中不应出现任何标记/参数残留)
    ("标准invoke块",
     '好的，<invoke name="get_car_price"><parameter name="brand">比亚迪</parameter></invoke>为您查询价格'),
    ("hy-前缀完整块",
     '<hy-tool_calls><hy-invoke name="get_car_price"><hy-parameter name="model_name" string="true">比亚迪 海豚</hy-parameter></hy-invoke></hy-tool_calls>结论如下'),
    ("单引号属性",
     "<invoke name='get_car_price'><parameter name='brand'>比亚迪</parameter></invoke>为您查询"),
    ("属性顺序调换",
     '<invoke id="1" name="get_car_price"><parameter name="brand">比亚迪</parameter></invoke>为您查询'),
    ("大小写混合",
     '<Invoke Name="get_car_price"><Parameter Name="brand">比亚迪</Parameter></Invoke>为您查询'),
    ("未闭合-无右尖括号(截断)",
     '为您对比两款车 <invoke name="compare_cars"'),
    ("未闭合-有左标签无闭合",
     '<invoke name="compare_cars"><parameter name="car1">小米 SU7'),
    ("嵌套invoke",
     '<invoke name="compare_cars"><invoke name="get_car_price"><parameter name="brand">蔚来</parameter></invoke></invoke>结果'),
    ("DSML完整块",
     '<|DSML|function_calls>get_car_price(brand=比亚迪)</|DSML|function_calls>为您查询'),
    ("DSML单标签无配对",
     '查询中 <|DSML|function_calls> 请稍等'),
    ("DSML半截",
     '查询中 <|DSML|invoke name="get_car_price">'),
    ("tool_calls包装但内层非invoke",
     '<tool_calls>{"name": "get_car_price", "args": {"brand": "比亚迪"}}</tool_calls>结果如下'),
    ("function_calls通用格式",
     '<function_calls><invoke name="recommend_cars"><parameter name="budget_max">20</parameter></invoke></function_calls>推荐如下'),
    ("合法代码块展示XML(误杀检查)",
     '工具调用格式示例：\n```xml\n<invoke name="get_car_price"></invoke>\n```\n以上就是格式'),
    ("尖括号+空格",
     '< invoke name="get_car_price"> < /invoke> 为您查询'),
    ("参数带命名空间",
     '<invoke name="tools.get_car_price"><parameter name="brand">比亚迪</parameter></invoke>查询中'),
    ("回答内嵌工具JSON(无标签)",
     '我将调用 {"name": "get_car_price", "arguments": {"brand": "比亚迪"}} 进行查询'),
    # ── 线上实测变体（2026-07-19 真实泄露样本）──
    ("全角双管道完整块",
     '<｜｜DSML｜｜tool_calls>\n<｜｜DSML｜｜invoke name="compare_cars">\n<｜｜DSML｜｜parameter name="car1" string="true">比亚迪海豚</｜｜DSML｜｜parameter>\n<｜｜DSML｜｜parameter name="car2" string="true">特斯拉 Model 3</｜｜DSML｜｜parameter>\n</｜｜DSML｜｜invoke>\n</｜｜DSML｜｜tool_calls>\n以上是对比结果'),
    ("全角单管道标签",
     '为您查询<｜DSML｜invoke name="get_car_price"><｜DSML｜parameter name="model_name">比亚迪 海豚</｜DSML｜parameter></｜DSML｜invoke>'),
    ("全角DSML裸标记",
     '查询中 <｜｜DSML｜｜tool_calls> 请稍等'),
    ("全角截断尾标签",
     '为您对比两款车 <｜｜DSML｜｜parameter name="car2"'),
    ("裸DSML截断",
     '查询中 <｜｜DSML'),
    ("ASCII_DSML完整块(修复验证)",
     '<|DSML|function_calls>\n[{"name": "recommend_cars", "arguments": {"budget_min": 13}}]\n</|DSML|function_calls>\n为您推荐'),
    ("真实泄露样本(probe1)",
     '用户问"海豚和元UP怎么选"，需要对比这两款车型。先查两款车的详细参数和价格。\n\n<｜｜DSML｜｜tool_calls>\n<｜｜DSML｜｜invoke name="compare_cars">\n<｜｜DSML｜｜parameter name="car1" string="true">比亚迪海豚</｜｜DSML｜｜parameter>\n<｜｜DSML｜｜parameter name="car2"'),
    ("markdown表格不受影响",
     '| 车型 | 价格 |\n|:--|:--|\n| 海豚 | 9.98万 |'),
]

LEAK_PAT = __import__("re").compile(
    r'<\s*/?\s*[\w-]*(?:invoke|parameter|tool[\w-]*calls?|function[\w-]*calls?)'
    r'|<\||\|>|</\|'
)


def verdict(out: str) -> str:
    """判断清理结果：CLEAN=纯净, TAG=仍有标签, RESID=标签没了但残留参数文本。"""
    if LEAK_PAT.search(out):
        return "TAG-LEAK"
    # 标签被剥掉但参数文本（如车型名）孤立残留也算轻度问题，仅记录
    return "CLEAN"


print(f"{'用例':<26} {'检测':<4} {'提取':<2} {'advisor结果':<10} {'route结果':<10} advisor输出")
print("-" * 110)
for name, text in CASES:
    detected = _has_any_xml_or_markup(text)
    extracted = len(_extract_xml_tool_calls(text))
    out_a = _strip_all_xml(text)
    out_r = _strip_xml(text)
    va, vr = verdict(out_a), verdict(out_r)
    print(f"{name:<26} {str(detected):<5} {extracted:<3} {va:<10} {vr:<10} {out_a[:60]!r}")
