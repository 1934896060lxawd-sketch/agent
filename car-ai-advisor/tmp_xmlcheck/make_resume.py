# -*- coding: utf-8 -*-
"""罗小安 - AI Agent开发工程师 简历PDF（简约单页）"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, HRFlowable)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "罗小安-AI-Agent开发工程师-简历.pdf")
PHOTO = os.path.join(ROOT, "个人相片.jpg")

# ── 字体 ──
pdfmetrics.registerFont(TTFont('CJK', os.environ['DAIMON_CJK_FONT_REGULAR']))
pdfmetrics.registerFont(TTFont('CJK-B', os.environ['DAIMON_CJK_FONT_BOLD']))
pdfmetrics.registerFontFamily('CJK', normal='CJK', bold='CJK-B',
                              italic='CJK', boldItalic='CJK-B')

INK = HexColor('#1a1a1a')      # 正文近黑
GRAY = HexColor('#555555')     # 次要信息
RULE = HexColor('#2b2b2b')     # 标题线
LIGHT = HexColor('#8a8a8a')

def st(name, **kw):
    base = dict(fontName='CJK', fontSize=10, leading=15, textColor=INK,
                alignment=TA_LEFT)
    base.update(kw)
    return ParagraphStyle(name, **base)

S = {
    'name':   st('name', fontName='CJK-B', fontSize=23, leading=27),
    'intent': st('intent', fontSize=11.5, leading=16, textColor=INK),
    'meta':   st('meta', fontSize=9.5, leading=14, textColor=GRAY),
    'sec':    st('sec', fontName='CJK-B', fontSize=12.5, leading=16,
                 textColor=INK, spaceBefore=2),
    'body':   st('body', fontSize=10, leading=15.5),
    'small':  st('small', fontSize=9.5, leading=14, textColor=GRAY),
    'bullet': st('bullet', fontSize=10, leading=15.5, leftIndent=10,
                 firstLineIndent=-10),
}

def section(title):
    return [
        Paragraph(title, S['sec']),
        HRFlowable(width='100%', thickness=1, color=RULE,
                   spaceBefore=2, spaceAfter=5),
    ]

story = []

# ═══ 头部：左 姓名+意向+联系方式 | 右 照片 ═══
left = [
    Paragraph('罗小安', S['name']),
    Spacer(1, 3),
    Paragraph('<b>求职意向：AI Agent 开发工程师</b>（2026 届本科应届 · 意向城市：深圳）', S['intent']),
    Spacer(1, 4),
    Paragraph('电话 131-1892-0989 ｜ 邮箱 1934896060@qq.com ｜ 现居 海口', S['meta']),
    Paragraph('GitHub：<a href="https://github.com/1934896060lxawd-sketch/agent" color="#555555">github.com/1934896060lxawd-sketch/agent</a>', S['meta']),
]
photo = Image(PHOTO, width=2.7*cm, height=2.7*cm*1290/921)
head = Table([[left, photo]], colWidths=[13.2*cm, 3.0*cm])
head.setStyle(TableStyle([
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
    ('LEFTPADDING', (0, 0), (-1, -1), 0),
    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ('TOPPADDING', (0, 0), (-1, -1), 0),
]))
story.append(head)
story.append(Spacer(1, 8))

# ═══ 教育经历 ═══
story += section('教育经历')
edu = [
    ['2024.09 – 2026.06（预计）', '海南师范大学', '计算机科学与技术', '本科'],
    ['2021.09 – 2024.06', '海口经济学院', '计算机网络技术', '专科'],
]
t = Table(edu, colWidths=[4.6*cm, 4.6*cm, 5.0*cm, 2.0*cm])
t.setStyle(TableStyle([
    ('FONTNAME', (0, 0), (-1, -1), 'CJK'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('TEXTCOLOR', (0, 0), (-1, -1), INK),
    ('FONTNAME', (3, 0), (3, -1), 'CJK-B'),
    ('LEFTPADDING', (0, 0), (-1, -1), 0),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ('TOPPADDING', (0, 0), (-1, -1), 0),
]))
story.append(t)
story.append(Spacer(1, 6))

# ═══ 获奖经历 ═══
story += section('获奖经历')
awards = [
    ('<b>一等奖</b>｜第十六届蓝桥杯 海南赛区 C/C++ 程序设计大学 B 组', '2025.05'),
    ('<b>团队二等奖</b>｜团体程序设计天梯赛（GPLT）', '2023'),
    ('<b>二等奖</b>｜中国高校计算机大赛 — 大学生 RPA+AI 创新挑战赛', '2022'),
]
for text, year in awards:
    row = Table([[Paragraph('• ' + text, S['body']),
                  Paragraph(year, S['small'])]], colWidths=[13.6*cm, 2.6*cm])
    row.setStyle(TableStyle([
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(row)
story.append(Spacer(1, 6))

# ═══ 项目经历（STAR）═══
story += section('项目经历')
story.append(Paragraph(
    '<b>智能汽车导购问答机器人</b>（独立开发 · 全栈）　'
    '<font size="9.5" color="#555555">FastAPI ｜ ReAct Agent ｜ RAG ｜ Streamlit ｜ Redis</font>',
    S['body']))
story.append(Spacer(1, 3))

star = [
    ('S 背景', '新能源车型参数与口碑信息分散、购车决策成本高；希望将 LLM Agent + RAG 技术落地为一个可公开访问的全栈产品，覆盖"检索→推理→回答→部署"完整链路。'),
    ('T 任务', '独立完成系统设计、后端 Agent 与混合检索管线、前端对话界面、会话管理、公网部署与全流程测试验收。'),
    ('A 行动', '设计 ReAct Agent 架构：非流式拦截工具调用、最终答案字符级流式输出，<b>根除 XML 工具协议向前端泄露</b>（25 组对抗用例回归全过）；构建 FAISS 向量 + BM25 + RRF 融合 + BGE 精排的混合检索管线，重依赖懒加载 + 启动预热，将首次提问延迟从 40s 优化至 8s；设计上下文锚定机制（结尾提议识别 + 客套话过滤 + 澄清分支），解决多轮对话中"需要/好的"类短回复答非所问；以首轮强制工具调用 + 车型名模糊解析，杜绝模型凭记忆编造参数；实现 Redis 会话 CRUD、滑动窗口限流与访问门禁。'),
    ('R 结果', '知识库覆盖 16 款主流新能源车型；经 8 轮真实场景回归验收（多轮指代 / 泄露对抗 / 会话 CRUD / 公网访问）全部通过；通过 Cloudflare 隧道公网分享实测，支持多访客并发问答。'),
]
for tag, text in star:
    story.append(Paragraph(f'• <b>{tag}</b>｜{text}', S['bullet']))
    story.append(Spacer(1, 2.5))
story.append(Spacer(1, 4))

# ═══ 专业技能 ═══
story += section('专业技能')
skills = [
    '<b>编程语言</b>：Python（熟练，异步编程）、C/C++（蓝桥杯省级一等奖）',
    '<b>AI / LLM</b>：ReAct Agent、Function Calling、Prompt Engineering、RAG 检索增强（FAISS / BM25 / RRF / BGE 精排）、GLM 与 DeepSeek API 接入与切换',
    '<b>后端与工具</b>：FastAPI、Streamlit、Redis、Uvicorn、Git / GitHub、Cloudflare 隧道部署',
]
for s_ in skills:
    story.append(Paragraph('• ' + s_, S['bullet']))
    story.append(Spacer(1, 2.5))
story.append(Spacer(1, 4))

# ═══ 自我评价 ═══
story += section('自我评价')
story.append(Paragraph(
    '对 LLM Agent 方向有从 0 到 1 的完整项目闭环：既能做检索与提示词层面的调优，也能处理工程化细节'
    '（并发、限流、部署、回归测试）。习惯用真实用户场景驱动迭代，本项目经 8 轮真人模拟测试持续修复 20+ 个体验问题。',
    S['body']))

doc = SimpleDocTemplate(OUT, pagesize=A4,
                        topMargin=1.5*cm, bottomMargin=1.4*cm,
                        leftMargin=1.9*cm, rightMargin=1.9*cm,
                        title='罗小安 - AI Agent开发工程师 - 简历',
                        author='罗小安')
doc.build(story)
print('PDF 已生成:', OUT)
