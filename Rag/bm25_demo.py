import math, re

docs = [  # 三条玩具"文档",当作我们的知识库
    "等待期说明:本产品等待期为90天,等待期内出险不予赔付",
    "理赔需要的材料清单与销售流程介绍",
    "本保险承保意外伤害导致的身故或残疾",
]


def tokenize(s):
    return list(re.sub(r"\s", "", s))  # 中文按"字"切(真实系统用 jieba 等做词级切分)


corpus = [tokenize(d) for d in docs]  # 把每条文档切成字的列表
N = len(corpus)  # 文档总数
avgdl = sum(len(d) for d in corpus) / N  # 平均文档长度(给"长度归一"用)
df = {}  # df[字] = 有多少篇文档出现过这个字
for d in corpus:
    for t in set(d):  # set 去重:同一篇里出现几次只算 1
        df[t] = df.get(t, 0) + 1


def idf(t):  # IDF:越稀有的字越值钱(命中它几乎锁定答案)
    n = df.get(t, 0)  # 出现过这个字的文档数
    return math.log(1 + (N - n + 0.5) / (n + 0.5))  # 标准 BM25 的 IDF 公式


def bm25(query, doc, k1=1.5, b=0.75):  # k1 控饱和速度, b 控长度归一强度
    score, dl = 0.0, len(doc)  # score 累加分, dl 当前文档长度
    for t in tokenize(query):  # 逐个看 query 里的字
        if t not in doc: continue  # 这个字文档里没有 → 不贡献分
        f = doc.count(t)  # 词频 TF:这个字在文档里出现几次
        tf_sat = f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))  # 词频饱和 + 文档长度归一
        score += idf(t) * tf_sat  # 稀有度(IDF) × 饱和后的词频,累加
    return score


q = "等待期多久"
for i in sorted(range(N), key=lambda i: bm25(q, corpus[i]), reverse=True):  # 按分数从高到低排
    print(f"  score={bm25(q, corpus[i]):.2f}  {docs[i]}")
