import json
#
# data = {
#     "name": "lxa",
#     "age": 25,
#     "is_student": False,
#     "hobbies": ["篮球", "编程"],
#     "score": {"math": 90, "english": 80},
#     "address": None
# }
#
# # 将python对象转化为json格式
# json_str = json.dumps(data, ensure_ascii=False) # 如需保留中文，加ensure_ascii=False参数
# print(json_str)
#
# # 用indent设置缩进可美化格式，sort_keys=True可按键名排序
# json_str = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
# print(json_str)
#
# with open('data.json', 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
#
#
# #将 JSON 字符串或 JSON 文件转换为 Python 对象
# json_str = '''
# {
#   "name": "张三",
#   "age": 25,
#   "is_student": false,
#   "hobbies": ["篮球", "编程"],
#   "scores": {"math": 90, "english": 85},
#   "address": null
# }
# '''
#
# data1 = json.loads(json_str)
# print(data1)
# print(type(data1))
# print(data1["name"])
# print(data1["hobbies"][0])
# print(data1["scores"]["math"])
#
# # 直接读取 JSON 文件并转换为 Python 对象
# with open("data.json", "r", encoding="utf-8") as f:
#     data = json.load(f)  # 直接从文件加载并转换
#     print(data["age"])

