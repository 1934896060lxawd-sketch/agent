import requests
import json

#
# # GET 请求
# response = requests.get("https://www.example.com")
# # 查看响应状态码
# print("响应状态码：", response.status_code)
# # 查看响应内容，content返回字节流，text返回字符串形式
# print("响应内容(字节流)：", response.content)
# print("响应内容(字符串)：", response.text)
# # 查看响应头信息
# print("响应头：", response.headers)
#
# # POST 请求
# # 目标登录URL
# login_url = "https://www.example.com/login"
# # 待提交的登录数据
# login_data = {
#     "username": "lxa",
#     "password": "123456"
# }
# # 发送POST请求
# response = requests.post(login_url, data=login_data)
# # 判断登录是否成功
# if response.status_code == 200 and "登录成功" in response.text:
#     print("登录成功！")
# else:
#     print("登录失败！")

# 请求头设置：为了避免被目标网站识别为爬虫并进行封禁，设置合理的请求头至关重要
# 自定义请求头
# headers = {
#     'User - Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
#     'Referer': 'https://www.example.com/index'
# }
# # 发送带有自定义请求头的GET请求
# response = requests.get('https://www.example.com/data', headers=headers)
#
# # Cookie 处理
# cookies = {
#     'session_id': '1234567890abcdef',
#     'user_id': '123'
# }
# # 发送带有手动设置Cookie的请求
# response = requests.get('https://www.example.com/user/info', cookies=cookies)
#
# # 自动处理 Cookie
# # 创建Session对象
# session = requests.Session()
# # 首先发送登录请求，Session会自动保存登录后的Cookie
# login_data = {
#     'username': 'your_username',
#     'password': 'your_password'
# }
# session.post('https://www.example.com/login', data=login_data)
# # 后续请求会自动携带登录后的Cookie，无需手动设置
# response = session.get('https://www.example.com/user/order')
# print('用户订单信息：', response.text)
#
# # 代理设置
# # 设置代理，key为协议类型，value为代理地址
# proxies = {
#     'http': 'http://127.0.0.1:8888',
#     'https': 'https://127.0.0.1:8888'
# }
# # 发送带有代理的请求
# response = requests.get('https://www.example.com', proxies=proxies)
#
# # 超时设置
# try:
#     # 设置超时时间为5秒
#     response = requests.get('https://www.example.com', timeout=5)
#     print('请求成功！')
# except requests.exceptions.Timeout:
#     print('请求超时！')
#


# API 接口数据爬取 —— 高德地图天气API
import requests  # 发送网络请求
import json  # 解析JSON格式数据

# 天气API接口URL（高德地图官方接口）
weather_api_url = 'https://restapi.amap.com/v3/weather/weatherInfo'

# API请求参数（必须填写正确的key）
params = {
    'key': '75621524805435bb3de5b03e30570e68',  # 重点：这里要换成你申请的真实key
    'city': '110000',  # 城市编码：110000 = 北京
    'extensions': 'base'  # base=基础天气信息
}

# 发送GET请求，调用API接口
response = requests.get(weather_api_url, params=params)

# 把返回的字符串转成Python字典（方便取值）
weather_data = json.loads(response.text)

# 提取并打印天气信息
if weather_data['status'] == '1':
    # 取第一条实时天气数据
    live_weather = weather_data['lives'][0]

    print("===== 实时天气信息 =====")
    print(f'城市：{live_weather["city"]}')
    print(f'天气：{live_weather["weather"]}')
    print(f'温度：{live_weather["temperature"]}℃')
    print(f'风向：{live_weather["winddirection"]}')
    print(f'风力：{live_weather["windpower"]} 级')
    print(f'湿度：{live_weather["humidity"]}%')
    print(f'发布时间：{live_weather["reporttime"]}')
else:
    print('获取天气数据失败！错误信息：', weather_data.get('info', '未知错误'))