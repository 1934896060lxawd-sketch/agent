# 字面量的写法
# print(100)  # int
# print(3.14) # float
# print(True) # bool
# print(False) # bool
# print("hello world") # str
# print(None) #NoneType
#
# # 布尔类型本质上是整数类型
# # True - 1     False - 0
# print(True + 1)
# # print(False - 1)

#变量  python中的变量是动态数据类型
# num = 100
# print(num)
#
# num = 3.14
# print(num)
#
# num = "OK"
# print(num)
#
# num = True
# print(num)

# 案例
# a = 20.7 # 基础播放量
# b = 50 # 每一个月新增播放量
# print("未来第一个月播放总量：",a + b)
# print("未来第二个月播放总量：",a + b + b)

# 交换变量
a = 10
b = 20
c = a
a = b
b = c
print(a)
print(b)