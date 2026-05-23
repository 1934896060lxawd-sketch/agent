# 列表
# 特点  1.可以存储不同类型元素   2.元素有序，可以重复，可以修改

# s = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# # 访问列表中的元素
# print(s[0])
# print(s[-1])
#
# # 修改列表
# s[0] = 10
# print(s)
#
# # 删除列表元素
# del s[0]
# print(s)
#
# # 循环
# for i in s:
#     print(i)
#
# print(type(s))

# 切片
# num = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# print(num[1:5])
# print(num[:5])
# print(num[::2])
# print(num[::-1])

"""
方法
append()    在列表尾部追加元素
insert()    指定位置之前插入元素
remove()    删除第一个匹配的元素
pop()       删除指定位置的元素并返回
sort()      排序
reverse()   反转列表i
"""

# num = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#
# num.append(10)
# print(num)
#
# num.insert(1, 11)
# print(num)
#
# num.remove(11)
# print(num)
#
# num.reverse()
# print(num)
#
# num.pop(5)
# num.pop()
# print(num)
#
# num.sort()
# print(num)

# 案例1
# i = 0
# a = 0
# s = []
# while i < 10:
#     num = input("输入数字：")
#     a += int(num)
#     s.append(num)
#     i += 1
# print(s)
#
# s.sort()
# print(s)
#
# print(s[0])
# print(s[-1])
# print(a / 10)


# num1 = [1,2,3,4,5,6,7,8,9,10]
# num2 = [11,22,33,44,55,66,77,88,99]
# #合并列表
# # num = num1 + num2
# # num = [*num1,*num2] 解包
# # num1.extend(num2)
# print(num)

# 列表推导式 [插入的值 for i in 序列 if 条件]  if条件可有可无
#推导式1
# num = [i**2 for i in range(1,21)]
# print(num)
#推导式2
# num = [i**2 for i in range(1,21) if i % 2 == 0]
# print(num)
#
# for i in range(10):# 默认从零开始
#     print(i)