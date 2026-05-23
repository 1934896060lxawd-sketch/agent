# 常见的数据类型
# type()--获取数据类型
# print(type(10))
# print(type(3.14))
# print(type("hello"))
# print(type(True))
# print(type(False))
# print(type(None))

# isinstance(数据，类型) 检查数据是否是指定数据类型
# num = 10
# print(isinstance(num,int))
# print(isinstance(num,float))
# print(isinstance(1,float))
# print(isinstance(1,int))


# 字符串格式化  方式一：f“...”
# name = "罗小安"
# age = 23
# pro = "计算机"
# hobby = "python,java"
# print(f"大家好，我是{name},今年{age}岁,学习的专业是{pro},爱好{hobby}")
#
# # 字符串格式化  方式二：%s 占位符
# name = "罗小安"
# age = 23
# pro = "计算机"
# hobby = "python,java"
# print("大家好，我是%s,今年%s岁,学习的专业是%s,爱好%s" % (name,age,pro,hobby))