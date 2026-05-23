# import random   # 导入模块
# from random import randint  # 导入模块的功能
# num = [1,2,3,4,5,6,7,8,9,10]
# # 调用方式：模块名.功能名
# print(random.choice(num))
#
# for i in range(10):
#     print(randint(1,100))

# 导入包中的模块
# 方式1
# import module.fun
# module.fun.log_1()

# 方式2
# from module import fun
# fun.log_2()

# 方式3
# 通过这种方式导入包下所有模块，需要在__init__.py 文件中添加__all__ = []
# from module import *
# fun.log_1()


# 导入包中模块的功能
# from module.fun import log_1
# log_1()