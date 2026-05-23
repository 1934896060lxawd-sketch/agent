def log_1():
    print("-" * 30)

def log_2():
    print("+" * 30)

def log_3():
    print("$" * 30)

def log_4():
    print("@" * 30)

# 测试函数
# __name__:python中的内置变量，表示当前模块的名字（直接运行当前模块，__name__的值为__main__,当模块被导入时__name__的值就是模块名）
#print(__name__)
if __name__ == '__main__':
    log_1()
