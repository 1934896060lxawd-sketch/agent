# 异常处理
try:
    print("-------------------")
    print(a)
    print("-------------------")
except NameError as e:
    print("程序运行出错！！！错误信息：",e)
except  Exception:
    print("未知错误！！！")