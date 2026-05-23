# # 写入文件（‘w’模式）
# with open('test.txt', 'w', encoding='utf-8') as f:  # with语句自动关闭文件
#     f.write('hello world\n')    # 写入字符串
#     f.write("哈喽，世界，你好！！！\n")     # 需指定encoding才能写入中文
#
# # 读取文件（‘r’模式）
# with open('test.txt', 'r', encoding='utf-8') as f:
#     content = f.read()  # 读取全部内容
#     print(content)
#
# # 二进制文件操作（如图片，音频）
# with open('source.jpg', 'rb') as scr, open('copy.jpg', 'wb') as dst:
#     data = scr.read()   # 读取二进制数据
#     dst.write(data)     # 写入二进制数据
# print("图片复制完成！！！")


# 5种读取方式
# # 先创建测试文件
# with open('read_demo.txt', 'w', encoding='utf-8') as f:
#     f.write('第一行内容\n')
#     f.write('第二行内容\n')
#     f.write('第三行内容\n')
#
# # 1. read()：读取全部内容
# with open('read_demo.txt', 'r', encoding='utf-8') as f:
#     print('1. read() 结果：\n', f.read())
#
# # 2. readline()：逐行读取
# with open('read_demo.txt', 'r', encoding='utf-8') as f:
#     print('\n2. readline() 结果：')
#     print(f.readline(), end='')  # 第一行（end='' 避免重复换行）
#     print(f.readline(), end='')  # 第二行
#
# # 3. readlines()：读取所有行到列表
# with open('read_demo.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     print('\n3. readlines() 结果：', lines)
#     print('按索引访问第二行：', lines[1])
#
# # 4. 迭代文件对象（最推荐的大文件处理方式）
# with open('read_demo.txt', 'r', encoding='utf-8') as f:
#     print('\n4. 迭代文件对象：')
#     for line in f:
#         print(f'行内容：{line}', end='')
#
# # 5. read(n)：读取指定字节数（文本模式下约等于字符数，非精确）
# with open('read_demo.txt', 'r', encoding='utf-8') as f:
#     print('\n\n5. read(n) 结果：')
#     print(f.read(5))  # 读取前5个字符
#     print(f.read(3))  # 继续读取3个字符
#
#
# # 3种写入方式
# # write():写入字符串
# with open('write_demo.txt', 'w', encoding='utf-8') as f:
#     count1 = f.write("用write写入第一行\n")
#     count2 = f.write("用write写入第一行")
#     print(f"write写入字符数：{count1}, {count2}")
#
# # writelines():写入序列（无自动换行）
# with open('write_demo.txt', 'a', encoding='utf-8') as f:
#     lines = ['\n用writelines写入第三行', '用writelines写入第四行']
#     f.writelines(lines)
#
# # 3. print() 函数写入（自动换行）
# with open('write_demo.txt', 'a', encoding='utf-8') as f:
#     print('用print写入第五行', file=f)  # 自动加换行符
#     print('用print写入第六行', file=f, end='---')  # 自定义结尾符

