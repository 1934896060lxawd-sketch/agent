# # 定义类
# class Car:
#     # 类属性   通过类名.属性
#     whlee = 4
#     tax_rate = 0.1
#
#     # __init__方法，初始化方法，在对象创建时自动调用
#     # self:表示当前创建的实例对象
#     def __init__(self, brand, name, price):
#         # 实例属性
#         self.brand = brand
#         self.name = name
#         self.price = price
#
#     # 定义实例方法
#     def run(self):
#         print(f"{self.brand} {self.name} 正在高速行驶。。。")
#
#     def total(self,dis,rate):
#         cost = self.price*dis+rate*self.price
#         return cost
#
#
# # 创建对象
# c1 = Car("BWM","X5",500000)
#
# print(c1)
# print(c1.brand)
# print(c1.__dict__)  # 将对象中的所有属性以字典形式输出
#
# c1.run()
# cost = c1.total(0.9,0.1)
# print(cost)
