# 学生类
class Student:
    def __init__(self, name, chinese ,math ,english):
        self.name = name
        self.chinese = chinese
        self.math = math
        self.english = english

    def __str__(self):
        return f'学生姓名：{self.name}  语文成绩： {self.chinese}  数学成绩： {self.math}  英语成绩： {self.english}'

    # 修改成绩
    def update(self, chinese = None, math = None, english = None):
        if chinese is not None:
            self.chinese = chinese
        if math is not None:
            self.math = math
        if english is not None:
            self.english = english


#教务系统类
class EduManagement:
    version = "1.0"
    name = "教务管理系统"

    def __init__(self):
        self.students_list = []

    # 添加学生信息
    def add_student(self):
        name = input("请输入学生姓名：")
        # 不能重复添加学生
        for s in self.students_list:
            if s.name == name:
                print("已经有学生姓名，不要重复添加！")
                return

        chinese = int(input("请输入学生语文成绩："))
        math = int(input("请输入学生数学成绩："))
        english = int(input("请输入学生英语成绩："))

        if 0 <= chinese <= 100 and 0 <= math <= 100 and 0 <= english <= 100:
            stu = Student(name,chinese,math,english)
            self.students_list.append(stu)
            print("成功添加学生信息！")
        else:
            print("各科成绩要在0 - 100 之间！")

    # 修改学生成绩
    def update_student(self):
        name = input("请输入要修改的学生姓名：")
        for s in self.students_list:
            if s.name == name:
                print(f"当前信息：{s}")

                chinese = int(input("请输入修改后学生语文成绩："))
                math = int(input("请输入修改后学生数学成绩："))
                english = int(input("请输入修改后学生英语成绩："))

                s.update(chinese, math, english)
                print("成绩修改成功！")
                print(f"当前信息：{s}")
                return
        print("未找到该学生，修改失败！")


    # 删除学生成绩
    def del_student(self):
        name = input("请输入要删除的学生姓名：")
        for s in self.students_list:
            if s.name == name:
                self.students_list.remove(s)
                print("删除成功")
                return

        print("未找到该学生，删除失败！")


    # 查询学生成绩
    def query_student(self):
        name = input("请输入要查询的学生姓名：")
        for s in self.students_list:
            if s.name == name:
                print(f"当前信息：{s}")
                return

        print("未找到该学生！")


    # 输出所有学生信息
    def all_student(self):
        for s in self.students_list:
            print(s)


    # 运行系统
    def run():
        # 创建教务系统对象
        edu_system = EduManagement()
        print(f"欢迎使用 {edu_system.name} 版本：{edu_system.version}")

        while True:
            print("\n===== 菜单 =====")
            print("1. 添加学生")
            print("2. 修改学生成绩")
            print("3. 删除学生")
            print("4. 查询学生")
            print("5. 显示所有学生")
            print("0. 退出系统")
            choice = input("请输入操作序号：")

            if choice == "1":
                edu_system.add_student()
            elif choice == "2":
                edu_system.update_student()
            elif choice == "3":
                edu_system.del_student()
            elif choice == "4":
                edu_system.query_student()
            elif choice == "5":
                edu_system.all_student()
            elif choice == "0":
                print("感谢使用，再见！")
                break
            else:
                print("输入无效，请重新输入！")


# 测试
if __name__ == '__main__':
    EduManagement.run()