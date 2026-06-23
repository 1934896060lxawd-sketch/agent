from fastapi import FastAPI, Path
import uvicorn

app = FastAPI(title="第一个fastapi")

# 测试接口
@app.get("/")
async def index():
    return {"msg": "FastAPI 服务启动成功"}

# 路由
# 使用装饰器将url与执行函数映射
@app.get('/hello')
async def get_hello():
    return {"msg": "hello,fastapi"}

# 路径参数
@app.get("/book/{id}")
async def get_book(id: int = Path(..., gt=0, le=101, description="书籍id，取值范围：1-100")): # Path类型注解
    return {"id": id, "title": f"这是第{id}本书"}

@app.get("/author/{name}")
async def get_name(name: str = Path(..., min_length=2, max_length=11, description="作者，名字长度范围：2-10")): # Path类型注解
    return {"name": name, "title": f"这是{name}的信息"}

# 查询参数


if __name__ == "__main__":
    uvicorn.run(
        app="s01_basic:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )