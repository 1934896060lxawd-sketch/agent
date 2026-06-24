from fastapi import FastAPI, Path, Query, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse, FileResponse
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
@app.get("/new/list")
async def get_new(
    skip: int = Query(0, description="跳过的记录数"),# Query类型参数
    limit: int = Query(10, description="返回的记录数")
):
    return {"skip": skip, "limit": limit}

# 请求体参数
class User(BaseModel):
    # Field类型注解
    username: str = Field(default="luoxiaoan", min_length=2, max_length=11, description="用户名")
    password: str = Field(default="193489", min_length=2, max_length=11, description="密码")

@app.post("/register")
async def register(user: User):
    return user

# 响应类型：默认支持json格式，还有html，文件下载
# 响应HTML,在装饰器中响应
@app.get("/html", response_class=HTMLResponse)
async def get_html():
    return "<h1>hello world<h1>"

# 响应文件格式
@app.get("/file")
async def get_file():
    path = "../agent-dev-project/prompt/images/su7.jpg"
    return FileResponse(path)

# 自定义响应数据格式
class News(BaseModel):
    id: int
    title: str
    content: str

@app.get("/news/{id}", response_model=News)
async def get_news(id: int):
    return{
        "id": id,
        "title": f"这是第{id}本书",
        "content": "这是一本好书"
    }

# 异常处理
@app.get("/id/{id}")
async def get_id(id: int):
    id_list = [1,2,3,4,5,6]
    if id not in id_list:
        raise HTTPException(status_code=404, detail="资源不存在")
    return {"id": id}

if __name__ == "__main__":
    uvicorn.run(
        app="s01_basic:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    ) 