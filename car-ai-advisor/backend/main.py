import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as redis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 修复 Windows GBK 终端下 emoji 等 Unicode 字符导致的 UnicodeEncodeError
# Python 默认用系统编码 (Windows: gbk) 写 stdout/stderr，HTTP 响应走 UTF-8 不受影响
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.append(str(Path(__file__).parent.parent))
from backend.config import settings
from backend.api.routes import chat, sessions

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时连接 Redis，关闭时释放连接。"""
    logger.info("Application startup...")
    try:
        app.state.redis = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password or None,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        await app.state.redis.ping()
        logger.info("Redis 连接成功")
    except Exception as e:
        logger.warning(f"Redis 连接失败 ({e})，降级为 fakeredis 内存模式")
        import fakeredis.aioredis
        app.state.redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        logger.info("fakeredis 就绪（数据不持久化）")

    yield

    if app.state.redis:
        await app.state.redis.close()
        logger.info("Redis 连接已关闭")
    logger.info("Application shutdown...")


app = FastAPI(
    title="智能汽车导购助手 API",
    description="全链路 AI Agent 系统后端",
    version="0.1.0",
    lifespan=lifespan,
)

# ============================================================
# 中间件
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 全局异常处理
# ============================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ============================================================
# 路由
# ============================================================
@app.get("/")
def read_root():
    return {"message": "智能汽车导购助手 API v0.1.0"}


@app.get("/health")
async def health_check(request: Request):
    redis_alive = False
    redis_client = request.app.state.redis
    if redis_client:
        try:
            await redis_client.ping()
            redis_alive = True
        except Exception:
            redis_alive = False
    return {
        "status": "running",
        "redis_alive": redis_alive,
    }


# ============================================================
# 注册业务路由
# ============================================================
app.include_router(chat.router)
app.include_router(sessions.router)
