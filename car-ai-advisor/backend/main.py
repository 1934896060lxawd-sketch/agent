import asyncio
import logging
import os
import sys
import time
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
    """应用生命周期：启动时连接 Redis，关闭时释放连接。

    环境变量 USE_FAKEREDIS=1 → 跳过 Redis 连接，直接使用内存模式。
    本机开发/演示场景必备：部分 Windows 环境下 localhost 拒绝连接会被
    网络过滤驱动延迟数秒，叠加 redis 客户端重试导致启动卡 40 秒以上。
    """
    logger.info("Application startup...")
    if os.getenv("USE_FAKEREDIS") == "1":
        import fakeredis.aioredis
        app.state.redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        logger.info("USE_FAKEREDIS=1 → fakeredis 内存模式（数据不持久化）")
    else:
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

    # 后台预热模型，不阻塞 /health（秒级就绪）；访客首问不再等 10-30 秒
    asyncio.create_task(_warmup_models())

    yield

    if app.state.redis:
        await app.state.redis.close()
        logger.info("Redis 连接已关闭")
    logger.info("Application shutdown...")


async def _warmup_models() -> None:
    """后台预热：构建 Agent + 跑一次真实检索，带起嵌入/精排模型加载。

    嵌入与精排模型（torch，20-40 秒）已改为函数内延迟导入，uvicorn 因此
    秒级就绪；真正的模型加载挪到本任务，让外网访客的第一个问题无需等待。
    预热失败不影响服务——首个真实请求会现场加载（等价于旧行为）。
    """
    loop = asyncio.get_running_loop()

    def _work() -> None:
        try:
            from backend.api.deps import build_agent_singleton

            t0 = time.monotonic()
            agent = build_agent_singleton()
            # 一次真实检索：触发嵌入模型 + jieba + 精排模型全部就位
            agent.executor._tool_search_car_knowledge("预热：比亚迪海豚参数")
            logger.info("模型预热完成，耗时 %.1fs", time.monotonic() - t0)
        except Exception as e:
            logger.warning("模型预热失败（首个请求将现场加载）: %s", e)

    await loop.run_in_executor(None, _work)


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
