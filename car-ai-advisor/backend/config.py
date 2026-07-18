"""
统一配置管理 — 所有配置项集中定义，通过环境变量 / .env 文件注入。

使用方式:
    from backend.config import settings
    api_key = settings.llm_api_key
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ============================================================
    # 服务
    # ============================================================
    app_port: int = 8000
    host: str = "0.0.0.0"
    debug: bool = False
    log_level: str = "info"

    # ============================================================
    # LLM
    # ============================================================
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model_id: str = "deepseek-chat"

    # ============================================================
    # Embedding
    # ============================================================
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # ============================================================
    # Redis
    # ============================================================
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    redis_url: str = "redis://redis:6379/0"

    # ============================================================
    # Session
    # ============================================================
    session_ttl_seconds: int = 1800
    max_concurrent_per_user: int = 10  # 开发环境放宽，生产改回3

    # ============================================================
    # RAG (检索增强生成)
    # ============================================================
    rag_top_k: int = 5
    rag_rrf_k: int = 60
    knowledge_base_dir: str = "knowledge_base"

    # ============================================================
    # 限流
    # ============================================================
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    # ============================================================
    # JWT
    # ============================================================
    jwt_secret: str = "change-me-to-a-random-string"
    jwt_expire_minutes: int = 60

    # ============================================================
    # API Keys (开发阶段简易鉴权)
    # ============================================================
    api_keys: str = "sk-dev-user-001:user_001,sk-dev-admin-002:user_admin"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # 允许额外字段，避免新增环境变量时启动失败
        extra = "ignore"


# 全局单例
settings = Settings()
