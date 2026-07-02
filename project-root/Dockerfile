# ===== Builder Stage =====
FROM python:3.11-slim AS builder
WORKDIR /app

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt

# ===== Runtime Stage =====
FROM python:3.11-slim AS runtime     
WORKDIR /app

# 创建非 root 用户和组
RUN groupadd -r app && useradd -r -g app app

COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# 完整复制当前项目所有文件（包含backend文件夹）
COPY . .
RUN chown -R app:app /app

USER app                               

EXPOSE 8000

# 【关键修改】启动路径改为 backend.main:app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
