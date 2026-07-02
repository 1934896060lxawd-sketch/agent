.PHONY: help install run dev test lint clean build-index docker-up docker-down

help: ## 显示帮助信息
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## 安装依赖
	pip install -r requirements.txt

run: ## 启动 FastAPI 服务
	cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

dev: ## 同时启动后端 + 前端
	@echo "启动 Redis..."
	@docker start car-advisor-redis 2>/dev/null || docker run -d --name car-advisor-redis -p 6379:6379 redis:7-alpine
	@echo "启动后端..."
	cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
	@sleep 2
	@echo "启动前端..."
	cd frontend && streamlit run app.py --server.port 8501

test: ## 运行测试
	pytest tests/ -v --cov=backend --cov-report=term-missing

lint: ## 代码检查
	ruff check backend/ frontend/ tests/

clean: ## 清理临时文件
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build-index: ## 构建知识库索引
	cd knowledge_base/scripts && python build_index.py

docker-up: ## Docker Compose 启动
	docker-compose up -d --build

docker-down: ## Docker Compose 停止
	docker-compose down
