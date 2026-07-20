@echo off
title 汽车导购-启动器
cd /d E:\coding\agent\car-ai-advisor
set PY=E:\coding\agent\.venv\Scripts\python.exe

echo [1/3] 启动后端 API (127.0.0.1:8000)...
rem USE_FAKEREDIS=1: 跳过Redis连接直接用内存模式，避免本机网络过滤导致启动卡40秒+
start "car-backend" cmd /k "set USE_FAKEREDIS=1&& %PY% -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

echo [2/3] 等待后端就绪（约5-10秒）...
:wait_backend
curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1
if errorlevel 1 (
    timeout /t 2 >/dev/null
    goto wait_backend
)
echo       后端就绪 OK

echo [3/3] 启动前端 (8501) 与公网隧道...
start "car-frontend" cmd /k "%PY% -m streamlit run frontend/app.py --server.port 8501"
timeout /t 5 >/dev/null
start "car-tunnel" cmd /k "tools\cloudflared.exe tunnel --url http://localhost:8501"

echo.
echo ====================================================
echo  全部启动完成！
echo  1. 打开 car-tunnel 窗口
echo  2. 找到 https://xxxx.trycloudflare.com 这一行
echo  3. 把该链接和访问密码发给朋友即可体验
echo  （密码在 .streamlit/secrets.toml 的 ACCESS_PASSWORD）
echo  提示：模型已随启动自动预热，访客提问无需等待加载
echo ====================================================
pause
