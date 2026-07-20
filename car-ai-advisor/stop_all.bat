@echo off
echo 停止汽车导购所有服务...
taskkill /FI "WINDOWTITLE eq car-backend" /F >/dev/null 2>&1
taskkill /FI "WINDOWTITLE eq car-frontend" /F >/dev/null 2>&1
taskkill /FI "WINDOWTITLE eq car-tunnel" /F >/dev/null 2>&1
taskkill /IM cloudflared.exe /F >/dev/null 2>&1
echo 已停止。
pause
