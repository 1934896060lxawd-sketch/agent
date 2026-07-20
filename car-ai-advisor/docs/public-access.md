# 公开访问指南 — 让任何人体验汽车导购助手

> 架构：访客浏览器 → Cloudflare 隧道（HTTPS）→ 本机 Streamlit(8501) → 本机后端(127.0.0.1:8000) → DeepSeek API
> 暴露面最小化：**只有 Streamlit 前端经隧道暴露**；后端只监听 127.0.0.1；DeepSeek Key 留在本机 .env。

## 一、快速开始（日常使用）

| 操作 | 方式 |
|---|---|
| 启动全部 | 双击 `start_all.bat`（后端→前端→隧道，自动等后端就绪） |
| 获取公网地址 | 在 `car-tunnel` 窗口找 `https://xxxx.trycloudflare.com` |
| 分享给朋友 | 公网地址 + 访问密码（见下） |
| 停止全部 | 双击 `stop_all.bat` |

## 二、访问控制（.streamlit/secrets.toml）

```toml
ACCESS_PASSWORD = "car-demo-2026"      # 改成你自己的密码，分享给朋友时告知
MAX_QUESTIONS_PER_SESSION = 50         # 单访客提问上限，0=不限
```

- 未设密码 → 任何人可访问（仅建议本机开发时留空）
- secrets.toml 已加入 .gitignore，不会被提交

## 三、注意事项

1. **临时地址每次重启都会变**：Cloudflare Quick Tunnel 是随机域名，重启隧道后要重新复制链接。
2. **电脑必须开机且脚本运行**，别人才能访问。
3. **DeepSeek 费用由你承担**：每个访客提问都会消耗你的 API 额度，密码+次数上限是基本防线。
4. 国内访问 trycloudflare.com 速度因地而异；若朋友打不开，可换 cpolar / natapp 等国内隧道（配置类似，指向 8501 即可）。
5. 后端 uvicorn 固定绑定 **127.0.0.1**，不要改成 0.0.0.0——隧道是唯一入口。

## 四、长期方案（正式对外）

| 方案 | 适合 | 要点 |
|---|---|---|
| **云服务器 + docker-compose** | 长期稳定运营 | 项目自带 Dockerfile/docker-compose.yml（含 Redis）；服务器上 `docker-compose up -d --build` 即可；配域名 + Caddy/Nginx 自动 HTTPS；注意 requirements.txt 变更后需重建镜像 |
| **Cloudflare Named Tunnel** | 免费但要固定域名 | 需自有域名接入 Cloudflare，隧道固定域名且更稳定 |
| **HuggingFace Spaces** | 免费演示 | Docker SDK 直接部署本仓库；冷启动较慢，注意 HF 在国内可达性 |

### 云服务器部署清单（到时照做）
1. 买一台 2C4G 云主机（镜像构建需约 3GB 磁盘 + torch 依赖较大）
2. 安装 Docker → 克隆仓库 → 配 `.env`（DeepSeek Key）
3. `docker-compose up -d --build`
4. 域名 A 记录指向服务器 → Caddy 反代 8501 自动签发 HTTPS
5. 云安全组只放行 80/443，8000/6379 不对外

## 五、公开前安全检查清单

- [x] 访问密码 + 提问次数上限（前端门禁）
- [x] 后端仅监听 127.0.0.1
- [x] DeepSeek Key 只在 .env（已 gitignore）
- [x] secrets.toml 已 gitignore
- [x] AI 回答 XML/DSML 标记清理（2026-07-19 修复）
- [x] 后端错误不暴露内部细节（统一"服务暂时不可用"）
- [x] 流式接口过滤内部工具事件

## 六、故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| start_all 后端卡 40 秒+ | 本机网络过滤导致 Redis 连接被延迟 | 已修复：脚本默认 USE_FAKEREDIS=1 直启内存模式 |
| 回答总是"服务暂时不可用" | DeepSeek 余额不足（402 Insufficient Balance） | 登录 platform.deepseek.com 充值；或更换 .env 里的 llm_base_url/llm_model_id/llm_api_key 到其他 OpenAI 兼容服务 |
| 隧道链接打不开 | 临时域名已失效（隧道重启过） | 到 car-tunnel 窗口复制新链接 |
| 第一个问题特别慢 | 首次请求加载嵌入模型 | 正常现象，10-30 秒 |
