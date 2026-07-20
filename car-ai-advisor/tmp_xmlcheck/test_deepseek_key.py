"""实测 DeepSeek 旧 key 当前是否可用（余额是否恢复）。不打印密钥。"""
import os, sys, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cfg = {}
for line in open(".env.bak.deepseek", encoding="utf-8"):
    line = line.strip()
    if line.startswith("LLM_") and "=" in line:
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()

from openai import AsyncOpenAI

async def main():
    print("BASE_URL:", cfg.get("LLM_BASE_URL"), "| MODEL:", cfg.get("LLM_MODEL_ID"),
          "| KEY:", cfg.get("LLM_API_KEY", "")[:6] + "***")
    client = AsyncOpenAI(api_key=cfg["LLM_API_KEY"], base_url=cfg["LLM_BASE_URL"], timeout=30)
    try:
        resp = await client.chat.completions.create(
            model=cfg["LLM_MODEL_ID"],
            messages=[{"role": "user", "content": "你好，一句话回答：1+1=?"}],
            max_tokens=30,
        )
        print("✅ 调用成功:", resp.choices[0].message.content.strip()[:50])
        print("usage:", resp.usage)
    except Exception as e:
        print("❌ 调用失败:", type(e).__name__, str(e)[:200])

asyncio.run(main())
