import os, sys, json, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from openai import AsyncOpenAI
from backend.agent.tools import TOOL_SCHEMAS

USER_Q = "比亚迪宋L的详细参数"

async def test_model(model: str):
    client = AsyncOpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_BASE_URL"])
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是汽车导购助手，回答车型参数前必须先调用工具查询真实数据。"},
                {"role": "user", "content": USER_Q},
            ],
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.7,
        )
        msg = resp.choices[0].message
        tcs = [tc.function.name for tc in (msg.tool_calls or [])]
        print(f"--- {model} ---")
        print("finish_reason:", resp.choices[0].finish_reason)
        print("tool_calls:", tcs or "(none)")
        print("content[:150]:", (msg.content or "")[:150].replace("\n", " "))
    except Exception as e:
        print(f"--- {model} --- ERROR: {type(e).__name__}: {str(e)[:200]}")

async def main():
    for m in ["glm-4-flash", "glm-4.7-flash", "glm-4.5-flash"]:
        await test_model(m)

asyncio.run(main())
