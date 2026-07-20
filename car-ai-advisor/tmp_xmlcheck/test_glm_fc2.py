import os, sys, json, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from openai import AsyncOpenAI
from backend.agent.tools import TOOL_SCHEMAS
from backend.agent.prompts import CAR_ADVISOR_SYSTEM_PROMPT

QUESTIONS = ["25万预算推荐什么SUV", "比亚迪宋L的详细参数"]

async def test_model(model: str, tool_choice: str):
    client = AsyncOpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_BASE_URL"])
    for q in QUESTIONS:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CAR_ADVISOR_SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                ],
                tools=TOOL_SCHEMAS,
                tool_choice=tool_choice,
                temperature=0.7,
            )
            msg = resp.choices[0].message
            tcs = [tc.function.name for tc in (msg.tool_calls or [])]
            print(f"[{model} | {tool_choice}] Q={q[:12]}... tools={tcs or '(NONE)'} content_len={len(msg.content or '')}")
        except Exception as e:
            print(f"[{model} | {tool_choice}] Q={q[:12]}... ERROR {type(e).__name__}: {str(e)[:150]}")

async def main():
    for m in ["glm-4-flash", "glm-4.7-flash", "glm-4.5-flash"]:
        await test_model(m, "auto")
    # 对照：强制首轮必须调工具
    await test_model("glm-4-flash", "required")

asyncio.run(main())
