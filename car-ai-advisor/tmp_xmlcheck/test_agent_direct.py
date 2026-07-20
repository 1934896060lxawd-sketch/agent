"""绕开 SSE 路由的 tool: 过滤，直接观测 agent 的工具调用与回答（真实模型全加载）。"""
import asyncio, json, sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_FAKEREDIS", "1")
from dotenv import load_dotenv
load_dotenv()

from backend.api.deps import build_agent_singleton

TURNS = ["25万预算推荐什么SUV", "详细说明比亚迪宋L的参数", "需要"]

async def main():
    agent = build_agent_singleton()
    print("model:", agent.model)
    history = []
    for i, q in enumerate(TURNS, 1):
        print(f"\n===== t{i}: {q} =====")
        answer, sources = "", []
        async for ev in agent.stream_chat(q, history):
            if ev["type"] == "token":
                answer += ev.get("content", "")
            elif ev["type"] == "source":
                for d in ev.get("documents", []):
                    sources.append((d.get("source",""), d.get("content","")[:80]))
        print("TOOLS:", json.dumps(sources, ensure_ascii=False) or "(none)")
        print("ANSWER:", answer[:400])
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})

asyncio.run(main())
