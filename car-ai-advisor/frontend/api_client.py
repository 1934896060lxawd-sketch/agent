"""前端 API 客户端 — 封装对 FastAPI 后端的 HTTP 调用。

支持：
- SSE 流式对话（httpx.stream + aiter_lines）
- 非流式 JSON 对话
- 会话 CRUD（创建/列表/重命名/删除/历史）

设计要点：
- 所有方法为 async，通过 run_async() 桥接到 Streamlit 同步上下文
- SSE 解析独立于传输层，每行 "data: {...}" 解析为事件 dict
- Bearer Token 鉴权，API Key 可配置
"""

import asyncio
import json
import logging
import threading
from typing import AsyncGenerator, Any, Coroutine

import httpx

logger = logging.getLogger(__name__)


def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """在独立线程中运行异步协程，避免与 Streamlit Tornado 事件循环冲突。

    Streamlit 底层使用 Tornado，它有自己的 asyncio 事件循环。
    如果直接调用 asyncio.run() 会触发 "event loop already running" 错误。
    本函数在独立线程中创建新事件循环，彻底隔离。

    Usage:
        result = run_async(client.create_session("新对话"))
        sessions = run_async(client.list_sessions())
    """
    result_container: list = []
    error_container: list = []

    def _target():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result_container.append(loop.run_until_complete(coro))
        except Exception as e:
            error_container.append(e)
        finally:
            loop.close()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()

    if error_container:
        raise error_container[0]
    return result_container[0] if result_container else None


class APIClient:
    """FastAPI 后端的 HTTP 客户端。"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "sk-dev-user-001",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ============================================================
    # 对话
    # ============================================================

    async def chat_stream(
        self,
        query: str,
        session_id: str = "default",
    ) -> AsyncGenerator[dict, None]:
        """流式对话 — 消费 SSE 事件流，逐事件 yield。"""
        url = f"{self.base_url}/chat"
        payload = {"query": query, "session_id": session_id, "stream": True}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", url, json=payload, headers=self._headers
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield {
                        "type": "error",
                        "message": f"HTTP {response.status_code}: {error_text.decode()[:200]}",
                    }
                    return

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[len("data: "):]
                        if data_str == "[DONE]":
                            break
                        try:
                            event = json.loads(data_str)
                            yield event
                            if event.get("type") == "done":
                                break
                        except json.JSONDecodeError:
                            logger.warning(f"SSE 解析失败: {data_str[:80]}")
                            continue

    async def chat_sync(
        self, query: str, session_id: str = "default"
    ) -> dict:
        """非流式对话 — 一次性返回完整 JSON 响应。"""
        url = f"{self.base_url}/chat"
        payload = {"query": query, "session_id": session_id, "stream": False}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=self._headers)
            response.raise_for_status()
            return response.json()

    # ============================================================
    # 会话管理
    # ============================================================

    async def create_session(self, title: str = "新对话") -> dict:
        """POST /sessions → 创建新会话。"""
        url = f"{self.base_url}/sessions"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url, json={"title": title}, headers=self._headers
            )
            response.raise_for_status()
            return response.json()

    async def list_sessions(self) -> list[dict]:
        """GET /sessions → 获取当前用户所有会话列表。"""
        url = f"{self.base_url}/sessions"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, headers=self._headers)
            response.raise_for_status()
            data = response.json()
            return data.get("sessions", [])

    async def rename_session(self, session_id: str, new_title: str) -> dict:
        """PATCH /sessions/{id} → 重命名会话。"""
        url = f"{self.base_url}/sessions/{session_id}"
        payload = {"session_id": session_id, "new_title": new_title}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.patch(url, json=payload, headers=self._headers)
            response.raise_for_status()
            return response.json()

    async def delete_session(self, session_id: str) -> bool:
        """DELETE /sessions/{id} → 删除会话。成功返回 True。"""
        url = f"{self.base_url}/sessions/{session_id}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.delete(url, headers=self._headers)
            return response.status_code == 204

    async def get_history(self, session_id: str) -> dict:
        """GET /sessions/{id}/history → 获取会话历史消息。"""
        url = f"{self.base_url}/sessions/{session_id}/history"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, headers=self._headers)
            response.raise_for_status()
            return response.json()
