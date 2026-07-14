"""
SSE（Server-Sent Events）流式输出工具。

SSE 协议格式:
  data: <json_payload>\n\n

每一帧以 "data: " 开头（注意空格），以 "\n\n" 结尾。
两个连续的换行符是 SSE 协议的消息分隔符。

协议参考: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
"""

import json
import logging
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)


def format_sse(data: dict) -> str:
    """将字典格式化为一条符合 SSE 协议的消息。

    示例:
        >>> format_sse({"type": "token", "content": "你好"})
        'data: {"type": "token", "content": "你好"}\\n\\n'

        >>> format_sse({"type": "source", "documents": [...]})
        'data: {"type": "source", "documents": [...]}\\n\\n'

    Args:
        data: 要发送的事件字典

    Returns:
        SSE 格式的字符串，可直接 yield 给 StreamingResponse
    """
    json_str = json.dumps(data, ensure_ascii=False)
    return f"data: {json_str}\n\n"


async def sse_generator(
    generator: AsyncGenerator[dict, None],
    done_event: Optional[dict] = None,
) -> AsyncGenerator[str, None]:
    """将 dict 异步生成器包装为 SSE 格式的字符串生成器。

    这是连接 "LLM token 生成器" 和 "FastAPI StreamingResponse" 的桥梁:

        LLM API Stream  →  {"type":"token","content":"...yc"}  (dict)
                ↓  sse_generator 包装
        StreamingResponse  →  'data: {"type":"token",...}\n\n'  (str)

    错误处理策略:
      - 如果生成器中途抛异常，发送 {"type":"error", ...} 事件
      - 不抛异常给 FastAPI（否则客户端收到 500 而非部分内容）
      - generated_count 在 try 外部定义，防止 CancelledError 时 NameError

    Args:
        generator: 产生 dict 事件的异步生成器
        done_event: 流正常结束时发送的完成事件，默认 {"type": "done"}

    Yields:
        SSE 格式的字符串
    """
    # 【关键】generated_count 在 try 外部声明
    # 如果客户端断开连接（CancelledError），try 块中的局部变量会被销毁
    # 放在外部保证异常处理时能安全访问
    generated_count = 0

    try:
        async for event in generator:
            yield format_sse(event)
            generated_count += 1

        # 正常结束，发送 [DONE] 信号
        if done_event is None:
            done_event = {"type": "done"}
        done_event["total_tokens"] = generated_count
        yield format_sse(done_event)

        logger.info(f"SSE 流完成: {generated_count} 个事件")

    except Exception as exc:
        # 生成器内部异常 → 发送 error 事件，不抛给 FastAPI
        # 这样客户端至少能收到已生成的内容 + 错误信息
        logger.error(f"SSE 流异常 (已生成 {generated_count} 个事件): {exc}")
        error_event = {
            "type": "error",
            "message": str(exc),
            "generated_tokens": generated_count,
        }
        yield format_sse(error_event)
