from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import subprocess
import os

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

@tool
def run_bash(command: str) -> str:
    """执行bash终端命令，仅用于查询信息
    Args:
        command: 需要执行的shell命令字符串
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

# 工具列表直接传入
TOOLS = [run_bash]
llm_with_tools = llm.bind_tools(TOOLS)


def agent_loop(messages: list):
    history = []
    history.append(SystemMessage(content=SYSTEM))
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))

    while True:
        response = llm_with_tools.invoke(history)
        history.append(response)

        if not response.tool_calls:
            res_history = [m.model_dump() for m in history]
            return res_history

        tool_msg_list = []
        for call in response.tool_calls:
            tool_name = call["name"]
            args = call["args"]
            call_id = call["id"]

            if tool_name == "run_bash":
                cmd = args["command"]
                print(f"\033[33m$ {cmd}\033[0m")
                result = run_bash(cmd)
                print(result[:200])
                tool_msg_list.append(ToolMessage(content=result, tool_call_id=call_id))

        history.extend(tool_msg_list)


if __name__ == "__main__":
    print("s01: Agent Loop")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        # 接收agent返回的全新完整对话历史
        full_history = agent_loop(history)
        # 把最新的对话同步回全局history，保存上下文记忆
        history = full_history

        # 取出最后一条assistant回答并打印文本内容
        last_msg = history[-1]
        final_text = last_msg["content"]
        print(f"\033[32mAI: {final_text}\033[0m")
        print()