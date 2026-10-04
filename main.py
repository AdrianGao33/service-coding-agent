"""
入口、主控循环 (REPL)

职责：
1. 维护终端输入 (PromptSession)
2. 解析斜杠命令 (Slash Commands, 如 /help, /clear)
3. 驱动核心 Agent 运行并传递历史对话上下文
4. 汇总与打印每一轮交互的 Token 消耗、API 日志与渲染输出
"""
import asyncio #标准库类似于 threading 的异步协程库，支持 async/await 语法

from prompt_toolkit import PromptSession #代替内置 input()，支持光标移动、历史输入等功能
from pydantic_ai import Agent

# pydantic_graph：PydanticAI 底层依靠图状态机驱动，End 代表图执行终点节点，是pydantic自己写的langchain，每个请求都变成了node，需要end节点来结束
from pydantic_graph import End

# 本地Agent文件夹 核心模块导入单例实例 agent、模型名称以及 API 调用的全局日志 Buffer
from agent import agent, MODEL_NAME, api_call_log

# 本地UI文件夹 命令模块导入相关命令映射、状态类及渲染工具函数
from ui.commands import (
    COMMANDS,
    SessionState,
    console,
    print_part,
    print_divider,
    print_welcome_banner,
)

prompt_session = PromptSession()


def read_user_input():
    """
    打印上下分割线，中间读取用户输入，返回字符串。
    处理 Ctrl+C / Ctrl+D (EOFError, KeyboardInterrupt) 异常退出。
    """
    print_divider() #打印分割线
    try:
        user_input = prompt_session.prompt("❯ ").strip() #拿到用户输入
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    print_divider() #打印分割线
    return user_input 


def handle_command(user_input, state): 
    """
    处理以 / 开头的命令。
    返回 'pass'：不是命令，主循环继续往下走交给 Agent
    返回 'continue'：命令已处理，主循环跳到下一轮
    返回 'break'：命令要求退出主循环
    """
    if not user_input.startswith("/"):
        return "pass"

    # 解析命令名并查找对应的 Command 对象
    cmd_name = user_input[1:].split()[0]
    command = COMMANDS.get(cmd_name)

    if command is None:
        console.print(f"未知命令：/{cmd_name}，输入 /help 查看可用命令\n")
        return "continue"

    # 执行命令的 handler，传入当前 SessionState
    return "continue" if command.handler(state) else "break"


def apply_result(state, result):
    """
    状态同步：跑完一轮 Agent (无论经历了多少次内部节点迭代) 彻底运行结束后，把结果同步到 SessionState 
    """
    state.history = result.all_messages()
    usage = result.usage
    state.input_tokens += usage.input_tokens
    state.output_tokens += usage.output_tokens
    state.last_api_calls = list(api_call_log)
    
async def run_agent_loop(user_input, state):
    """
    核心白盒驱动引擎：
    摒弃黑盒 run_sync()，使用 agent.iter() 逐节点 (Node) 驱动 Agent 图状态机流转。
    实现“边思考、边调工具、边实时输出”的流式效果。
    """
    api_call_log.clear() # 每轮新对话发起前，清空全局 API 调用日志 Buffer
 
    # 开启 Agent 图迭代器上下文，传入当前输入 + 全量历史上下文 (state.history)
    async with agent.iter(user_input, message_history=state.history) as run: #iter是个异步的可迭代对象里面是多个节点（大模型请求和工具调用）
        node = run.next_node
        
        # 只要当前节点不是终点 End，就一直在图状态机里往下走
        while not isinstance(node, End):
            node = await run.next(node)

            if Agent.is_call_tools_node(node):          # 节点类型判断 A：如果是“模型决定调用工具”
                for part in node.model_response.parts:
                    print_part(part)                    # 实时将“正在准备调用的工具及其参数”渲染到终端

            elif Agent.is_model_request_node(node):     # 节点类型判断 B：如果是“准备发起下一次模型请求”
                for part in node.request.parts:
                    if part.part_kind == "tool-return":
                        print_part(part)                # 实时将“工具在本地执行返回的结果”渲染到终端

    apply_result(state, run.result)         # 当循环遇到 End 节点退出后，run.result 会自动结算出最终的 AgentRunResult
    console.print()


def main():
    state = SessionState(model_name=MODEL_NAME) # 初始化会话状态
    print_welcome_banner("Coding Agent")

    while True:
        # 读用户输入
        user_input = read_user_input()
        if user_input is None:
            break
        if not user_input:
            continue

        # 处理 / 开头的命令
        action = handle_command(user_input, state)
        if action == "break":
            break
        if action == "continue":
            continue

        # 核心 Agent 循环：自己驱动节点流转，实时打印每一步
        asyncio.run(run_agent_loop(user_input, state))


if __name__ == "__main__":
    main()
