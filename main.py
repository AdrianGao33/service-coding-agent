"""
入口与主控循环 (REPL)

职责：
1. 界面常驻：启动终端交互框 (Repl)，监听键盘回车提交。
2. 命令拦截：优先解析以 / 开头的指令（如 /help, /resume），避免传给大模型。
3. 白盒图引擎：用 agent.iter() 逐个节点 (Node) 驱动状态机，实现终端实时流式渲染。
4. 结算与落盘：每轮对话结束后，统计 Token 消耗并持久化到本地 Session 文件。
"""
import asyncio # Python 标准异步库，支持 async/await 协程语法

from pydantic_ai import Agent # End 是 PydanticAI 图状态机的终点节点。每个模型请求/工具调用都是一个 Node，遇到 End 循环终止

from pydantic_graph import End

import session

# 从本地 agent 模块导入单例 agent 实例、模型名称、 API 调用日志缓存
from agent import agent, MODEL_NAME, api_call_log

# 本地UI文件夹 命令模块导入相关命令映射、状态类及渲染工具函数
from ui.commands import (
    COMMANDS,
    SessionState,
    console,
    print_part,
    print_welcome_banner,
)

# 导入手搓的终端常驻交互式输入组件 Repl
from ui.input_ui import Repl

async def handle_command(user_input, state) -> str:
    """
    处理以 / 开头的命令。
    返回 'pass'：不是命令，交给 Agent；
    返回 'continue'：命令已处理，进入下一轮；
    返回 'break'：命令要求退出程序。
    """
    if not user_input.startswith("/"):
        return "pass"
    cmd_name = user_input[1:].split()[0]
    command = COMMANDS.get(cmd_name)
    if command is None:
        console.print(f"未知命令：/{cmd_name}，输入 /help 查看可用命令\n")
        return "continue"
    result = command.handler(state)
    # 个别命令（如 /resume）要弹交互式列表，是异步函数，会返回一个coroutine，跑起来前面需要加 await
    if asyncio.iscoroutine(result): 
        result = await result
    return "continue" if result else "break"


def apply_result(state, result):
    """
    状态同步：跑完一轮 Agent (无论经历了多少次内部节点迭代) 彻底运行结束后，把结果同步到 SessionState 
    """
    state.history = result.all_messages()
    usage = result.usage
    state.input_tokens += usage.input_tokens
    state.output_tokens += usage.output_tokens
    state.last_api_calls = list(api_call_log)
    # 把本轮新增的消息追加到会话文件
    session.append_messages(state.session_id, result.new_messages())
    
async def run_agent_loop(user_input, state):
    """
    核心白盒驱动引擎：
    摒弃黑盒 run_sync()，使用 agent.iter() 逐节点 (Node) 驱动 Agent 图状态机流转。
    实现“边思考、边调工具、边实时输出”的流式效果。
    """
    api_call_log.clear() # 每轮新对话发起前，清空全局 API 调用日志 Buffer（只记上次调用）
 
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
                     #异常报错是agent会把报错包成一个“retry-prompt"part,用来给大模型理解错误原因
                    if part.part_kind in ("tool-return", "retry-prompt"): 
                        print_part(part)                # 实时将“工具在本地执行返回的结果”渲染到终端

    apply_result(state, run.result)         # 当循环遇到 End 节点退出后，run.result 会自动结算出最终的 AgentRunResult


async def main():
    state = SessionState(
        model_name=MODEL_NAME,
        session_id=session.new_session_id(),
    )
    print_welcome_banner("my-claude-code")

    # 常驻输入区：输入框整个会话期间不消失
    repl = Repl(state)

    async def on_submit(user_input):
        # 用户每次按下回车提交，均触发此函数

        # 1. 优先拦截并处理斜杠命令
        action = await handle_command(user_input, state)
        if action == "break":
            # 命令要求退出，结束常驻输入区
            repl.exit()
            return
        if action == "continue":
            return

        # 2. 切换 UI 为工作状态（显示思考中动画/指示器）
        repl.start_working()

        # 3. 驱动 Agent 图状态机开始思考与执行
        await run_agent_loop(user_input, state)

    # 启动 Repl 事件循环，等待用户敲击回车唤起 on_submit
    await repl.run(on_submit)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        pass # 优雅响应 Ctrl+C 或 Ctrl+D 退出