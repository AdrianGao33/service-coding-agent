"""
入口、主控循环 (REPL)

职责：
1. 维护终端输入 (PromptSession)
2. 解析斜杠命令 (Slash Commands, 如 /help, /clear)
3. 驱动核心 Agent 运行并传递历史对话上下文
4. 汇总与打印每一轮交互的 Token 消耗、API 日志与渲染输出
"""

from prompt_toolkit import PromptSession #代替内置 input()，支持光标移动、历史输入等功能

# 本地Agent文件夹 核心模块导入单例实例 agent、模型名称以及 API 调用的全局日志 Buffer
from agent import agent, MODEL_NAME, api_call_log

# 本地UI文件夹 命令模块导入相关命令映射、状态类及渲染工具函数
from ui.commands import (
    COMMANDS,
    SessionState,
    console,
    print_agent_steps,
    print_divider,
    print_welcome_banner,
)

prompt_session = PromptSession()


def read_user_input():
    """
    打印上横线并读一行用户输入；回车后再补一条下横线，让输入在滚动历史里保持上下边界。返回 None 表示用户希望退出（Ctrl-C / Ctrl-D）。
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
    跑完一轮 Agent 后，把结果同步到 SessionState 并显示新增的中间过程。
    """
    state.history = result.all_messages()
    usage = result.usage
    state.input_tokens += usage.input_tokens
    state.output_tokens += usage.output_tokens
    state.last_api_calls = list(api_call_log)
    
    # result.new_messages() 直接拿到这一轮新增的 message，不需要手动算偏移
    print_agent_steps(result.new_messages())


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

        # 核心 Agent 循环：清空收集 buffer，跑一轮，把结果应用到 state
        api_call_log.clear()

        # - 同步调用 Agent，传入当前用户 Prompt 以及之前的全部对话历史上下文
        result = agent.run_sync(user_input, message_history=state.history)

        # - 将 Agent 的响应、新增 Token 和中间步骤同步到 state 并打印到终端
        apply_result(state, result)


if __name__ == "__main__":
    main()
