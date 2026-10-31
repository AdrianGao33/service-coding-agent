"""
入口与主控循环 (REPL)

职责：
1. 界面常驻：启动终端交互框 (Repl)，监听键盘回车提交。
2. 命令拦截：优先解析以 / 开头的指令（如 /help, /resume），避免传给大模型。
3. 白盒图引擎：用 agent.iter() 逐个节点 (Node) 驱动状态机，实现终端实时流式渲染。
4. 结算与落盘：每轮对话结束后，统计 Token 消耗并持久化到本地 Session 文件。
5. 引用文件预处理：@文件，预读，更新session的read_file_state
"""
import asyncio # Python 标准异步库，支持 async/await 协程语法

from pydantic_ai import Agent # End 是 PydanticAI 图状态机的终点节点。每个模型请求/工具调用都是一个 Node，遇到 End 循环终止
from pydantic_graph import End

import compact
import mcp_servers
import session
from agent import agent, MODEL_NAME, api_call_log
from agent.deps import AgentDeps
from agent.reminders import build_job_reminder_text
from background_jobs import JobRegistry
from file_history import FileHistory
from memory import background, recall, store
from tasks_store import TasksStore

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
from mentions import build_mention_messages, extract_at_mentions

async def handle_command(user_input, state) -> str:
    """
    处理以 / 开头的命令。
    返回 'pass'：不是命令，交给 Agent；
    返回 'continue'：命令已处理，进入下一轮；
    返回 'break'：命令要求退出程序。
    """
    if not user_input.startswith("/"):
        return "pass"
    cmd_name, _, args = user_input[1:].partition(" ")
    command = COMMANDS.get(cmd_name)
    if command is None:
        console.print(f"未知命令：/{cmd_name}，输入 /help 查看可用命令\n")
        return "continue"
    # 声明接收参数的命令（如 /compact），把命令名后面的整段文本传给它
    if command.takes_args:
        result = command.handler(state, args.strip())
    else:
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

def inject_at_mentions(user_input, state):
    """
    解析用户输入里的 @path，把每个被引用的文件伪装成一次 read_file 调用，在 Agent 跑起来之前塞进对话历史。
    """
    paths = extract_at_mentions(user_input) #提取@后面的路径
    if not paths:
        return
    mention_messages = build_mention_messages(paths, state.read_file_state) # 2. 读取文件、登记 ReadFileState 账本，并构建成对的伪造消息 (ModelResponse + ModelRequest)
    if not mention_messages:
        return
    # 塞进历史上下文：模型下一轮就能看到这些「读文件」记录，以为是自己读的
    state.history += mention_messages
    # 更新jsonl持久化，/resume 恢复会话时能连同引用进来的文件内容一起还原
    session.append_messages(state.session_id, mention_messages)
    # 终端里也回显一下注入了哪些文件，让用户看到 @ 引用确实生效了
    for msg in mention_messages:
        for part in msg.parts:
            print_part(part)


async def run_agent_loop(user_input, state):
    """
    核心白盒驱动引擎：
    摒弃黑盒 run_sync()，使用 agent.iter() 逐节点 (Node) 驱动 Agent 图状态机流转。
    实现“边思考、边调工具、边实时输出”的流式效果。
    """
    api_call_log.clear() # 每轮新对话发起前，清空全局 API 调用日志 Buffer（只记上次调用）

   # deps 把本会话的 readFileState 和 tasksStore 一起打包成 AgentDeps 注入工具层，工具内通过 ctx.deps.* 访问
    deps = AgentDeps(
        read_file_state=state.read_file_state,
        tasks_store=state.tasks_store,
        file_history=state.file_history,
        job_registry=state.job_registry,
    )
 
    # 开启 Agent 图迭代器上下文，传入当前输入 + 全量历史上下文 (state.history) + mcp工具
    async with agent.iter(user_input, message_history=state.history, deps=deps,
                          toolsets=mcp_servers.active_toolsets(),
    ) as run: #iter是个异步的可迭代对象里面是多个节点（大模型请求和工具调用）
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
    return run.result                       # 把本轮结果交回给调用方，后台记忆提炼要用本轮新增的消息判断该不该跑

async def main():
    session_id = session.new_session_id()
    # 启动时建好记忆目录，system prompt 里承诺过「目录已存在」，模型就不必浪费回合去确认
    store.ensure_memory_dir()
    # 本会话的 TasksStore 落盘在 ~/.my-code-agent/tasks/<session_id>/，会话级隔离
    tasks_store = TasksStore(session_id=session_id)
    state = SessionState(
        model_name=MODEL_NAME,
        session_id=session_id,
        tasks_store=tasks_store,
        file_history=FileHistory(session_id=session_id),
        job_registry=JobRegistry(session_id=session_id),
    )
    print_welcome_banner("my-code-agent")

    # 转圈提示连接进度，否则冷启动拉包时用户会以为卡死了
    with console.status("正在连接 MCP server..."):
        mcp_summary = await mcp_servers.startup()
    if mcp_summary:
        console.print(mcp_summary + "\n")

    # 常驻输入区：输入框整个会话期间不消失
    repl = Repl(state)

    # =========================================================================
    # 1. 用户主动驱动轮：on_submit
    # 【触发场景】用户在 REPL 终端按下回车键提交输入时触发。
    # 【生命周期】按需触发（即点即跑），走完整的前置用户意图解析流程。
    # =========================================================================

    async def on_submit(user_input):
        # 用户每次按下回车提交，均触发此函数

        # 1. 拦截并处理斜杠命令（如 /exit, /clear）
        action = await handle_command(user_input, state)
        if action == "break":
            # 命令要求退出，结束常驻输入区
            repl.exit()
            return
        if action == "continue":
            return

        # 2. 检查上下文水位，超限自动压缩
        await compact.auto_compact_if_needed(state)

        # 3. 创建回退检查点：此刻 @ 引用和记忆召回还没注入，len(history) 就是干净的回退下标
        state.file_history.make_checkpoint(len(state.history), user_input)

        # 4. 解析 @ 引用，把被引用的文件伪装成 read_file 调用塞进历史
        inject_at_mentions(user_input, state)

        # 2. 切换 UI 为工作状态（显示思考中动画/指示器）
        repl.start_working()

        # 4. 专属流程：召回长效记忆，让模型带着历史经验处理本轮输入
        await recall.inject_memories(user_input, state)

        # 5. 跑主循环 Agent 图
        result = await run_agent_loop(user_input, state)

        # 6. 本轮结束后由后台提炼记忆
        background.schedule(state, result.new_messages())

    # =========================================================================
    # 2. 系统后台驱动轮：on_system
    # 【触发场景】由后台任务监控哨兵（watch_jobs）在空闲时检测到后台 Job 完成后自主触发。
    # 【生命周期】异步事件驱动，无需用户干预，让 Agent 实现“自主向用户汇报”。
    # =========================================================================
    async def on_system(text):
        # 系统触发的对话轮（后台 job 完成通知）：跳过检查点、@ 解析和记忆召回，直接跑 Agent
        await compact.auto_compact_if_needed(state)
        repl.start_working()
        result = await run_agent_loop(text, state)
        background.schedule(state, result.new_messages())

    # =========================================================================
    # 3. 后台轮询哨兵：watch_jobs
    # 【触发场景】以常驻后台 Task 的形式存在，每隔 1 秒循环检查一次。
    # 【核心职能】扮演“雷达”角色。如果 REPL 正处于空闲（is_idle），且后台有刚跑完的 Job，
    #            就自动调用 `repl.submit_system(text)` 强行塞入系统汇报输入。
    # =========================================================================
    async def watch_jobs():
        # 每隔一秒轮询查看一下repl，job 在 REPL 空闲时完成的话，把通知作为一轮系统输入提交，让模型自己开口汇报
        try:
            while True:
                await asyncio.sleep(1)          # 每秒醒来一次，零 CPU 压力
                if not repl.is_idle:
                    continue
                text = build_job_reminder_text(state.job_registry)
                if text:
                    repl.submit_system(text)    # 触发系统自主对话轮
        except asyncio.CancelledError:
            pass        # 优雅捕获退出取消信号

    # 挂载后台哨兵 Task 到事件循环       
    job_watcher = asyncio.ensure_future(watch_jobs())
    try:
        await repl.run(on_submit, on_system) # 进入 REPL 主交互死循环（接管前后端输入与系统汇报）
    finally:
        # 安全网：无论正常退出还是异常报错，必须注销后台哨兵任务，防止产生孤儿协程
        job_watcher.cancel()

    # 退出前给后台记忆任务留收尾时间，提炼写到一半被杀会留下半截记忆
    await background.drain()
    # 后台 job 是真进程，不等待、直接终止，不让它们活过本次会话
    killed = state.job_registry.shutdown()
    if killed:
        console.print(f"[dim]退出时终止了 {killed} 个还在运行的后台 job[/]")
    await mcp_servers.shutdown()    # 关闭 MCP 服务连接管道

if __name__ == "__main__":
    try:
        # 贯通全局的事件循环开启
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        pass # 优雅响应 Ctrl+C 或 Ctrl+D 退出