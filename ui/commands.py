"""
ui/commands.py - 终端 UI 渲染中心与斜杠系统命令处理

职责：
1. 重写 Rich 库渲染规则（将默认居中的 Markdown 标题修复为左对齐）。
2. 消息 Part 格式化：给 Prompt、Thinking、Text、ToolCall 等节点配上颜色图标与美化缩进。
3. 斜杠命令注册：实现 /new, /resume, /status, /api-detail, /help, /exit 命令的处理逻辑。
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

# 从 Rich 库导入终端渲染的核心组件
import questionary #cmdline 交互式选择库，支持上下键选择、回车确认
from rich.markdown import Heading, Markdown
from rich.markup import escape
from rich.padding import Padding
from rich.rule import Rule

import permissions
import session
from .render import console, print_step, print_welcome_banner


class LeftAlignedHeading(Heading):
    """
    rich 默认把 Rich的Markdown 标题渲染成居中对齐，宽终端里看着像错位，覆盖成左对齐。
    """
    def __rich_console__(self, console, options):
        text = self.text
        text.justify = "left"
        yield text


# 全局替换 Markdown 的标题渲染元素
Markdown.elements["heading_open"] = LeftAlignedHeading


# 在main.py会实例化一个用来做会话状态记录
@dataclass
class SessionState:
    """
    跨命令与主循环共享的会话状态类（主循环实例化后作为参数传给各个命令）
    """
    history: list = field(default_factory=list)         # 全量上下文消息历史
    input_tokens: int = 0                               # 本 Session 累计消耗的 input token
    output_tokens: int = 0                              # 本 Session 累计消耗的 output token
    model_name: str = ""
    session_id: str = ""                                # 当前会话 ID（决定写入哪个 jsonl 文件）  
    last_api_calls: list = field(default_factory=list)  # 最近一轮对话触发的 API 调用日志

def _truncate(text, limit: int = 120) -> str:
    """
    截断长文本并做 Rich 字符转义（防止工具参数或报错文本撑爆终端）
    """
    text = str(text).strip()
    text = text if len(text) <= limit else text[:limit] + "..."
    return escape(text)


def _full(text) -> str:
    """
    不截断文本（保留完整思考过程 Thinking 和大模型最终回复 Text）
    """
    return escape(str(text).strip())

# 每个role分配不一样的打印格式
def _format_part_line(part) -> Optional[str]:
    """
    将不同类型的消息 Part 格式化为带有颜色、图标与缩进的 Rich 格式字符串
    """
    kind = part.part_kind

    # 1. 用户输入
    if kind == "user-prompt":
        return f"[cyan]❯ user[/]\n  {_truncate(part.content)}"

    # 2. 模型深度思考过程 (Thinking)
    if kind == "thinking":
        # thinking 整块 dim，弱化视觉权重；不截断，完整保留思考过程
        return f"[dim]✻ thinking[/]\n  [dim]{_full(part.content)}[/]"

    # 3. 模型最终回答 (Text)
    if kind == "text":
        content = (part.content or "").strip()
        if not content:
            return None
        # assistant 是用户最关心的最终回答，完整显示
        return f"[green]● assistant[/]\n  {_full(content)}"

    # 4. 模型申请调用工具 (Tool Call)
    if kind == "tool-call":
        # 命令、路径动辄上百字符，参数放宽到 500 字符再截断
        return f"[yellow]⏺ tool_call[/]\n  [yellow dim]{part.tool_name}({_truncate(part.args, 500)})[/]"

    # 5. 工具返回结果 (Tool Return)
    if kind == "tool-return":
        return f"[magenta]✔ tool_return[/]\n  [magenta dim]{part.tool_name} -> {_truncate(part.content)}[/]"

    # 6. 工具失败重试提示 (Retry Prompt)
    if kind == "retry-prompt":
        # 工具抛 ModelRetry 后，SDK 生成 retry-prompt 把错误反馈给模型
        return f"[yellow]✘ tool_retry[/]\n  [yellow dim]{part.tool_name} -> {_truncate(part.content)}[/]"
    return None

# 特殊情况：llm回复的是Markdown的打印格式
def print_assistant_markdown(content: str) -> None:
    """
    将模型的回答渲染为标准的富文本 Markdown 块（左缩进 2 格）
    """
    console.print("[green]● assistant[/]")
    # Markdown 是块级渲染对象，没法跟在行内前缀后面，所以另起一行渲染；左缩进 2 格和 role 名对齐
    console.print(Padding(Markdown(content), (0, 0, 0, 2)))

# _format_part_line判断转换过后，打印
def print_part(part) -> None:
    """
    渲染单个消息 Part：Text 走 Markdown 块渲染，其他 Part 走单行缩进渲染
    """
    if part.part_kind == "text":
        content = (part.content or "").strip()
        if content:
            print_assistant_markdown(content)
            # 每个 role block 末尾留一个空行，块与块之间不那么挤
            console.print()
        return
    line = _format_part_line(part)
    if line:
        label, _, body = line.partition("\n")
        # body 形如 "  [markup]…"，去掉字面前导 2 空格，交给 print_step 用 Padding 缩进（折行续行也保持缩进）
        print_step(label, body[2:])


# ================= 斜杠命令处理器 (Command Handlers) =================

# Command 命令抽象类，所有斜杠命令都要注册成 Command 对象
@dataclass
class Command:
    name: str
    description: str
    # handler Callable[[入参1类型, 入参2类型, ...], 返回值类型]
    handler: Callable[["SessionState"], bool]

def cmd_exit(state: SessionState) -> bool:
    console.print("再见 👋")
    return False


def cmd_help(state: SessionState) -> bool:
    console.print("可用命令：")
    for cmd in COMMANDS.values():
        console.print(f"  /{cmd.name:<10} {cmd.description}")
    console.print()
    return True


def cmd_new(state: SessionState) -> bool:
    """
    开启新会话：清空历史、token 计数、API 调用记录。
    """
    state.history.clear()
    state.input_tokens = 0
    state.output_tokens = 0
    state.last_api_calls.clear()
    state.session_id = session.new_session_id()
    permissions.state.session_allowed.clear()
    console.print("已开启新会话\n")
    return True

def _summary_line(mtime, prompt: str) -> str:
    """
    格式化菜单选项行：显示 [修改时间 + 首条问题摘要]
    """
    prompt = " ".join(str(prompt).split())
    if len(prompt) > 50:
        prompt = prompt[:50] + "..."
    return f"{mtime:%m-%d %H:%M}  {prompt}"


def cmd_resume(state: SessionState) -> bool:
    """
    恢复历史会话：调用 session.py 扫描磁盘历史，弹 Questionary 交互菜单供用户选择并回放上下文
    """

    # 获取所有会话文件，按修改时间从新到旧返回 (session_id, 修改时间, 首条用户输入) 列表
    sessions = session.list_sessions()
    if not sessions:
        console.print("(当前项目还没有历史会话)\n")
        return True
    
    # 生成 questionary 的选择列表，每行显示修改时间 + 首条用户输入摘要
    choices = [
        questionary.Choice(title=_summary_line(mtime, prompt), value=sid)
        for sid, mtime, prompt in sessions
    ]
    # 交互式选择菜单，用户上下键移动，回车确认
    selected = questionary.select(
        "选择要恢复的会话（上下键移动，回车确认）：", choices=choices
    ).ask()
    
    # 用户按 Ctrl+C 取消选择
    if selected is None:
        return True

    # 1. 替换 SessionState 里的历史消息与 session_id
    state.history = session.load_history(selected)
    state.session_id = selected
    # 权限白名单是会话级的，切换会话后清空
    permissions.state.session_allowed.clear()

    # 2. 重新累加盘存历史 Token 消耗
    # jsonl 里每条模型回复都带 usage，把会话的 token 用量累加回来
    state.input_tokens = sum(
        m.usage.input_tokens for m in state.history if m.kind == "response"
    )
    state.output_tokens = sum(
        m.usage.output_tokens for m in state.history if m.kind == "response"
    )
    # 最近一轮的 API 调用记录只在进程内有效，没法恢复，清空
    state.last_api_calls.clear()

    # 3. 重新把历史消息富文本回放到屏幕上
    console.print(f"\n已恢复会话 {selected[:8]}，共 {len(state.history)} 条消息：\n")
    for msg in state.history:
        for part in msg.parts:
            # 回放和实时输出共用同一套 part 渲染逻辑
            print_part(part)
    console.print()
    return True

def cmd_status(state: SessionState) -> bool:
    console.print(f"模型：           {state.model_name}")
    console.print(f"历史消息条数：    {len(state.history)}")
    console.print(f"累计输入 tokens：{state.input_tokens}")
    console.print(f"累计输出 tokens：{state.output_tokens}\n")
    return True


def cmd_api_detail(state: SessionState) -> bool:
    """
    查看 API 详情：读取最近一轮对话中记录在 api_call_log 里的底层请求与响应元数据
    """
    if not state.last_api_calls:
        console.print("(还没有任何模型调用记录，先发一条消息再来看)\n")
        return True

    console.print(f"最近一轮共发起 {len(state.last_api_calls)} 次 model API 调用\n")

    for i, call in enumerate(state.last_api_calls, 1):
        console.print(f"[bold]Call #{i}[/]")
        console.print(f"  Request:")
        console.print(f"    model:        {call.model}")
        console.print(f"    messages:     {call.messages_count} 条")
        if call.last_part is not None:
            preview = _format_part_line(call.last_part)
            if preview:
                console.print(f"    last_message: {preview}")
        console.print(f"    tools:        {', '.join(call.tools)}")
        console.print(f"  Response:")
        console.print(f"    finish_reason: {call.finish_reason}")
        console.print(f"    parts:         {', '.join(call.parts_kinds)}")
        console.print(f"    usage:         input={call.input_tokens}, output={call.output_tokens}")
        console.print()
    return True


COMMANDS = {
    "new": Command("new", "开启新会话", cmd_new),
    "resume": Command("resume", "恢复历史会话", cmd_resume),
    "status": Command("status", "显示当前会话状态", cmd_status),
    "api-detail": Command("api-detail", "显示最近一轮 model API 调用详情", cmd_api_detail),
    "help": Command("help", "显示可用命令", cmd_help),
    "exit": Command("exit", "退出程序", cmd_exit),
}
