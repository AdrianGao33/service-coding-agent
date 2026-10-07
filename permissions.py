"""
权限管控系统 (Permissions Management System)

职责：
1. 状态管理：维护全局权限模式（default / acceptEdits / bypass）以及会话级放行白名单。
2. 决策计算：根据模式、工具属性与自检结果，计算出放行 ("allow") 或 询问 ("ask")。
3. 当需要询问时：基于 prompt_toolkit 渲染单选 Terminal UI 供用户选择。
"""

from dataclasses import dataclass, field

# 从 prompt_toolkit 导入构建终端交互 UI 所需的组件
from prompt_toolkit.application import Application, in_terminal
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style


# 三种权限模式: default / acceptEdits / bypass
# default 模式：读文件放行，写文件、跑命令等高危操作需审批
DEFAULT = "default"
# acceptEdits 模式：读写文件放行，跑命令等高危操作需审批
ACCEPT_EDITS = "acceptEdits"
# bypass 模式：一切放行，不再询问
BYPASS = "bypass"

# 权限模式按 Shift+Tab 循环切换的顺序
MODES = [DEFAULT, ACCEPT_EDITS, BYPASS]

# 只读工具，任何模式都自动放行（读取不会改动系统，放行没风险）
READONLY_TOOLS = {"read_file"}
# 编辑文件类工具，acceptEdits 模式下自动放行
EDIT_TOOLS = {"write_file"}

# 工具自检注册表字典: key 是工具名 ("run_command")，value 是回调函数 (run_command_self_check)
TOOL_SELF_CHECKS = {}


def register_self_check(tool_name: str, check) -> None:
    """
    提供给各个工具模块的回调注册接口。
    
    Args:
        tool_name: 工具名称 (如 "run_command")
        check: 自检回调函数，接收 args 字典，返回 "ask" (强行要求审批) 或 None (交给通用规则)
    """
    TOOL_SELF_CHECKS[tool_name] = check


@dataclass
class PermissionState:
    """
    进程级权限状态：当前模式，加上本会话的放行白名单。
    权限模式是进程级的：跨 /new、/resume 保持，不写进 jsonl，重启程序才回到 default。
    白名单是会话级的：用 /new、/resume 切换会话时会清空。
    """
    # 当前权限模式
    mode: str = DEFAULT
    # 本会话内点过「不再询问」的工具名，后续直接放行
    session_allowed: set = field(default_factory=set)

# 实例化单例状态对象，供全程序共享
state = PermissionState()


def compute_decision(tool_name: str, args: dict) -> str:
    """
    纯规则判定：在当前模式下，对给定的工具调用返回 "allow" 或 "ask"。
    """
    # bypass 模式：全部放行，连工具自检都不再过问（用户主动选择了这个模式，后果自负）
    if state.mode == BYPASS:
        return "allow"
    # 工具自检：自检要求审批的调用，会话白名单也盖不过
    check = TOOL_SELF_CHECKS.get(tool_name)
    if check and check(args) == "ask":
        return "ask"
    # 本会话点过「不再询问」的工具：放行
    if tool_name in state.session_allowed:
        return "allow"
    # 只读工具：任何模式都自动放行
    if tool_name in READONLY_TOOLS:
        return "allow"
    # acceptEdits 模式：编辑文件放行，命令等其他工具仍要审批
    if state.mode == ACCEPT_EDITS and tool_name in EDIT_TOOLS:
        return "allow"
    # default 模式，或没命中任何放行规则：询问用户
    return "ask"


def cycle_mode() -> str:
    """
    按 default -> acceptEdits -> bypass -> default 循环切换。
    """
    index = MODES.index(state.mode)
    state.mode = MODES[(index + 1) % len(MODES)]
    return state.mode


# 可能装着整份文件内容的参数，预览里只留开头；命令、路径等参数完整展示
BULKY_ARGS = {"content", "old_string", "new_string"}


def _format_call(tool_name: str, args: dict) -> str:
    """
    把一次工具调用渲染成审批预览，例如 run_command(command=npm test)。
    """
    def show(key, value):
        text = " ".join(str(value).split())
        if key in BULKY_ARGS and len(text) > 60:
            return text[:60] + "..."
        return text

    inner = ", ".join(f"{key}={show(key, value)}" for key, value in args.items())
    return f"{tool_name}({inner})"


_STYLE = Style.from_dict({
    "question": "bold",
    "label-current": "#3b82f6 bold",
    "label": "",
    "footer": "#6b7280",
})

# UI样式类
class _ApprovalPicker:
    """
    手绘单选审批 picker，视觉对齐 ask_user_question 的 picker。
    """

    def __init__(self, question: str, options: list[tuple[str, str]]):
        # options 是 (value, label) 列表
        self.question = question
        self.options = options
        self.cursor = 0
        # 选中项的 value；取消时保持 None，run() 据此返回 deny
        self.result = None
        self.app = self._build_app()

    def _render_question(self):
        return FormattedText([("class:question", self.question)])

    def _render_options(self):
        lines: list[tuple[str, str]] = []
        for i, (_value, label) in enumerate(self.options):
            is_cursor = (i == self.cursor)
            pointer = "❯" if is_cursor else " "
            cls_label = "class:label-current" if is_cursor else "class:label"
            lines.append((cls_label, f" {pointer}  {i + 1}. {label}"))
            lines.append(("", "\n"))
        return FormattedText(lines)

    def _render_footer(self):
        return FormattedText([("class:footer", "  ↑↓ 选择 · Enter 确认 · Esc 取消")])

    def _move(self, delta: int):
        self.cursor = (self.cursor + delta) % len(self.options)

    def _build_app(self) -> Application:
        kb = KeyBindings()

        # 堆叠装饰器让一个回调绑定多个键（kb.add 多参数是「按键序列」而非「任选其一」）
        @kb.add("up")
        @kb.add("k")
        def _(event):
            self._move(-1)

        @kb.add("down")
        @kb.add("j")
        def _(event):
            self._move(1)

        @kb.add("enter")
        def _(event):
            self.result = self.options[self.cursor][0]
            self.app.exit()

        @kb.add("escape")
        @kb.add("c-c")
        def _(event):
            self.app.exit()

        # always_hide_cursor 藏掉终端光标，否则会落在左上角压住问句首字
        layout = Layout(HSplit([
            Window(FormattedTextControl(self._render_question), wrap_lines=True, dont_extend_height=True, always_hide_cursor=True),
            Window(FormattedTextControl(self._render_options), dont_extend_height=True, always_hide_cursor=True),
            Window(FormattedTextControl(self._render_footer), height=1, always_hide_cursor=True),
        ]))
        return Application(
            layout=layout,
            key_bindings=kb,
            style=_STYLE,
            full_screen=False,
            mouse_support=False,
            # 选完擦掉整个 picker，滚动区里不留下选项
            erase_when_done=True,
        )

    async def run(self) -> str:
        await self.app.run_async()
        return self.result if self.result is not None else "deny"


async def prompt_approval(tool_name: str, args: dict) -> str:
    """
    工具执行前弹出审批 picker，返回 "once" / "always" / "deny"。
    """
    options = [
        ("once", "允许"),
        ("always", f"允许，且本会话不再询问 {tool_name}"),
        ("deny", "拒绝"),
    ]
    async with in_terminal():
        question = f"是否允许执行 {_format_call(tool_name, args)}？"
        choice = await _ApprovalPicker(question, options).run()
    return choice
