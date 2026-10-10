"""
ui/input_ui.py - 终端常驻交互式输入组件 (REPL UI)

职责：
1. 界面常驻：将输入框固定在屏幕底部，配合 patch_stdout 让 Agent 的输出安全打印在输入框上方。
2. 动画与状态指示：处理思考状态转圈动画 (Spinner)、已耗时计时，以及底部的权限模式提示。
3. 键盘事件响应：绑定 Enter 提交、ESC/Ctrl+C 打断思考、Ctrl+D 退出、Shift+Tab 切换权限模式。
4. 后台任务托管：将 Agent 执行包装为后台 Task，支持任务的实时 Cancel 强行打断。
"""

import asyncio
import time

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.patch_stdout import patch_stdout
from rich.markup import escape

import permissions
from .render import console

# working 指示器的转圈动画帧
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Repl:
    """
    常驻终端输入界面组件：挂载在终端最下方，监听键盘输入，触发 main 注入的 on_submit 回调
    """
    def __init__(self, state):
        self.state = state
        self._on_submit = None  # 由 main 注入的输入处理异步回调
        self._task = None       # 正在后台跑的 Agent Task（用于 ESC/Ctrl+C 打断）
        self.working = False    # Agent 是否正在思考/执行工具（控制 Working 指示器显隐）
        self._work_start = 0.0  # 思考开始时间戳（用于计算耗时秒数）
        self._frame = 0         # 动画当前帧索引
        self._buffer = Buffer(multiline=False, history=InMemoryHistory())
        self.app = self._build_app()

    def _prompt_prefix(self, line_number, wrap_count):
        return HTML("<ansicyan>❯ </ansicyan>")

    def _working_line(self):
        # 渲染 Working 指示器：[转圈动画] Working... (已耗时 Xs · 按 esc 打断)
        frame = _SPINNER[self._frame % len(_SPINNER)]
        elapsed = int(time.monotonic() - self._work_start)
        return HTML(f"<ansigreen>{frame}</ansigreen> <b>Working…</b><ansibrightblack>（已耗时 {elapsed}s · 按 esc 打断）</ansibrightblack>")

    def _mode_line(self):
        # 渲染底部状态栏：显示当前权限模式 (▶▶ default/acceptEdits/bypass) 及切换提示
        mode = permissions.state.mode
        return HTML(f"  <ansimagenta><b>▶▶ {mode}</b></ansimagenta><ansibrightblack>（Shift+Tab 切换）</ansibrightblack>")

    def _divider(self):
        # 渲染 1 字符高的水平横向分割线
        return Window(height=1, char="─", style="fg:ansibrightblack")

    def _build_app(self):
        """
        构建终端 UI 上下五层布局：
        1. ConditionalContainer: Working 思考指示器（仅 working=True 时显示）
        2. 分割线
        3. 文本输入框 (BufferControl)
        4. 分割线
        5. 底部权限模式状态栏
        """
        layout = Layout(
            HSplit(
                [
                    ConditionalContainer(
                        Window(FormattedTextControl(self._working_line), height=1),
                        filter=Condition(lambda: self.working),
                    ),
                    self._divider(),
                    # 输入行只占内容高度，多行自动换行撑开
                    Window(
                        BufferControl(buffer=self._buffer),
                        get_line_prefix=self._prompt_prefix,
                        height=Dimension(min=1),
                        wrap_lines=True,
                        dont_extend_height=True,
                    ),
                    self._divider(),
                    Window(FormattedTextControl(self._mode_line), height=1),
                ]
            )
        )
        return Application(layout=layout, key_bindings=self._build_key_bindings())

    def _build_key_bindings(self):
        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            self._on_enter()

        @kb.add("c-c")
        def _(event):
            # 思考中：打断 Agent 任务；空闲时：退出程序
            if self._task is not None:
                self._task.cancel()
            else:
                event.app.exit()

        @kb.add("escape")
        def _(event):
            # 思考中：打断 Agent 任务；空闲时：清空当前输入框
            if self._task is not None:
                self._task.cancel()
            else:
                self._buffer.reset()

        @kb.add("c-d")
        def _(event):
            # 输入框为空且空闲时：退出程序
            if self._task is None and not self._buffer.text:
                event.app.exit()

        @kb.add("s-tab")
        def _(event):
            # Shift+Tab：快捷切换权限模式 (default -> acceptEdits -> bypass)
            permissions.cycle_mode()

        return kb

    def _on_enter(self):
        # 回车按键响应处理
        if self._task is not None:
            return
        text = self._buffer.text.strip()
        if not text:
            self._buffer.reset()
            return
        # 存进输入历史，清空输入行
        self._buffer.history.append_string(text)
        self._buffer.reset()
        # 把刚敲的用户输入回显到上方历史滚动区
        self._echo_input(text)
        # 启动后台异步 Task 执行 _process，防止阻塞 UI 线程
        self._task = self.app.create_background_task(self._process(text))

    def _echo_input(self, text):
        # 回显用户输入到上方打印区域（用上下分割线包围）
        rule = "─" * console.width
        console.print(f"[bright_black]{rule}[/]")
        console.print(f"[cyan]❯[/] {escape(text)}")
        console.print(f"[bright_black]{rule}[/]")
        console.print()

    async def _process(self, text):
        # 后台 Agent 执行管道：调用 on_submit 并兜底捕获异常与用户打断信号 (CancelledError)
        try:
            await self._on_submit(text)
        except asyncio.CancelledError:
            console.print("\n[bold yellow]已中断[/]\n")
        except Exception as e:
            console.print(f"\n[bold red]✗ {type(e).__name__}: {e}[/]\n")
        finally:
            self._task = None
            self.working = False
            self.app.invalidate()

    def start_working(self):
        # 开启思考状态：显示 Working... 动画并记录起始时间戳
        self.working = True
        self._work_start = time.monotonic()
        self.app.invalidate()

    def exit(self):
        # 退出常驻 UI 循环
        self.app.exit()

    async def run(self, on_submit):
        # 启动 REPL 常驻循环（挂起等待用户交互）
        self._on_submit = on_submit
        # 转圈动画的心跳：请求期间定时重绘
        ticker = asyncio.ensure_future(self._tick())
        try:
            # patch_stdout 让 Agent 的 rich 输出打印在输入框上方而不是冲掉它
            with patch_stdout(raw=True):
                await self.app.run_async()
        finally:
            ticker.cancel()

    async def _tick(self):
        try:
            while True:
                await asyncio.sleep(0.1)
                if self.working:
                    self._frame += 1
                    self.app.invalidate()
        except asyncio.CancelledError:
            pass
