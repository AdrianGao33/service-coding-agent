"""
两件事共用同一个 fork agent：模型和 instructions 与主 agent 相同，但只挂文件工具，写入被闸门限制在记忆目录内。 todo

后台自动化管理与沉淀引擎：每轮对话结束后 fork 一份对话提炼值得保存的记忆，并定期合并整理记忆文件。

核心职责与工作机制：
1. 独立的 Fork Agent（分身代理）：
   - 克隆主 Agent 的模型和系统指令（INSTRUCTIONS），但剥离了无关工具，只挂载文件操作三件套（read_file, edit_file, write_file）。
   - 通过钩子函数（Hooks）进行严格的权限关卡限制：只准读写项目的 .memory/ 记忆目录，绝对不准碰用户工程源码。

2. 两大核心后台任务：
   - 每轮对话后提炼（Extract）：检查这一轮对话有没有漏掉的干货，自动生成或更新记忆文件。
   - 定期记忆整理/梦境模式（Dream）：满足“距离上次超24小时 且 期间活跃了至少5个会话”的双重门槛闸门后，自动对所有记忆文件进行盘点、合并、消重和修剪 MEMORY.md 索引。

3. 异步非阻塞调度（Fire-and-Forget）：
   - 采用 asyncio 异步后台任务调度，绝对不卡顿主对话的流畅度；退出时支持 drain 收尾，防止写文件写到一半被强制杀掉。
"""

import asyncio
import time
from pathlib import Path

from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import Hooks
from pydantic_ai.usage import UsageLimits

import session
from agent.core import INSTRUCTIONS, model
from agent.deps import AgentDeps
from agent.file_state import ReadFileState
from agent.tools.file import edit_file, read_file, write_file
from ui.render import print_step

from . import store

# 后台任务单次运行最多允许的模型请求次数，防止跑飞空转
MAX_FORK_REQUESTS = 10

# 自动合并的两道闸门：距上次合并至过了多少小时，且期间活跃了至少多少个会话
DREAM_MIN_HOURS = 24
DREAM_MIN_SESSIONS = 5

fork_hooks = Hooks()

# 单独给fork_agent挂的工具调用操作的前置检查hook
@fork_hooks.on.tool_execute
async def _memory_write_gate(ctx, *, call, tool_def, args, handler):
    """
    【安全写入闸门】：确保后台 Agent 绝不碰用户工程源码。
    规则：读文件（read_file）全行业放行；写/改文件（write_file/edit_file）必须严格落在记忆目录内，其余一律拒绝。
    注意：后台是静默运行的，没有前端弹窗给用户审批，因此拒绝就是唯一的防线。
    """
    if call.tool_name == "read_file":
        return await handler(args)
    if call.tool_name in ("write_file", "edit_file") and store.is_memory_path(str(args.get("path", ""))):
        return await handler(args)
    return f"后台记忆 agent 只允许写记忆目录内的文件，这次 {call.tool_name} 调用被拒绝。"


# fork agent：instructions 和主 agent 是同一份（包含记忆系统约定），工具只挂文件三件套
fork_agent = Agent(
    model,
    instructions=INSTRUCTIONS,
    tools=[read_file, Tool(edit_file, sequential=True), Tool(write_file, sequential=True)],
    deps_type=AgentDeps,
    capabilities=[fork_hooks],
)

# 初始化一个AgentDeps，readFileState新的，task用不着为none
def _fork_deps() -> AgentDeps:
    # fork 一次性运行，用全新的 readFileState；不挂 task 工具，tasks_store 留空
    return AgentDeps(read_file_state=ReadFileState(), tasks_store=None)

# 记忆目录清单大str
def _manifest_section() -> str:
    """
    把现有记忆清单预注入 prompt，省去 fork 自己列目录的回合。
    """
    headers = store.scan_memory_files()
    if not headers:
        return ""
    return (
        "\n## 现有记忆文件\n\n"
        + store.format_manifest(headers)
        + "\n\n写之前先对照这份清单——更新已有文件，而不是新建重复的。"
    )

#【成果统计】：检查后台 Agent 这次运行中到底写了哪些记忆文件名，用于最后在终端打印提示。
def _written_memory_files(messages) -> list[str]:
    """
    收集一次运行里写过的记忆文件名（索引 MEMORY.md 不算），用于完成提示。
    """
    written = []
    for message in messages:
        for part in message.parts:
            if getattr(part, "part_kind", "") != "tool-call":
                continue
            if part.tool_name not in ("write_file", "edit_file"):
                continue
            path = str(part.args_as_dict().get("path", ""))
            if store.is_memory_path(path):
                name = Path(path).name
                if name != "MEMORY.md" and name not in written:
                    written.append(name)
    return written


# ---------- 第一部分：提炼（每轮对话结束后复查有没有漏记的）----------

_EXTRACT_PROMPT = """你现在是记忆提取子代理。分析上面的对话，用其中的内容更新你的持久记忆。

你只能使用对话里出现的内容来更新记忆。不要花任何回合去进一步调查验证——不要读源码确认某个模式是否存在，也不要跑命令。
你的回合数有限。edit_file 之前必须先 read_file 同一个文件，所以高效的策略是：第一轮把可能要更新的文件并行读完，第二轮并行写入，不要读写交错拖回合。
该记什么、不该记什么、怎么保存，遵循 system prompt 里记忆系统的约定；没有值得保存的内容就直接回复「无需保存」，不要为了保存而保存。
{manifest}"""

# 驱动fork agent 审视对话历史，提取有价值的记忆并写入磁盘，随后在终端打印日志。
async def _extract(history: list) -> None:
    """
    fork 对话让 agent 提炼记忆，结束后回显写了哪些记忆文件。
    """
    result = await fork_agent.run(
        _EXTRACT_PROMPT.format(manifest=_manifest_section()), # 在run下，第一个位置参数是字符串就行，会当做一个user prompt处理
        message_history=history,
        deps=_fork_deps(),
        usage_limits=UsageLimits(request_limit=MAX_FORK_REQUESTS),
    )
    written = _written_memory_files(result.new_messages())
    if written:
        print_step("[cyan]✦ memory[/]", f"[cyan dim]后台提炼保存了记忆：{'、'.join(written)}[/]")


# ---------- 第二部分：Dream（定期合并整理记忆）----------

_DREAM_PROMPT = """# Dream：记忆整理

现在做一次 dream——对你的记忆文件做一轮回顾整理，把最近积累的东西沉淀成持久、组织良好的记忆，让未来的会话能快速进入状态。

记忆目录：{memory_dir}
目录已经存在，直接写即可，不要 mkdir。

---

## 第一阶段：盘点

- 读 MEMORY.md，了解当前的索引
- 对照下面的清单，粗读可能重叠或过期的记忆文件——之后是改进它们，而不是另写重复的

现有记忆文件清单：
{manifest}

## 第二阶段：收集

找出值得处理的信号：
- 同主题分散在多个文件的记忆
- 互相矛盾的记忆
- 和当前事实冲突、明显过期的记忆

## 第三阶段：合并

对每个值得保留的主题，写入或更新记忆文件。重点：
- 把新信号合并进已有主题文件，而不是新建近似重复的文件；同主题只是场景不同的多条记忆，也收进同一个文件分条列出（比如三条 commit message 相关的约定合并成一条「commit 规范」）
- 「昨天」「下周」这类相对日期换算成绝对日期，时间过去之后依然可读
- 删除被证伪的事实——当下的观察推翻了旧记忆，就从源头改掉
- 被合并掉的旧文件用 write_file 清空（空文件不再参与召回）

## 第四阶段：修剪

更新 MEMORY.md，让它保持在 200 行、25KB 以内。它是索引，不是内容堆放处——每条一行、150 字符以内：`- [标题](文件名.md) — 一句话钩子`。绝不要把记忆内容直接写进去。
- 删掉指向已过期、已清空、已被取代记忆的行
- 超过 200 字符的索引行说明内容装错了地方：缩短这行，细节挪进记忆文件
- 给新的重要记忆补上指向行
- 两个文件说法冲突时，修正错的那个

---

最后简短总结这次合并了什么、更新了什么、修剪了什么。如果记忆本来就很紧凑没有变化，直接说明。
"""


def _lock_path() -> Path:
    """
    返回记录上次 Dream 整理时间的锁文件路径（.consolidate-lock）。
    """
    return store.memory_dir() / ".consolidate-lock"


def _dream_due(current_session_id: str) -> bool:
    """
    【闸门检查】：判断是否满足定期整理（Dream）的条件。
    条件 1：距离上次整理已过去至少 24 小时（通过锁文件的修改时间 mtime 判断）。
    条件 2：自上次整理以来，项目下修改过（touched）的其他会话文件jsonls数量达到或超过 5 个（当前会话排除）。
    """
    lock = _lock_path()
    last = lock.stat().st_mtime if lock.exists() else 0.0
    if time.time() - last < DREAM_MIN_HOURS * 3600:     # 距离上次未满 24 小时，不触发
        return False
    if not session.project_dir().exists():              # 如果项目目录不存在，跳过
        return False
    # 数一数上次合并之后动过的会话文件，当前会话不算——它还在进行中
    touched = [
        p for p in session.project_dir().glob("*.jsonl")
        if p.stem != current_session_id and p.stat().st_mtime > last    #st是stat + m是modification + time 修改时间
    ]
    return len(touched) >= DREAM_MIN_SESSIONS


async def dream(history: list, *, force: bool = False, session_id: str = "") -> None:
    """
    【Dream 入口】：执行记忆大扫除。
    - 如果不是强制执行（force=False），会先过 _dream_due 闸门。
    - 借助 fork_agent 运行复杂的整理 Prompt，最后更新锁文件的修改时间。
    """
    if not force and not _dream_due(session_id):
        return
    headers = store.scan_memory_files()
    if not headers:
        if force:
            print_step("[cyan]✦ memory[/]", "[cyan dim]还没有任何记忆，无需整理[/]")
        return
    
    # 干活的还是同一个 fork agent，只是这次传给 run() 的新任务换成了 dream 整理指令
    result = await fork_agent.run(
        _DREAM_PROMPT.format(memory_dir=store.memory_dir(), manifest=store.format_manifest(headers)),
        message_history=history,
        deps=_fork_deps(),
        usage_limits=UsageLimits(request_limit=MAX_FORK_REQUESTS),
    )
    # touch 锁文件，把这次合并的时间记在 mtime 上
    _lock_path().touch()            #创建lock空文件
    print_step("[cyan]✦ memory[/]", f"[cyan dim]{result.output}[/]")


# ---------- 第三部分：调度与生命周期管理 ----------

# 用一个全局集合持有正在运行的后台任务引用，防止被 Python 垃圾回收机制提前销毁/取消
_background_tasks: set = set()


def schedule(state, new_messages) -> None:
    """
    【后台调度总入口】：在每轮对话结束后以 Fire-and-Forget（发射后不管）方式触发。
    - 如果上一轮的后台任务还在跑，直接跳过，防止并发写文件导致冲突。
    - 异步创建后台任务，不阻塞用户继续输入。
    """
    # 上一轮的后台任务还没跑完就跳过这一轮：两个提炼并发读同一份清单，会写出重复记忆
    if any(not t.done() for t in _background_tasks):
        return
    task = asyncio.create_task(
        _run_background(list(state.history), list(new_messages), state.session_id)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)       #任务结束自我删除


async def drain(timeout: float = 60) -> None:
    """
    【退出收尾】：程序退出/exit ctrl+c前调用，给尚未跑完的后台记忆任务最多留出 timeout 秒收尾时间，防止写记忆文件写到一半进程被强杀。
    """
    tasks = [t for t in _background_tasks if not t.done()]
    if not tasks:
        return
    print_step("[dim]◇ memory[/]", "[dim]等待后台记忆任务收尾（Ctrl+C 强退）...[/]")
    await asyncio.wait(tasks, timeout=timeout)


async def _run_background(history: list, new_messages: list, session_id: str) -> None:
    """
    【后台异步执行流水线】：
    1. 检查主对话这一轮是否已经自己动手写过记忆了（has_memory_writes）。如果写过了，后台就不重复提炼。
    2. 顺便跑一下 dream 检查，看是否达到了定期整理的门槛。
    3. 异常兜底：后台任务报错绝不能影响主流程，捕获后打印一行警告即可。
    """
    try:
        # 主对话这轮已经自己写过记忆，就不再重复提炼
        if not store.has_memory_writes(new_messages):
            await _extract(history)
        await dream(history, session_id=session_id)
    except Exception as e:
        # 后台任务失败不打扰主流程，提示一行即可
        print_step("[yellow]⚠ memory[/]", f"[yellow]后台记忆任务出错：{type(e).__name__}: {e}[/]")
