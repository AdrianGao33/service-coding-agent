"""
本文件是 Shell 工具交互层 (`tools/shell.py`)，负责为 Agent 提供执行本地命令行及管理后台任务的核心能力。
它的核心职责与功能范围包括：
1. 增强型 run_command 工具：支持通过 `run_in_background` 参数将命令直接放入后台，或在前台限时运行（默认 10 秒超时）。
2. 前后台动态转换桥梁：即使默认作为前台运行的命令，如果用户通过快捷键（如 Ctrl+B）介入，也能无缝将其转入后台。
3. 孤儿进程防御：配合 asyncio 的取消机制（CancelledError），当用户在中途按 Esc 打断对话时，自动清理正在前台跑的进程。
4. job_kill 工具：允许 Agent 根据 Job ID 安全终止指定的后台任务及其整个进程树，并结合 ModelRetry 处理非法 ID 输入。
"""
import asyncio
import re
import shlex
import time

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry

import permissions

from ..deps import AgentDeps

FOREGROUND_TIMEOUT = 10


async def run_command(
    ctx: RunContext[AgentDeps], command: str, run_in_background: bool = False
) -> str:
    """
    执行一条 shell 命令。默认前台运行，直接返回输出。

    长驻或耗时命令（dev server、长测试）用 run_in_background=True，且只在不需要立刻拿到结果时使用：
    调用立即返回 job id（命令末尾不需要加 &），job 结束后会收到 <task-notification> 通知，不要主动轮询等待；
    期间随时可以用 read_file 读它的日志文件查看已有输出。

    Args:
        command: 要执行的 shell 命令
        run_in_background: 作为后台 job 运行，不等待它结束
    """

    # 从 Pydantic-AI 的依赖注入上下文（ctx.deps）中获取会话专属的 Job 注册表
    registry = ctx.deps.job_registry

    # 架构设计亮点：
    # 无论是前台命令还是后台命令，统一先经由注册表通过 spawn_shell 启动进程。
    # 这样做的妙处在于：即使命令一开始是以「前台」运行的，用户随时可以按下 Ctrl+B 将其转入后台，底层无需重新拉起进程。
    job = await registry.spawn_shell(command, background=run_in_background)

    # 如果明确指定了在后台运行，直接封装并返回后台通知指引，不阻塞主流程
    if run_in_background:
        return _background_notice(job, f"已放入后台运行，job id: {job.id}")

    # --- 前台命令限时等待逻辑 ---
    start = time.monotonic()
    try:
        # 非阻塞轮询：只要任务还在跑，且用户没有通过快捷键把它转为后台（job.background 仍为 False），每 50 毫秒轮询 job 的状态
        while job.status == "running" and not job.background:
            # 检查是否超过前台最大容忍时间（10秒）
            if time.monotonic() - start > FOREGROUND_TIMEOUT:
                # 超时则强行杀死该任务及其进程树
                registry.kill(job.id)
                return f"[错误] 命令执行超时（{FOREGROUND_TIMEOUT}秒）；长驻或耗时命令请改用 run_in_background=True 重新执行"
            # 极短的异步休眠，让出事件循环，避免 CPU 空转
            await asyncio.sleep(0.05)
            
    except asyncio.CancelledError:
        # 异常防御：当用户按下 Esc 键打断本轮对话时，顺手将前台命令终止，防止留下无人管理的孤儿进程
        registry.kill(job.id)
        raise

    # 如果在等待期间，用户按了 Ctrl+B 将该前台任务转为了后台运行
    if job.background:
        return _background_notice(job, f"用户把这条命令转入了后台，它作为 job {job.id} 继续运行")

    # --- 前台命令在 10 秒内顺利执行完毕的处理 ---
    # 从落盘的日志文件中读取命令输出
    output = job.log_path.read_text(errors="replace")
    
    # 如果进程退出码不为 0，追加错误提示
    if job.returncode != 0:
        output += f"\n[错误] exit code {job.returncode}"
        
    return output or "(无输出)"


def _background_notice(job, lead: str) -> str:
    """
    辅助函数：
    生成统一的后台任务指引说明文本，供「主动放后台」和「Ctrl+B 转后台」共同复用。
    明确告诉 Agent 日志路径以及如何等待异步通知，防止模型原地傻等。
    """
    return (
        f"{lead}\n"
        f"输出日志：{job.log_path}（随时可用 read_file 查看）\n"
        "完成后你会收到 <task-notification> 通知，不要原地等待。"
    )


def job_kill(ctx: RunContext[AgentDeps], job_id: str) -> str:
    """
    终止一个正在运行的后台 job（它的整个进程树都会被终止）。

    Args:
        job_id: 要终止的 job id（如 "b3k9f02qx"）
    """
    registry = ctx.deps.job_registry
    job = registry.get(job_id)
    
    # 鲁棒性校验：如果 Agent 传入了一个根本不存在的 job_id
    if job is None:
        # 抛出 Pydantic-AI 的 ModelRetry 异常，提示模型修正参数重新调用
        raise ModelRetry(f"job {job_id} 不存在")
        
    # 如果任务已经处于结束状态，直接返回提示，不重复做杀死动作
    if not registry.kill(job_id):       #在判断语句中调用了kill
        return f"job {job_id} 已经结束（{job.status}），无需终止"
        
    return f"job {job_id} 已终止"

# 高危命令的特征：删除文件、提权、直写磁盘
DANGEROUS_PATTERNS = [
    r"\brm\b",          # 删除文件/目录指令 (如 rm -rf)
    r"\bsudo\b",        # 提权执行指令
    r"\bdd\b",          # 底层磁盘/块设备直接写指令
    r"\bmkfs\w*\b",     # 格式化文件系统指令 (如 mkfs, mkfs.ext4)
]

# 白名单：严格认可的保守只读命令列表
READONLY_COMMANDS = {"ls", "cat", "head", "tail", "grep", "wc", "pwd"}
# Shell 元字符：涉及管道 |、重定向 >/calc<、串联 ;/&&、环境变量/命令替换 $()`\n 的一律拒绝判定为只读
SHELL_META = re.compile(r"[;&|<>`$\n]")

def is_readonly_command(command: str) -> bool:
    """
    保守地判断命令是否只读：必须是白名单里的单条命令，不带任何 shell 元字符。
    """
    if SHELL_META.search(command):  # 1. 发现任何管道、重定向、分号组合拳，直接返回 False
        return False
    try:    # 2. 使用 shlex 展开词法分析（处理空格与引号路径），提取第一个 Token 主命令
        tokens = shlex.split(command)
    except ValueError:
        return False
    # 3. 校验主命令是否在只读白名单内
    return bool(tokens) and tokens[0] in READONLY_COMMANDS

def run_command_self_check(args: dict):
    """
    run_command 专属安全拦截器：扫一遍命令，命中高危特征则返回 "ask" 要求用户手动审批。
    """
    command = args.get("command", "")
    if any(re.search(pattern, command) for pattern in DANGEROUS_PATTERNS):
        return "ask"
    if is_readonly_command(command):
        return "allow"
    # 没命中高危特征，交给通用规则决定
    return None

#【注册表模式 (Registry Pattern)】在 Python 里，函数也是一种数据类型，
# 把 run_command_self_check 函数作为回调引用，注册到 permissions 模块的 "run_command" key下。
# 当权限引擎校验 run_command 工具时，会自动查表并触发此回调函数。
permissions.register_self_check("run_command", run_command_self_check)
# permissions.register_self_check(run_command.__name__, run_command_self_check)