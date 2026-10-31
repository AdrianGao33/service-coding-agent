"""
本文件是后台任务执行核心层 (`background_jobs.py`)，负责管理 Agent 在本地执行的所有长周期任务后台任务执行层：
把 shell 命令作为后台 job 运行，输出落盘到日志文件，并跟踪完成状态用于通知

它的核心职责与功能范围包括：
1. Job 实体定义 (Job)：封装任务的唯一标识、生命周期状态、输出日志落盘路径、进程终止句柄等。
2. 进程衍生与隔离 (spawn_shell)：通过异步子进程执行 shell 命令，并针对 Windows (CREATE_NEW_PROCESS_GROUP)
   和 POSIX (start_new_session) 分别实现进程组隔离，确保能够级联杀死整个进程树（避免孤儿进程）。
3. 异步状态监视 (_watch)：利用 asyncio 协程非阻塞地等待进程退出，并自动更新任务的成败状态。
4. 进程树安全终止 (_kill_process_tree)：提供跨平台的强力终止机制（Windows 的 taskkill / POSIX 的 killpg）。
5. 注册表与生命周期调度 (JobRegistry)：
   - 会话级（Session-level）的任务生命周期中心。
   - 追踪前台与后台任务的转换（如用户通过 Ctrl+B 将前台耗时任务转入后台）。
   - 维护「已完成但尚未通知模型」的任务队列 (pop_unnotified)，供后续的 hook 和 reminder 注入给 Agent。
   - 会话退出时的安全清理 (shutdown)。
"""
from __future__ import annotations

import asyncio
import os
import random
import signal
import string
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

JOBS_ROOT = Path.home() / ".my-code-agent" / "jobs"

_ID_ALPHABET = string.digits + string.ascii_lowercase


def _new_job_id() -> str:
    """
    生成一个唯一的 Job ID。
    使用随机字符串而非自增计数器的设计考量：
    即使重启进程、恢复会话后也不会发生 ID 撞车，更不会覆盖掉旧的任务日志。
    前缀 b 标的是 job 的类型（取 bash 的首字母）
    """
    return "b" + "".join(random.choices(_ID_ALPHABET, k=8))

# 定义任务可能处于的几种状态类型
JobStatus = Literal["running", "completed", "failed", "killed"]

# 状态码到人类可读文本的映射，用于生成通知摘要
_STATUS_TEXT = {
    "completed": "执行成功",
    "failed": "执行失败",
    "killed": "已被终止",
}


@dataclass
class Job:
    """
    Job 数据类：
    代表一个具体的后台任务实体，记录了该任务从出生到终结的全生命周期元数据。
    """
    id: str
    # "shell"，"subagent"
    kind: str
    description: str
    # Job 的日志输出存储到文件
    log_path: Path
    # 任务当前状态，初始为运行，另有三种：completed\failed\killed
    status: JobStatus = "running"
    # 进程退出码，进程结束前为 None
    returncode: int | None = None
    # 标记该任务的结束状态是否已经通知过 Agent（防止重复通知）
    notified: bool = False
    # background 标志位：
    # - False 表示前台命令：其结果会直接随 tool result 返回给模型，不发送异步完成通知。
    # - True 表示后台任务：用户按 ctrl+b 或模型指定时会翻转为 True。
    background: bool = True
    # 由任务创建方（如 spawn_shell）注入的闭包函数，用于后续安全终止该任务及其子进程
    kill_func: Callable[[], None] | None = None

    def summary(self) -> str:
        """
        生成一句话描述这个 job 的最终结局，专门用于组装推送给 Agent 的通知文本。
        """
        detail = _STATUS_TEXT[self.status]
        if self.returncode is not None:
            detail += f"，exit code {self.returncode}"
        return f"后台命令「{self.description}」{detail}"


class JobRegistry:
    """
    会话级 Job 注册表 (JobRegistry)：
    作为中央管理器，负责创建、查询、终止后台 Job，并把「已完成但还没通知」的 Job 交付给通知通道。
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._jobs: dict[str, Job] = {}          # 内存中的任务字典：key 是 job_id，value 是 Job 实例
        self._jobs_dir = JOBS_ROOT / session_id  
        # 将该 Task 强引用存入 `self._watchers` 集合，防止被 Python 的 GC 机制提前误杀。
        self._watchers: set = set()
    
    # 起进程
    async def spawn_shell(self, command: str, background: bool = True) -> Job:
        """
        衍生（Spawn）一个后台 Shell 任务：
        1. 创建唯一的 Job ID 与日志文件句柄。
        2. 启动平台隔离的异步子进程，并将输出直接重定向落盘到日志文件。
        3. 注册异步监视协程（_watch），在后台静默等待子进程运行完毕。
        """
        # 1. 生成全局唯一的 Job ID，创建会话专属的日志落盘目录
        job_id = _new_job_id()
        # 2. 确保日志存储目录存在
        self._jobs_dir.mkdir(parents=True, exist_ok=True)      
        log_path = self._jobs_dir / f"{job_id}.log"
        # 3. 以二进制写模式打开日志文件，用于吸纳子进程的 stdout/stderr
        log_file = open(log_path, "wb")

        # 跨平台进程组隔离处理：
        # 单独起一个进程组（Process Group），之后才能确保不仅能杀掉主进程，还能把其衍生的子孙进程一并清理掉。
        if os.name == "nt":
            group_kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        else:
            group_kwargs = {"start_new_session": True}

        # 启动，初始化，异步执行 shell 命令，将输出绑定到日志文件
        # create_subprocess_shell拉起子进程和初始化，绑定给proc异步子进程对象，任务此时会在后台进行
        # 将进程的 stdout（标准输出流）和 stderr（错误输出流）合并到一起，保存到 log_file 文件中
        proc = await asyncio.create_subprocess_shell(
            command, stdout=log_file, stderr=subprocess.STDOUT, **group_kwargs
        )
        pid = proc.pid

        # 实例化 Job 对象，并注入对应的终止闭包（绑定了具体的进程 PID）
        job = Job(
            id=job_id,
            kind="shell",
            description=command,
            log_path=log_path,
            background=background,
            kill_func=lambda: _kill_process_tree(pid),
        )
        self._jobs[job_id] = job        #一对一的关系

        # watcher任务加到状态保存里一直盯着后台command运行情况
        # 调用 `asyncio.create_task()` 将 `_watch` 协程抛给事件循环，形成一个独立的后台 Task（分身任务）
        # 主流程（主光标）毫不停留，瞬间释放并返回 Job 实例
        watcher = asyncio.create_task(self._watch(job, proc, log_file))
        # 将该 Task 强引用存入 `self._watchers` 集合，防止被 Python 的 GC 机制提前误杀
        self._watchers.add(watcher)
        # 挂载 `add_done_callback(self._watchers.discard)`，预定在任务终结时由事件循环自动清理
        watcher.add_done_callback(self._watchers.discard)
        return job

    async def _watch(self, job: Job, proc: asyncio.subprocess.Process, log_file) -> None:
        """
        异步监视器（Private Watcher Coroutine）：
        非阻塞地等待子进程运行结束，关闭日志文件，并根据退出码更新任务状态。
        """
        # 该协程把自己挂起并退出事件循环的活跃队列，
        # 不消耗任何 CPU 资源。它完全依赖操作系统内核的进程退出信号（如 SIGCHLD）由事件循环唤醒。
        returncode = await proc.wait()
        log_file.close()
        if job.status == "killed":
            return
        # 防御性检查：若已被外部强行终止（killed），则保留终止态，不作状态覆盖
        job.returncode = returncode
        # 终结状态写入
        job.status = "completed" if returncode == 0 else "failed"

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return list(self._jobs.values())

    def running(self, background: bool | None = None) -> list[Job]:
        """
        查询当前仍在运行（running）的 Job。
        可通过传入 background 参数来精准筛选：
        - False: 只筛出前台命令
        - True: 只筛出后台任务
        - None: 所有正在跑的任务
        """
        return [
            j for j in self._jobs.values()
            if j.status == "running" and (background is None or j.background is background)
        ]

    def move_to_background(self) -> None:
        """
        将所有还在运行的前台命令批量转为后台任务（通常绑定到用户的 Ctrl+B 快捷键触发）。
        """
        for job in self.running(background=False):
            job.background = True

    def kill(self, job_id: str) -> bool:
        """
        安全终止指定的后台任务：
        通过先修改状态（置为 killed 并标记为已通知），再调用底层终止函数，
        从而有效防止 watcher 协程随后将其改成 completed 并引发重复通知的竞态条件。
        """
        job = self._jobs.get(job_id)
        if job is None or job.status != "running":
            return False
        # 先置 killed + 已通知再动手终止，防 watcher 随后改成 completed 又通知一遍
        job.status = "killed"
        job.notified = True
        if job.kill_func:
            job.kill_func()
        return True

    def pop_unnotified(self) -> list[Job]:
        """
        获取所有「已完成、属于后台任务、但还未曾通知过 Agent」的 Job 列表，
        并在获取的同时将其标志位 `notified` 设置为 True（即消费掉这些未读通知）。
        """
        done = [j for j in self._jobs.values() if j.background and j.status != "running" and not j.notified]
        for job in done:
            job.notified = True
        return done

    def shutdown(self) -> int:
        """
        会话退出清理 (Shutdown)：
        强行终止所有还在运行的 Job（确保子进程不会变成孤儿进程活过会话周期），并返回被终止的任务数量。
        """
        running = self.running()
        for job in running:
            self.kill(job.id)
        return len(running)


def _kill_process_tree(pid: int) -> None:
    """
    跨平台的进程树强力终止函数 (Kill Process Tree)：
    针对 Windows 和类 Unix (Linux/macOS) 系统采取不同的系统级杀死策略，
    确保连同该进程衍生的所有子孙进程一并干净彻底地干掉。
    """
    if os.name == "nt":
        # Windows 平台：使用 taskkill 命令通过 /F (强制) /T (终止指定的进程及由它启动的任何子进程)
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
