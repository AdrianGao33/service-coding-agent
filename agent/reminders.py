"""
System Reminder 提示词构建器：环境状态感知与 Task 进度的 Prompt 反向注入。

核心职责：
1. 过时文件提醒 (build_file_reminder_text)：
   基于 ReadFileState 账本检测文件是否被外部修改，自动生成警告，提示 Agent 编辑前重新 read_file。

2. Task 状态提醒 (build_task_reminder_text)：
   结合 TasksStore 主动将最新任务列表推回 Prompt（免去 task_list），并提供哨兵文案供 Hook 识别冷却时间。
"""
from rich.markup import escape

from background_jobs import JobRegistry
from tasks_store import TasksStore
from ui.render import console

from .file_state import ReadFileState

def _wrap(lines: list[str]) -> str:
    # 所有 reminder 正文都包在 <system-reminder>...</system-reminder> 标签里，集中一处避免分散重复
    return f"<system-reminder>\n{chr(10).join(lines)}\n</system-reminder>"


def build_file_reminder_text(state: ReadFileState) -> str | None:
    """
    根据当前 ReadFileState 账本检测过时文件；若有文件变更则返回提醒 XML 文本，无变更返回 None。
    """
    stale = state.stale_paths()
    if not stale:
        return None

    # 拼装提示词
    lines = [
        "以下文件在你读取之后被外部修改过，你 context 里的内容可能已过时，编辑前请重新用 read_file 读取：",
    ]
    lines += [f"- {path}" for path in stale]
    return _wrap(lines)

def build_task_reminder_text(store: TasksStore) -> str:
    """
    拼出 task 提醒的正文：一段温和的提醒 + 当前 task 列表。列表直接从 store 读，所以这条提醒同时承担了"把磁盘状态反向推回 prompt"的作用，模型不必主动 task_list。

    开头那句话同时被 hooks.py 用来扫历史识别"上一条已经是 reminder"，保持稳定不要改。
    """
    tasks = store.list()
    lines = [
        "task 工具最近没有被使用。如果你正在处理的工作适合用 task 跟踪进度，建议用 task_create 新建 task，用 task_update 维护状态（开工时切 in_progress，做完切 completed）；如果列表里有过时的 task，也可以顺手清理掉。只在与当前工作相关时再用这些工具。这只是一句友好的提醒——和当前工作无关的话忽略即可。",
    ]
    if tasks:
        lines.append("")    #空行
        lines.append("现存的 task 列表：")
        lines.append("")    
        for t in tasks:
            lines.append(f"#{t.id}. [{t.status}] {t.subject}")
    return _wrap(lines)

def build_job_reminder_text(registry: JobRegistry | None) -> str | None:
    """
    把「已结束但还没通知」的后台 job 拼成 <task-notification> 通知；没有新完成的就返回 None。
    """
    if registry is None:
        return None
    jobs = registry.pop_unnotified()
    if not jobs:
        return None
    blocks = []
    for job in jobs:
        # 通知模型的同时，也给用户打一行完成提示
        color = "green" if job.status == "completed" else "red"
        console.print(f"[{color}]●[/] {escape(job.summary())}\n")
        summary = f"{job.summary()}，可用 read_file 读输出文件"
        blocks.append(
            "<task-notification>\n"
            f"<task-id>{job.id}</task-id>\n"
            f"<task-type>{job.kind}</task-type>\n"
            f"<output-file>{job.log_path}</output-file>\n"
            f"<status>{job.status}</status>\n"
            f"<summary>{summary}</summary>\n"
            "</task-notification>"
        )
    return "\n\n".join(blocks)