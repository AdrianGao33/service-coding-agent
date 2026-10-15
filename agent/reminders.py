"""
system-reminder 正文构造。真正的注入在 agent/hooks.py 里挂 hook。

核心职责：
检查当前会话的 ReadFileState 账本，若发现已读文件在磁盘被外部（如用户在 IDE 中）改动过，
自动拼装 `<system-reminder>` XML 标签文本，提示 Agent 在编辑前必须重新 read_file。
"""
from .file_state import ReadFileState


def build_reminder_text(state: ReadFileState) -> str | None:
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
    body = "\n".join(lines)

    # XML标签规范：用 <system-reminder> 标签包裹，便于模型区分系统指令与用户对话
    return f"<system-reminder>\n{body}\n</system-reminder>"
