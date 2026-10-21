"""
store.py 是整个跨会话长期记忆系统（Long-term Memory System）的底层存储与持久化引擎。它通过文件系统实现项目级隔离的记忆存储，核心职责包括：

1. 文件目录与规范管理：基于当前会话所属的项目目录，自动定位与初始化 .memory/ 记忆存储空间。

2. Markdown + Frontmatter 解析：每条长期记忆均独立存储为带 YAML 风格头部（Frontmatter）的 Markdown 文件，支持通过轻量级解析提取文件名、描述、类型等元信息，无需读取全文即可完成关联判断。

3. 安全截断与索引维护：对全局扁平索引文件（MEMORY.md）及单条召回记忆实施严格的行数与字节双重上限保护（防止上下文污染与 Token 溢出），并附带智能的截断警告。

4. 新鲜度评估与闸门控制：将文件修改时间转换为自然语言新鲜度描述（如“今天”、“昨天”），并提供路径校验工具，用于权限放行及检测当前对话是否已经包含记忆写入行为。

底层工具支持：轻量扫描元数据 -> 动态生成清单 -> 按需精准加载 -> 双重安全截断与时效防呆。
"""
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import session

# MEMORY.md 索引每轮全量注入上下文，必须封顶：超过任一上限就截断
MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25_000 #一个英文字符基本就是1byte

# 单条记忆召回注入时的截断上限
MAX_MEMORY_LINES = 200
MAX_MEMORY_BYTES = 4096

# 解析 frontmatter 只看文件头这么多行，不必读完正文
FRONTMATTER_MAX_LINES = 30


def memory_dir() -> Path:
    """
    当前项目的记忆目录，和会话文件放在同一个项目目录下，记忆按项目隔离。
    """
    return session.project_dir() / "memory"


def index_path() -> Path:
    """
    获取全局索引文件 MEMORY.md 的完整路径。
    """
    return memory_dir() / "MEMORY.md"


def ensure_memory_dir() -> None:
    """
    确保记忆目录存在。
    在系统启动时调用，建好目录后，system prompt 里才能告诉模型「目录已存在，不要重复 mkdir」。
    """
    memory_dir().mkdir(parents=True, exist_ok=True)


@dataclass
class MemoryHeader:
    """
    单个记忆文件的元信息，从 frontmatter 解析而来；召回只凭它判断相关性，不必读正文。
    """
    filename: str           #"2026-06-06-async-fix.md"
    path: Path              # Path("/path/to/project/.memory/2026-06-06-async-fix.md")
    mtime: float            # 1780704000.0 (对应 2026-06-06 的文件修改时间戳)
    name: str               #"Python异步编程事件循环阻塞"
    description: str        #"记录了在 asyncio 中同步阻塞 I/O 导致事件循环卡死的排查与修复方案"
    type: str               #""bug_fix"

# 单个记忆子文件元信息
def _parse_frontmatter(path: Path) -> dict:
    """
    解析记忆文件头部 frontmatter 里的 name / description / type 字段。
    限制只读取前 FRONTMATTER_MAX_LINES(30) 行，避免大文件带来的性能开销。
    输出一个格式化好的dict，fields:{"name":"...", "description":"...", "type":"..."}
    """
    try:
        with open(path, encoding="utf-8") as f:
            lines = [f.readline() for _ in range(FRONTMATTER_MAX_LINES)]
    except OSError:
        return {}
    if not lines or lines[0].strip() != "---":  #没有yaml格式的元数据，不读了返回空
        return {}
    fields = {}
    for line in lines[1:]:          #略过起始的---
        if line.strip() == "---":   #读到结束线---
            break
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields

# 全部项目记忆元信息菜单
def scan_memory_files() -> list[MemoryHeader]:
    """
    扫描所有记忆文件（排除索引 MEMORY.md 和空文件），并提取头部元数据，按修改时间从新到旧排序。
    项目下（可能cross多session）全部记忆子文件.md的头部元数据整体list all
    """
    directory = memory_dir()
    if not directory.exists():
        return []
    headers = []
    for path in directory.glob("*.md"):
        if path.name == "MEMORY.md":
            continue
        # 空文件是合并整理时被清空的旧记忆，跳过
        if path.stat().st_size == 0:
            continue
        fields = _parse_frontmatter(path)
        headers.append(MemoryHeader(
            filename=path.name,
            path=path,
            mtime=path.stat().st_mtime,
            name=fields.get("name", path.stem),
            description=fields.get("description", ""),
            type=fields.get("type", ""),
        ))
    headers.sort(key=lambda h: h.mtime, reverse=True)
    return headers

# 压缩记忆清单内容成大str的prompt给大模型
def format_manifest(headers: list[MemoryHeader]) -> str:
    """
    把记忆元信息拼成每条一行的清单，给召回和提炼的 prompt 用。
    """
    lines = []
    for h in headers:
        day = datetime.fromtimestamp(h.mtime).strftime("%Y-%m-%d")
        lines.append(f"- [{h.type}] {h.filename}（{day}）：{h.description}")
    return "\n".join(lines)

# 防御性读取工具，限制最大行数和最大bytes，尾部添加提示语句
def truncate_content(content: str, max_lines: int, max_bytes: int, notice: str) -> str:
    """
    安全截断核心函数：
    1. 先按最大行数 (max_lines) 截断；
    2. 再按最大字节数 (max_bytes) 截断（自动回退到最后一个完整行，防止乱码或断行）；
    3. 若发生删减，则在文本末尾附上友好的提示通知 (notice)。
    """
    lines = content.splitlines()
    truncated = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    text = "\n".join(lines)
    if len(text.encode("utf-8")) > max_bytes:
        text = text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
        if "\n" in text:
            text = text[: text.rfind("\n")]
        truncated = True
    if truncated:
        text += f"\n\n> {notice}"
    return text

# 读取项目记忆菜单
def read_index() -> str | None:
    """
    读取 MEMORY.md 索引文件内容：
    超过行数或字节上限时会自动截断，并附带一条提醒模型保持索引精炼的警告。
    """
    path = index_path()
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not content.strip():
        return None
    return truncate_content(
        content, MAX_INDEX_LINES, MAX_INDEX_BYTES,
        "警告：MEMORY.md 超过上限（200 行 / 25KB），只加载了一部分。请保持索引每条一行，把细节挪进记忆文件。",
    )


def age_days(mtime: float) -> int:
    """
    计算给定时间戳（mtime）距离当前过去了多少天。
    """
    return max(0, int((time.time() - mtime) // 86400))


def age_text(days: int) -> str:
    """
    将天数转换为自然语言描述（「今天 / 昨天 / N 天前」）。
    大模型对相对天数比直接做日期算术敏感得多，能更好地理解记忆的时效性。
    """
    if days == 0:
        return "今天"
    if days == 1:
        return "昨天"
    return f"{days} 天前"


def staleness_text(days: int) -> str:
    """
    当记忆超过一定时间时，在注入时开头附带的新鲜度警告文本，
    提醒模型注意历史记忆可能已失效，需核对当前代码。
    """
    return (
        f"这条记忆已保存 {days} 天。记忆是写入当时的快照，不是实时状态——"
        "其中关于代码行为、文件位置的描述可能已经过期，先核对当前代码再当作事实使用。"
    )


def is_memory_path(path: str) -> bool:
    """
    校验目标路径是否落在当前项目的记忆目录内 (.memory/)。
    权限放行和后台 Agent 的写入闸门共用这个判断逻辑。
    """
    if not path:
        return False
    try:
        return Path(path).resolve().is_relative_to(memory_dir().resolve())
    except (OSError, ValueError):
        return False

# 结束对话时候模型记录记忆点、运行过程中可能模型判断需要记录记忆点，这个防重复
# 注意检查的messages整个上下文
def has_memory_writes(messages) -> bool:
    """
    检查当前对话的消息历史中，是否已经包含对记忆目录内文件的写入工具调用。
    如果主对话自己已经记过账了，后台自动提炼就可以跳过，避免重复写入。
    """
    for message in messages:
        for part in message.parts:
            if getattr(part, "part_kind", "") != "tool-call":
                continue
            if part.tool_name not in ("write_file", "edit_file"):
                continue
            if is_memory_path(str(part.args_as_dict().get("path", ""))):
                return True
    return False
