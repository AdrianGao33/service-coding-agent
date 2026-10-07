"""
会话持久化模块：把对话历史写成 JSONL 文件，支持按项目隔离存储、追加写入、历史扫描与恢复会话。

设计模式：
1. 采用 JSONL (JSON Lines) 格式实现 Append-only 追加写入，保证崩塌安全与高并发性能。
2. 依据当前工作目录 (CWD) 进行 Hash/Sanitize 隔离，确保不同项目间的会话上下文独立。
3. 利用 PydanticAI 的 ModelMessagesTypeAdapter 实现强类型的消息序列化与反序列化。
"""
import json #json.dumps转换: dict -> json / json.loads转换: json -> dict
import re
import uuid
from datetime import datetime
from pathlib import Path

# PydanticAI 提供的消息类型适配器，转换 dict -> ModelMessage 对象
from pydantic_ai.messages import ModelMessagesTypeAdapter
# pydantic_core 提供的底层函数，能把 Pydantic 对象 -> 普通 Python dict
from pydantic_core import to_jsonable_python

# 所有会话记录的根目录
STORAGE_ROOT = Path.home() / ".adrian-code-agent" / "projects"


def sanitize_path(path: str) -> str:
    """
    把项目绝对路径转码成合法的目录名：非字母数字字符一律换成 -
    例如：'/Users/adrian/code' -> '-Users-adrian-code'
    """

    return re.sub(r"[^a-zA-Z0-9]", "-", path) # 用法: re.sub(pattern, replacement, string)


def project_dir() -> Path:
    """
    当前项目（工作目录）对应的会话存储目录。
    """
    return STORAGE_ROOT / sanitize_path(str(Path.cwd()))


def new_session_id() -> str:
    """
    生成一个全局唯一的会话 ID (UUID v4 格式)。
    """
    return str(uuid.uuid4())


def session_file(session_id: str) -> Path:
    """
    根据会话 ID 拼接出其在本地磁盘上的完整 JSONL 文件路径。
    """
    return project_dir() / f"{session_id}.jsonl"


def append_messages(session_id: str, messages) -> None:
    """
    把本轮新增的消息追加到会话文件末尾，一行一条。
    """
    path = session_file(session_id)
    # 防御性创建：确保项目的存储子目录存在，若不存在则递归创建
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        for msg in messages:
            # 1. to_jsonable_python: 将 PydanticAI 对象转为干净的 Python dict/list
            # 2. json.dumps: 转为 JSON 字符串，ensure_ascii=False 保持中文原样展示
            # 3. 追加换行符 \n 形成标准 JSONL 格式
            f.write(json.dumps(to_jsonable_python(msg), ensure_ascii=False) + "\n")


def load_history(session_id: str) -> list:
    """
    读取整个会话文件，把每行 JSON 还原成 SDK 的消息对象列表。
    """
    # 一次性读取文件的所有行并按行切分
    lines = session_file(session_id).read_text(encoding="utf-8").splitlines()

    # 逐行 json.loads 解析为 dict，再通过 PydanticAI 的 TypeAdapter 一键反序列化为消息对象列表
    return ModelMessagesTypeAdapter.validate_python(
        [json.loads(line) for line in lines]
    )

# 提取会话总结（为list_sessions提供菜单行展示文本）
def first_prompt(path: Path) -> str:
    """
    只读文件第一行，提取首条用户输入作为这个会话的摘要。
    """
    with open(path, encoding="utf-8") as f:
        head = f.readline() # 极速读取第一行，无需加载整个文件
    msg = json.loads(head)
    for part in msg.get("parts", []):
        if part.get("part_kind") == "user-prompt":
            return str(part.get("content", ""))
    return "(空会话)"

# 提供菜单
def list_sessions() -> list:
    """
    扫描当前项目的所有会话文件，按修改时间从新到旧返回
    (session_id, 修改时间, 首条用户输入) 列表。
    """
    if not project_dir().exists():
        return []

    # 获取目录下所有 .jsonl 文件，并依据文件的修改时间 (st_mtime) 倒序排列
    files = sorted(
        project_dir().glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [
        (p.stem, datetime.fromtimestamp(p.stat().st_mtime), first_prompt(p))
        for p in files
    ]
