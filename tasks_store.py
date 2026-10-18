"""
TasksStore 任务持久化存储层：基于本地 JSON 文件的 Session 级任务状态账本。

核心设计：
1. 会话隔离与磁盘持久化：
   - 数据按照 session_id 隔离存储在 ~/.my-code-agent/tasks/<session_id>/ 目录下。
   - 每个 Task 独立落盘为一个 <id>.json 文件，支持中断后通过 /resume 命令完整恢复任务状态链。

2. 极简读缓存机制 (Read-Through Cache)：
   - 解决 TUI 高频刷新（~100ms/帧）导致的 I/O 瓶颈问题：在 __init__ 初始化阶段将所有 Task 一次性加载至内存字典 _tasks。
   - 所有读操作 (list/get) 完全走内存查询，不触发任何磁盘 I/O；写操作 (create/update/delete) 采用“内存字典 + 磁盘同步写”双更新策略，兼顾性能与崩溃安全。

3. 水位线 ID 防复用算法 (.highwatermark)：
   - 维护 .highwatermark 隐藏文件记录历史上分配过的“最大 Task ID”。
   - ID 生成规则为 max(当前内存最大 ID, 水位线 ID) + 1，确保即使中间某个 Task 被删除，其 ID 也绝不会被后续新 Task 复用，彻底避免依赖链乱码。

4. 列表增量更新与线程/进程边界：
   - update() 内部针对 blocks / blocked_by 依赖列表提供追加与去重语义（dict.fromkeys），其余字段利用 dataclasses.replace 实现安全的不可变拷贝覆盖。
   - 设计定位为单进程模型（Single-Process CLI Engine），mutation 过程无需重量级文件锁，保持极高的轻量性与执行速度。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterator, Literal


# Task 状态类型强约束（'deleted' 仅作为 update 的内部信号，不直接落盘）
TaskStatus = Literal["pending", "in_progress", "completed"]


@dataclass
class Task:
    """
    单个任务的数据结构
    """
    id: str                         # 任务唯一编号（自增数字字符串，如 "1", "2"，数字自增）
    subject: str                    # 任务标题/简述
    description: str                # 任务详细描述与要求
    active_form: str | None = None  # 任务执行时的动态表单/补充状态
    status: TaskStatus = "pending"  # 当前状态：未开始、进行中、已完成
    blocks: list[str] = field(default_factory=list)     # 当前任务阻塞了哪些后续 Task ID
    blocked_by: list[str] = field(default_factory=list) # 当前任务被哪些前置 Task ID 阻塞


_HIGHWATERMARK = ".highwatermark"   # 存放历史最大 ID 的隐藏文件名


def _data_root() -> Path:
    """获取 Agent 本地数据的根目录（~/.my-code-agent）"""
    return Path.home() / ".my-code-agent"


class TasksStore:
    """
    按 session_id 隔离的任务存储管理类，提供 CRUD 接口与内存缓存；每次 mutation 都同步写盘，后续 /resume 时可以恢复历史 task。
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.dir = _data_root() / "tasks" / session_id
        self.dir.mkdir(parents=True, exist_ok=True)

        # 1. 启动时一次性把磁盘上的 JSON 读入内存字典，后续 list/get/update/delete 都走内存 + 同步写盘
        self._tasks: dict[str, Task] = {}
        for path in self._json_paths(): #如果有曾经在对话存放的任务json，使用resume时候加载到内存；如果空文件夹下，这里一次都不会循环
            try:
                task = Task(**json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            self._tasks[task.id] = task
        # 同样的，highwatermark 一次读入，create/delete 直接维护内存里的整数
        self._highwatermark = self._read_highwatermark()

    # ---------- TasksStore类：公共 crud API ----------

    def create(self, subject: str, description: str, active_form: str | None = None) -> str:
        """
        创建新任务并同步写盘
        ID 生成规则：取（当前已有最大 ID）与（历史水位线 ID）的最大值 + 1，防复用
        """
        current_max = max((int(tid) for tid in self._tasks), default=0)
        next_id = max(current_max, self._highwatermark) + 1
        task = Task(
            id=str(next_id),
            subject=subject,
            description=description,
            active_form=active_form,
        )
        self._tasks[task.id] = task
        self._write(task)
        return task.id

    def list(self) -> list[Task]:
        """
        获取所有任务列表（按 ID 数字升序排序）
        直接读取内存字典，不触发磁盘 I/O，适合 UI 100ms 级别的高频刷新
        """
        return sorted(self._tasks.values(), key=lambda t: int(t.id))

    def get(self, task_id: str) -> Task | None:
        """根据 ID 查任务（纯内存查询）""" 
        return self._tasks.get(task_id)

    def update(self, task_id: str, **fields) -> Task:   #**argus
        """
        更新指定任务的属性
        特例：add_blocks / add_blocked_by 会做追加 + 去重，其余字段直接覆盖
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"task #{task_id} not found")
        add_blocks = fields.pop("add_blocks", None)
        add_blocked_by = fields.pop("add_blocked_by", None)
        new_blocks = list(dict.fromkeys(task.blocks + list(add_blocks))) if add_blocks else task.blocks     #append新的blocks
        new_blocked_by = list(dict.fromkeys(task.blocked_by + list(add_blocked_by))) if add_blocked_by else task.blocked_by #append新的blocked_by
        updated = replace(task, **fields, blocks=new_blocks, blocked_by=new_blocked_by)     #更新替换
        self._tasks[task_id] = updated
        self._write(updated)
        return updated

    def delete(self, task_id: str) -> None:
        """
        删除任务并更新高水位线文件，确保被删的 ID 以后不会被重复分配
        """
        if task_id not in self._tasks:
            raise KeyError(f"task #{task_id} not found")
        
        # 删除前把当前最大 ID 抬入水位线持久化
        current_max = max((int(tid) for tid in self._tasks), default=0)
        if current_max > self._highwatermark:
            self._highwatermark = current_max
            self._write_highwatermark(self._highwatermark)

        # 删内存并删磁盘文件    
        del self._tasks[task_id]
        self._task_path(task_id).unlink()

    # ---------- TasksStore类：内部工具函数 ----------

    def _json_paths(self) -> Iterator[Path]:
        """遍历当前 session 目录下的所有 task JSON 文件（过滤掉 .highwatermark 等隐藏文件）"""
        for path in self.dir.iterdir():     # 空文件夹下，这里一次都不会循环
            if path.suffix == ".json" and not path.name.startswith("."):
                yield path

    def _task_path(self, task_id: str) -> Path:
        """拼接单个 Task 的 JSON 文件路径"""
        return self.dir / f"{task_id}.json"

    def _highwatermark_path(self) -> Path:
        """获取水位线记录文件的路径"""
        return self.dir / _HIGHWATERMARK

    def _read_highwatermark(self) -> int:
        """从文件读取历史最大 ID，不存在或解析失败则返回 0"""
        path = self._highwatermark_path()
        if not path.exists():
            return 0
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return 0

    def _write_highwatermark(self, value: int) -> None:
        """将当前最高水位线数字写入磁盘文件"""
        self._highwatermark_path().write_text(str(value), encoding="utf-8")

    def _write(self, task: Task) -> None:
        """将 Task dataclass 序列化为美化的 JSON 并写入文件"""
        self._task_path(task.id).write_text(
            json.dumps(asdict(task), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
