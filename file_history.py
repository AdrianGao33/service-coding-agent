"""
为整个会话期间被 Agent 修改的代码文件提供版本控制、检查点记录、差异对比和一键回滚能力。

核心机制亮点：
1. 两次关键备份时机：
   - track_edit：在 Agent 第一次修改某个文件前，把初始状态备份为版本 v1（如果文件本来不存在，则记录为 None）。
   - make_checkpoint：在每条用户输入时触发，给内容有变化的文件自增拍新版本，并记录当时的对话历史下标。

2. 完美持久化：
   - 二进制备份文件存放在 file-history/<session_id>/ 目录下。
   - 检查点清单存在 checkpoints.json 中，即使退出重启（/resume）也能继续回溯。
"""
import difflib
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import session


@dataclass
class Checkpoint:
    """
    【检查点对象】：代表代码和对话在某一个特定时间点的“快照”。
    在每条用户输入进入对话历史之前建立。
    """
    # 检查点时刻对话历史的长度（Index），回滚对话时直接把历史记录截断到这个下标
    history_index: int
    # 用户的输入原文，回滚后可以重新填回输入框供用户参考或再次发送
    prompt: str
    timestamp: float
    # 字典映射：文件路径 (path) -> 该文件在当前检查点对应的版本号 (int)
    backups: dict = field(default_factory=dict)


@dataclass
class FileChange:
    """
    【文件改动结构体】：描述某个文件在回滚或单轮迭代中发生了什么变化。
    """
    path: str
    # action 动作类型：
    # - 恢复计划中：'restore'（恢复旧版本）/ 'delete'（目标版本当时根本不存在，需删除）
    # - 单轮改动统计中：'create'（新建）/ 'edit'（修改）/ 'delete'（删除）
    action: str
    insertions: int     # 增加的行数
    deletions: int      # 删除的行数


def _count_diff(current, target) -> tuple:
    """
    【差异行数计算】：借助 difflib.SequenceMatcher 统计从当前内容 (current) 
    回退到目标内容 (target) 会带来多少行数的增减。
    """
    # 将二进制字节码解码为字符串并按行分割，对异常编码做容错处理（replace）
    cur = (current or b"").decode("utf-8", "replace").splitlines()
    tgt = (target or b"").decode("utf-8", "replace").splitlines()
    insertions = deletions = 0
    # 使用 Python 内置的序列匹配器比对两个文本列表
    matcher = difflib.SequenceMatcher(None, cur, tgt, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            deletions += i2 - i1
        if tag in ("replace", "insert"):
            insertions += j2 - j1
    return insertions, deletions


class FileHistory:
    """
    【会话级文件历史管理器】：负责把文件各个版本备份到磁盘，并将检查点列表维护在 checkpoints.json 中。
    """

    def __init__(self, session_id: str):
        # 确定当前会话专属的文件历史备份目录：project_dir/file-history/<session_id
        self.dir = session.project_dir() / "file-history" / session_id
        # 检查点列表（按时间先后从旧到新排列）
        self.checkpoints: list[Checkpoint] = []
        # 路径映射：path -> 备份文件名列表 (list)。
        # 列表下标 i 对应版本号 i+1（如下标 0 是 v1）；如果对应版本文件不存在则为 None
        self.versions: dict[str, list] = {}
        # 初始化时尝试从本地恢复历史记录
        self._load()

    # —— 记录 ——
    # 首次修改前备份+落盘，非首次直接返回不处理 todo
    def track_edit(self, path: str) -> None:
        """
        【首次修改拦截/埋点】：在工具真正写文件之前调用。
        如果这是当前会话第一次碰这个文件，立刻把改动前的原始内容备份成版本 v1。
        """
        path = os.path.abspath(path)
        if path in self.versions:
            return
        self.versions[path] = [self._backup(path, 1)]   #写入状态
        self._save()                                    #落盘

    def make_checkpoint(self, history_index: int, prompt: str) -> None:
        """
        【创建检查点】：在用户每输入一条新消息时执行。
        遍历所有被跟踪的文件：如果内容在上一轮被改动过了，就为它拍个照，生成一个新版本。
        """
        backups = {}
        for path, names in self.versions.items():
            latest = len(names)
            if self._current(path) != self._read_version(path, latest):
                names.append(self._backup(path, latest + 1))
            backups[path] = len(names)
        self.checkpoints.append(
            Checkpoint(history_index, prompt, time.time(), backups)
        )
        self._save()

    # —— 回退 ——

    def diff_stats(self, cp: Checkpoint) -> list:
        """
        【生成回滚执行计划（只读）】：计算“从当前磁盘状态”回到指定检查点 `cp` 
        所需要做的所有文件改动清单（FileChange）。内容没有变的文件不会被包含进来。
        """
        plan = []
        for path in self.versions:
            target = self._content_at(cp, path)
            current = self._current(path)
            if target == current:
                continue
            # 如果目标内容为 None，说明当时文件根本不存在，回退时需要执行删除动作
            action = "delete" if target is None else "restore"
            insertions, deletions = _count_diff(current, target)
            plan.append(FileChange(path, action, insertions, deletions))
        return plan

    def turn_stats(self, cp: Checkpoint) -> list:
        """
        【单轮迭代改动统计】：计算由该检查点 `cp` 开启的那一轮对话期间，文件具体发生了什么改动。
        （通过和下一个检查点做对比；如果是最新的检查点，则和当前磁盘状态做对比）
        """
        index = self.checkpoints.index(cp)
        nxt = self.checkpoints[index + 1] if index + 1 < len(self.checkpoints) else None
        changes = []
        for path in self.versions:          
            before = self._content_at(cp, path)
            after = self._content_at(nxt, path) if nxt else self._current(path)
            if before == after:
                continue
            if before is None:
                action = "create"
            elif after is None:
                action = "delete"
            else:
                action = "edit"
            insertions, deletions = _count_diff(before, after)
            changes.append(FileChange(path, action, insertions, deletions))
        return changes

    def rewind_files(self, cp: Checkpoint) -> list:
        """
        【执行文件回退】：把生成的差异执行计划真正应用到磁盘上，完成代码文件的时光倒流。
        """
        plan = self.diff_stats(cp)
        for change in plan:
            if change.action == "delete":
                # 如果当时文件不存在，现在直接把它删掉
                os.remove(change.path)
            else:
                # 获取该文件在目标检查点对应的具体版本号
                version = self._target_version(cp, change.path)
                # 在备份列表中找到对应的备份文件名
                name = self.versions[change.path][version - 1]
                # 使用 shutil.copy2 把备份文件拷回原路径（同时连同文件的修改时间等元数据一起恢复）
                shutil.copy2(self.dir / name, change.path)
        return plan

    def drop_from(self, cp: Checkpoint) -> None:
        """
        【检查点修剪】：当对话回退到某个检查点后，这个检查点之后的所有“未来检查点”
        对应的消息和代码状态都失效了，直接从列表中切掉并保存。
        """
        self.checkpoints = self.checkpoints[: self.checkpoints.index(cp)] #从开始到前一步，本身都不要
        self._save()

    # —— 内部辅助 ——

    def _target_version(self, cp: Checkpoint, path: str) -> int:
        # 如果检查点记录里没有这个文件，说明它是在这个检查点之后才首次被修改的，
        # 故在更早的检查点看它，应当回退到最原始的 v1 版本
        return cp.backups.get(path, 1)

    def _content_at(self, cp: Checkpoint, path: str):
        # 某文件在检查点时刻的内容，None 表示当时不存在
        return self._read_version(path, self._target_version(cp, path))

    def _backup(self, path: str, version: int):
        """
        【核心备份】：把文件当前内容拷进备份目录，并返回生成的备份文件名；若文件不存在返回 None。
        """
        if not os.path.exists(path):
            return None
        digest = hashlib.sha256(path.encode()).hexdigest()[:16]
        name = f"{digest}@v{version}"
        self.dir.mkdir(parents=True, exist_ok=True)
        # 拷贝文件到备份目录
        shutil.copy2(path, self.dir / name)
        return name

    def _read_version(self, path: str, version: int):
        # 读取某个版本对应的备份内容（统一用 bytes 二进制读取，对非文本文件也安全兼容）
        name = self.versions[path][version - 1]
        return None if name is None else (self.dir / name).read_bytes()

    def _current(self, path: str):
        # 读取当前磁盘文件的二进制内容，若文件已被删则返回 None
        try:
            return Path(path).read_bytes()
        except FileNotFoundError:
            return None

    # —— 持久化 ——

    def _save(self) -> None:
        # 将当前的所有检查点（转成字典）和文件版本映射保存到 checkpoints.json 中
        self.dir.mkdir(parents=True, exist_ok=True)
        data = {
            "checkpoints": [asdict(s) for s in self.checkpoints],
            "versions": self.versions,
        }
        (self.dir / "checkpoints.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        # 恢复会话时从 checkpoints.json 重建检查点；新会话没有这个文件，从空白开始
        path = self.dir / "checkpoints.json"
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.checkpoints = [Checkpoint(**s) for s in data["checkpoints"]]
        self.versions = data["versions"]
