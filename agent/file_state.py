"""
agent/file_state.py - 会话级文件读取状态追踪器 (ReadFileState)

职责：
1. 共享依赖注入：read_file / edit_file / write_file 工具共享此状态单例（跟随 Session 刷新）。
2. 观察者账本：登记已读文件的绝对路径、读取内容、行号范围以及文件修改时间戳 (mtime)。
3. 安全闸门校验：为 edit_file 和 write_file 提供校验依据，防止“未读盲改”或“竞态覆盖”。
4. 过时文件检测：对比磁盘 mtime 与账本时间戳，识别在读取后被外部进程改动过的文件 (stale_paths)。
"""
import os

class ReadFileState:
    """
    持有一个会话的读取状态：记录本会话中所有读过或写入过的文件元数据。
    edit_file 和 write_file 改文件前都查它：没读过不让改，读完又被别人改过也不让改。
    """

    def __init__(self):
        # 字典映射：绝对路径 -> {1."content": 文本, 2."timestamp": 3.mtime时间戳, 4."offset": 行偏移, 5."limit": 读取行数}
        self._state: dict = {}

    def record(self, path: str, content: str, offset: int | None = None, limit: int | None = None) -> None:
        """
        登记一个文件：存下刚读到的内容、此刻文件的 mtime、以及读取范围（offset/limit可以理解成读时候的起始行到结束行）。
        read_file 会传入非 None 的 offset；edit_file / write_file 不传（保持 None），标记这是一条全量登记、去重时跳过，借此区分两种来源。
        """
        self._state[os.path.abspath(path)] = {
            "content": content,
            "timestamp": os.path.getmtime(path), #系统捕获触发自动记录
            "offset": offset,
            "limit": limit,
        }

    def get(self, path: str):
        """
        取出一个文件的登记，从没读过就返回 None。
        """
        return self._state.get(os.path.abspath(path))

    def stale_paths(self) -> list[str]:
        """
        返回磁盘 mtime 比登记的读取时间更新的文件的绝对路径——
        也就是被读取之后又被本进程之外的东西改动过的文件。
        模型自己的 edit_file / write_file 会通过 record() 刷新 timestamp，所以它自己的改动绝不会算进来。
        """
        stale = []
        for path, record in self._state.items():
            # 文件可能已被删除，getmtime 抛错就跳过，不当成被修改
            try:
                # 在操作系统中，越晚发生的时间，时间戳（秒数数字）越大（例如：10:00 的时间戳是 100，10:05 的时间戳是 400）
                # 【磁盘文件上保存的最后修改时间】>【内存里记录的历史时间】做比对 (正常应该相等，但是大于说明改过)
                if os.path.getmtime(path) > record["timestamp"]:
                    stale.append(path)
            except OSError:
                continue
        return stale