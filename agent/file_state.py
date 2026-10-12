"""
agent/file_state.py - 会话级文件读取状态追踪器 (ReadFileState)

职责：
1. 共享依赖注入：read_file / edit_file / write_file 工具共享此状态单例（跟随 Session 刷新）。
2. 观察者账本：登记已读文件的绝对路径、读取内容、行号范围以及文件修改时间戳 (mtime)。
3. 安全闸门校验：为 edit_file 和 write_file 提供校验依据，防止“未读盲改”或“竞态覆盖”。
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
