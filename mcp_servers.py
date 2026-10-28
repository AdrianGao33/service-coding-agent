"""
mcp_servers.py 的核心职责是管理整个 Agent 的外部工具箱。它在业务上主要完成四件事：

路径扫描：去用户家目录（~/.my-code-agent/mcp.json）和当前工作目录（.mcp.json）寻找配置文件。

状态登记：把每一个解析出来的 MCP 服务包装成统一的体检卡（ServerRecord），跟踪它是连通还是失败。

安全代理与组装：自动给每个服务的工具加上前缀（防止重名冲突），并处理参数。

生命周期托管：在 Agent 启动时通过 AsyncExitStack 批量拉起子进程或建立连接，在退出时一键销毁。
"""
import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 把 .mcp.json 里的静态配置，变成内存里真正可以被 LLM 调用的Toolset对象。
# 它会去读配置、建立连接、把不同服务器吐出来的工具收编、洗净（Normalize），最后打包成 Toolset
from pydantic_ai.mcp import load_mcp_toolsets

# 定义全局用户配置文件（MCP通讯录）和当前项目配置文件的路径
USER_CONFIG = Path.home() / ".my-code-agent" / "mcp.json"
PROJECT_CONFIG = Path(".mcp.json")


@dataclass
class ServerRecord:
    toolset: Any                                        # 实际的工具集对象
    transport: str = ""                                 # 传输通道描述（如本地命令行进程或远程 HTTP 地址）
    status: str = "pending"                             # 状态：pending（未连接）、connected（已连接）、failed（连接失败）
    error: str = ""                                     # 记录连接失败时的错误原因
    tool_names: list[str] = field(default_factory=list) # 该服务包含的工具名称清单
    toolset_id: str = ""                                # 服务的唯一标识名

# 全局状态：所有 MCP 服务的记录表，以及用来统一管理异步上下文退出的堆栈
RECORDS: list[ServerRecord] = []
_stack = AsyncExitStack()
# 它允许你在一个循环里，动态地把一个个要异步打开的连接塞进这个栈里（比如代码里的 await _stack.enter_async_context(record.toolset)）。
# 当整个程序要退出时，只要在最外层调用一次 await _stack.aclose()，它就会按照先进后出的顺序，把里面所有注册的异步连接全部关闭。
# 比起asyn with open() as f: 是固定数量的需要写死，AsynExistStack更灵活动态

def _toolset_id(toolset: Any) -> str:
    """
    通过检查对象的各种候选属性或类名，提取出当前 toolset 的唯一业务 ID。
    第三方MCP服务端开发者并没有完全遵循官方规范，先尝试寻找一些常见变量名当id，最后拿函数类名兜底
    """
    wrapped = getattr(toolset, "wrapped", toolset)
    for candidate in (
        getattr(toolset, "id", None),
        getattr(wrapped, "id", None),
        getattr(toolset, "prefix", None),
        getattr(wrapped, "prefix", None),
        getattr(toolset, "label", None),
        getattr(wrapped, "label", None),
        getattr(toolset, "name", None),
        getattr(wrapped, "name", None),
        toolset.__class__.__name__,
    ):
        if candidate not in (None, ""):
            return str(candidate)
    return toolset.__class__.__name__


def _normalize_toolset(toolset: Any) -> Any:
    """
    标准化 toolset 配置：确保它拥有统一的工具前缀（防止工具名撞车）、超时时间和指令开关。
    """
    toolset_id = _toolset_id(toolset)
    # 自动生成形如 mcp__服务名_ 的前缀
    if not hasattr(toolset, "tool_prefix"):
        setattr(toolset, "tool_prefix", f"mcp__{toolset_id}_")   #工具前缀防止工具重名用错
    if not hasattr(toolset, "include_instructions"):
        setattr(toolset, "include_instructions", True)          #包含服务说明书
    if not hasattr(toolset, "timeout"):
        setattr(toolset, "timeout", 30)                        #调用工具等待最大时间
    return toolset

# 识别并返回该 MCP 服务底层的通信传输方式及其具体的连接参数。
def _transport(toolset: Any) -> str:
    """
    识别当前 MCP 服务是通过本地进程（stdio）拉起来的，还是通过远程网络（http/sse）调用的。
    """
    wrapped = getattr(toolset, "wrapped", toolset)
    # 有 command 说明是本地子进程命令行通信
    if hasattr(wrapped, "command"):
        args = list(getattr(wrapped, "args", []))
        return "stdio: " + " ".join([getattr(wrapped, "command", ""), *args])
    # 有 url 说明是远程网络通信
    if hasattr(wrapped, "url"):
        return f"http: {wrapped.url}"
    raw_url = getattr(getattr(toolset, "client", None), "url", None)
    if raw_url:
        return f"http: {raw_url}"
    return "transport: unknown"

# 依次读取用户全局与本地项目的 MCP 配置文件，解析并标准化所有工具集对象，按 ID 覆盖去重后重新刷进全局记录表 RECORDS 中。
def load_servers() -> None:
    """
    加载业务配置：顺序扫描用户目录和项目目录下的 mcp 配置文件，去重并初始化记录表。
    """
    merged: dict[str, Any] = {}

    for config_path in (USER_CONFIG, PROJECT_CONFIG):
        if not config_path.is_file():
            continue
        
        try:
            toolsets = list(load_mcp_toolsets(config_path))
        except TypeError:
            toolsets = list(load_mcp_toolsets(str(config_path)))

        for toolset in toolsets:
            toolset = _normalize_toolset(toolset)
            merged[_toolset_id(toolset)] = toolset  

    # 刷新全局记录列表
    RECORDS.clear()
    RECORDS.extend(    
        ServerRecord(
            toolset=toolset,
            transport=_transport(toolset),
            toolset_id=_toolset_id(toolset),
        )
        for toolset in merged.values() #逐个iter添加每个ServerRecord
    )


async def startup() -> str:
    """
    启动流程：加载配置，并发起连接所有配置好的 MCP 服务，最后返回一段人性化的状态汇报字符串。
    """
    load_servers()
    # 并发连接多个MCP服务器
    # * 解包运算符（Unpacking Operator）
    await asyncio.gather(*(_connect(record) for record in RECORDS))

    connected = [r for r in RECORDS if r.status == "connected"]
    failed = [r for r in RECORDS if r.status == "failed"]
    parts = []
    if connected:
        parts.append(
            "已连接 MCP server：" + "、".join(
                f"{r.toolset_id}（工具按需加载）" if not r.tool_names else f"{r.toolset_id}（{len(r.tool_names)} 个工具）"
                for r in connected
            )
        )
    if failed:
        parts.append("连接失败：" + "、".join(r.toolset_id for r in failed) + "（详情见 /mcp）")
    return "；".join(parts)


async def _connect(record: ServerRecord) -> None:
    """
    单个服务的连接执行函数：通过异步上下文管理器正式激活该 toolset（拉起本地进程或完成网络握手）。
    """
    try:
        await _stack.enter_async_context(record.toolset)
        record.status = "connected"
        record.tool_names = []
    except Exception as e:
        # 剥离嵌套异常，提取最干净的错误信息
        while getattr(e, "exceptions", None):
            e = e.exceptions[0]
        record.status = "failed"
        record.error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__


async def shutdown() -> None:
    """
    优雅退出：一键关闭所有通过 AsyncExitStack 托管的 MCP 进程和连接资源。
    """
    try:
        await _stack.aclose()
    except Exception:
        pass


def active_toolsets() -> list[Any]:
    """
    对外接口：返回当前所有连接成功的 toolset 实例列表（直接交给我们主流程的 Agent 挂载使用）。
    """
    return [record.toolset for record in RECORDS if record.status == "connected"]
