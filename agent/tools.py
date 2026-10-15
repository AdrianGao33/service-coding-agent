"""
Coding Agent 用到的三个工具：读文件、写文件、跑 shell 命令。

核心设计：
1. 声明即 Schema：函数的类型提示 (Type hint) + 文档注释 (docstring) 会被 PydanticAI
   自动打包成 JSON Schema 发给大模型，大模型据此感知工具名称、功能与入参格式。

2. 错误分级治理：
   - raise ModelRetry模型可修正错误：抛出 raise ModelRetry(...) 触发框架让大模型自我纠错重试；
   - return执行客观报错：返回 "错误/日志..." 字符串，作为工具结果交由大模型看日志 Debug；
   - 严重崩溃：依靠全局 Hook 捕获底层未知异常。   

3. 安全拦截：搭配 permissions 模块注册高危特征检查（如 rm、sudo），执行前触发审批。

4. 状态管理 (ReadFileState)：
    - 依赖注入：利用 RunContext[ReadFileState] 共享跨工具的会话级文件状态账本。
    - 防盲改与防竞态：强约束“先读后改”，通过 mtime 时间戳与 snapshot 内容对比拦截过期修改。
    - 精准编辑与切片：read_file 输出行号并支持分页；edit_file 提供上下文唯一性校验与局部替换。
    - 串行化保障：对写操作施加 Tool(..., sequential=True)，防止同轮 Tool Call 并发写盘导致踩踏。
"""

# re 是 Python 正则表达式库，这里用来精准扫描命令文本；通过 \b（单词边界）约束，
# 能严格区分独立的危险命令（如单独的 rm），避免误伤包含相同字母的正常单词（如 terminal 或 format）。
import os
import re 
import shlex
import subprocess

# PydanticAI 专有异常：抛出后框架会自动生成 retry-prompt 塞回给大模型，要求其修正参数重试
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai import RunContext, Tool

import permissions
from .file_state import ReadFileState

DEFAULT_MAX_LINES = 2000

# 给文本格式化附加 cat -n 风格的行号前缀（右对齐 + 制表符），方便 LLM 精准定位
def _with_line_numbers(content: str, start_line: int = 1) -> str:
    """
    给每行加上行号前缀，格式类似 cat -n：行号右对齐 + 一个制表符 + 原行内容。
    模型按行号定位代码，给 edit_file 的 old_string 才能对得准。
    start_line 是首行的行号；分段读取时它等于 offset，行号才能和文件对得上。
    """
    lines = content.splitlines()
    if not lines:
        return "(空文件)"
    # 按本段最大行号算对齐宽度（分段读取时行号从 start_line 起，不一定从 1 开始）
    width = len(str(start_line + len(lines) - 1))
    return "\n".join(f"{i:>{width}}\t{line}" for i, line in enumerate(lines, start_line))

def read_and_register(state: ReadFileState, path: str, offset: int = 1, limit: int | None = None) -> str:
    """
    读取文件、按 offset/limit 切片、登记进 readFileState(大str)，返回带行号的内容。
    read_file 工具和 @ 引用注入都调用它，所以经 @ 引入的文件和模型自己读的文件，在 readFileState 里长得一模一样。
    """
    # 1. 读取文件全文
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()  #格式化1：这边的问题就是读完 文本会被拉长成一个大字符串，原文本的行会用无数个 /n 表达

    # 2. 第一次读完处理格式：按 offset/limit 切片：不指定 limit 时最多取 DEFAULT_MAX_LINES 行
    all_lines = content.splitlines()    # 格式化2：把长str，变成列表拿掉/n：['第1行内容...', '第2行内容...', ...]
    total = len(all_lines)
    start = offset - 1
    if start >= total and total > 0:
        return f"警告：文件只有 {total} 行，但 offset 是 {offset}，没有内容可读"
    end = start + limit if limit is not None else start + DEFAULT_MAX_LINES
    selected = all_lines[start:end]  # 格式化3：从全量列表里，取出要用的切片
    selected_content = "\n".join(selected)   # 格式化4：用/n组装回长str
    truncated = limit is None and end < total   #boolean，是否被截断了（没指定 limit 且 end < total）

    # 3. 登记进会话的 readFileState：存重新组装还原的长str
    state.record(path, selected_content, offset=offset, limit=limit)

    # 4. 返回给模型的是带行号的版本；如果被截断，末尾附提示
    result = _with_line_numbers(selected_content, start_line=offset) # 调用 _with_line_numbers，虽然还是大str，但是有行号模型读的更方便
    if truncated:
        result += f"\n\n（文件共 {total} 行，还有 {total - end} 行未显示。用 offset={end + 1} 继续读取）"
    return result

# 读取文件内容（带行号格式）。大文件支持 offset/limit 分段读取。
# 后续用 edit_file 编辑时，old_string/new_string 不要包含行号前缀。
def read_file(ctx: RunContext[ReadFileState], path: str, offset: int = 1, limit: int | None = None) -> str:
    """
    读取文件内容，输出带行号（类似 cat -n 格式）。
    大文件请用 offset 和 limit 分段读取，不必一次读完。
    后续用 edit_file 编辑时，old_string 和 new_string 里不要包含行号前缀。

    Args:
        path: 要读取的文件的绝对路径
        offset: 从第几行开始读，从 1 算起（默认 1 = 文件开头）
        limit: 读多少行；不传则最多读 DEFAULT_MAX_LINES 行
    """
    # 1. 省token不读：readFileState 里有记录、同范围、同 mtime → 跳过不读
    record = ctx.deps.get(path)
    if record is not None and record.get("offset") is not None:
        if record["offset"] == offset and record["limit"] == limit:
            try:
                if os.path.getmtime(path) <= record["timestamp"]:
                    return "文件内容没有变化，请参考上次 read_file 的结果，不必重复读取"
            except FileNotFoundError:
                pass

    # 2. 读取 + 登记都交给 read_and_register，异常在这里转成 ModelRetry 反馈给模型
    try:
        return read_and_register(ctx.deps, path, offset, limit)
    except FileNotFoundError:
        raise ModelRetry(f"文件 {path} 不存在，请确认路径或换一个文件")
    except PermissionError:
        raise ModelRetry(f"没有权限读取 {path}，请换一个可读的文件")
    except IsADirectoryError:
        raise ModelRetry(f"{path} 是一个目录，请指定目录下的具体文件")
    except UnicodeDecodeError:
        return f"错误：{path} 不是文本文件，无法读取"


def edit_file(ctx: RunContext[ReadFileState], path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """
    在文件里精确替换字符串，只传改动的片段，省 token。
    编辑前必须先用 read_file 读取文件，否则会报错。
    修改已有文件时优先用 edit_file 而不是 write_file。
    old_string 必须在文件里唯一匹配；多处匹配时补充上下文使其唯一，或传 replace_all=true 全部替换。
    old_string 和 new_string 不要包含 read_file 输出的行号前缀。

    Args:
        path: 要编辑的文件路径
        old_string: 要被替换掉的原文片段，必须和文件里的内容逐字符一致（不含行号前缀）
        new_string: 替换后的新内容
        replace_all: 是否替换所有匹配项，默认只替换唯一的一处
    """
    # 1. 拦截新旧一样
    if old_string == new_string:
        raise ModelRetry("old_string 和 new_string 完全相同，这次编辑没有任何改动")

    # 2. 拦截未读就要改
    record = ctx.deps.get(path)
    if record is None:
        raise ModelRetry(f"还没读过 {path}，请先用 read_file 读取它，再基于真实内容编辑")

    # 3. 拦截竞态覆盖（比对读取后是否有外部改动）
    try:
        current_mtime = os.path.getmtime(path)
    except FileNotFoundError:
        raise ModelRetry(f"{path} 已不存在，无法编辑")
    if current_mtime > record["timestamp"]:
        raise ModelRetry(f"{path} 在你读取之后被改动过，请重新 read_file 拿到最新内容再编辑")

    # 4. 拦截多处old_string不唯一的情况，驳回再议
    snapshot = record["content"]
    count = snapshot.count(old_string)
    if count == 0:
        raise ModelRetry("old_string 在文件里找不到，请对照 read_file 的输出确认原文")
    if count > 1 and not replace_all:
        raise ModelRetry(
            f"old_string 在文件里出现了 {count} 次，请补充上下文让它唯一，或传 replace_all=true 全部替换"
        )

    # 5. 校验都过了，开始替换新旧内容
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if replace_all:
        updated = content.replace(old_string, new_string)
    else:
        updated = content.replace(old_string, new_string, 1) # 如果多个有重复，只替换遇见的第一个

    # 7. 整体写回磁盘，再用新内容和新 mtime 刷新登记
    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)
    ctx.deps.record(path, updated)
    return f"已编辑 {path}（替换 {count if replace_all else 1} 处）"


def write_file(ctx: RunContext[ReadFileState], path: str, content: str) -> str:
    """
    把内容整体写入文件，已有文件会被覆盖。
    覆盖已有文件前必须先用 read_file 读取过，否则会报错。
    修改已有文件优先用 edit_file（只传改动片段，省 token），write_file 只用于新建文件或整体重写。
    """
    # 拦截未读就要改 + 读之后被改动
    if os.path.exists(path):
        record = ctx.deps.get(path)
        if record is None:
            raise ModelRetry(f"{path} 已存在，覆盖前请先用 read_file 读一遍，确认不会误删内容")
        if os.path.getmtime(path) > record["timestamp"]:
            raise ModelRetry(f"{path} 在你读取之后被改动过，请重新 read_file 再覆盖")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except FileNotFoundError:
        raise ModelRetry(f"目录不存在，无法写入 {path}，请换一个已存在的目录")
    except PermissionError:
        raise ModelRetry(f"没有权限写入 {path}，请换一个可写的路径")
    except OSError as e:
        return f"错误：写入 {path} 失败 ({e})"
    # 新写入的内容同样登记进 readFileState，后续要再改就不必重读
    ctx.deps.record(path, content)
    return f"已写入 {path}"
    
def run_command(command: str) -> str:
    """
    执行一条 shell 命令并返回输出。
    """
    try:
        result = subprocess.run(
            command, 
            shell=True,           # 允许通过 Shell 解释执行完整命令
            capture_output=True,  # 截获标准输出和标准错误，供程序读取而非打到屏幕
            text=True,            # 输出自动转为字符串，非 bytes
            errors="replace",     # 遇到特殊编码字符用 ? 替换，防止报错崩溃
            timeout=10,           # 设置 10 秒超时拦截，防止死循环命令卡死后台
        )
        output = result.stdout
        if result.returncode != 0:
            output += f"\n[错误] {result.stderr}"
        return output or "(无输出)"
    except subprocess.TimeoutExpired:
        return "[错误] 命令执行超时（10秒）"
    except OSError as e:
        return f"[错误] 无法执行命令 ({e})"


# 高危命令的特征：删除文件、提权、直写磁盘
DANGEROUS_PATTERNS = [
    r"\brm\b",          # 删除文件/目录指令 (如 rm -rf)
    r"\bsudo\b",        # 提权执行指令
    r"\bdd\b",          # 底层磁盘/块设备直接写指令
    r"\bmkfs\w*\b",     # 格式化文件系统指令 (如 mkfs, mkfs.ext4)
]

# 白名单：严格认可的保守只读命令列表
READONLY_COMMANDS = {"ls", "cat", "head", "tail", "grep", "wc", "pwd"}
# Shell 元字符：涉及管道 |、重定向 >/calc<、串联 ;/&&、环境变量/命令替换 $()`\n 的一律拒绝判定为只读
SHELL_META = re.compile(r"[;&|<>`$\n]")

def is_readonly_command(command: str) -> bool:
    """
    保守地判断命令是否只读：必须是白名单里的单条命令，不带任何 shell 元字符。
    """
    if SHELL_META.search(command):  # 1. 发现任何管道、重定向、分号组合拳，直接返回 False
        return False
    try:    # 2. 使用 shlex 展开词法分析（处理空格与引号路径），提取第一个 Token 主命令
        tokens = shlex.split(command)
    except ValueError:
        return False
    # 3. 校验主命令是否在只读白名单内
    return bool(tokens) and tokens[0] in READONLY_COMMANDS

def run_command_self_check(args: dict):
    """
    run_command 专属安全拦截器：扫一遍命令，命中高危特征则返回 "ask" 要求用户手动审批。
    """
    command = args.get("command", "")
    if any(re.search(pattern, command) for pattern in DANGEROUS_PATTERNS):
        return "ask"
    if is_readonly_command(command):
        return "allow"
    # 没命中高危特征，交给通用规则决定
    return None

#【注册表模式 (Registry Pattern)】在 Python 里，函数也是一种数据类型，
# 把 run_command_self_check 函数作为回调引用，注册到 permissions 模块的 "run_command" key下。
# 当权限引擎校验 run_command 工具时，会自动查表并触发此回调函数。
permissions.register_self_check("run_command", run_command_self_check)
# permissions.register_self_check(run_command.__name__, run_command_self_check)

# Pydantic AI 支持 tools=[plain_function]，从函数签名 + docstring 自动生成 JSON Schema
# edit_file 和 write_file 标记 sequential=True：同一轮里的多个改文件调用必须串行执行，
# 否则它们会基于同一份旧快照并发写盘、互相覆盖（这正是 readFileState + mtime 想防住的并发问题）
TOOLS = [
    read_file,
    Tool(edit_file, sequential=True),
    Tool(write_file, sequential=True),
    run_command,
]