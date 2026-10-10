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
"""
# re 是 Python 正则表达式库，这里用来精准扫描命令文本；通过 \b（单词边界）约束，
# 能严格区分独立的危险命令（如单独的 rm），避免误伤包含相同字母的正常单词（如 terminal 或 format）。
import re 
import subprocess

# PydanticAI 专有异常：抛出后框架会自动生成 retry-prompt 塞回给大模型，要求其修正参数重试
from pydantic_ai.exceptions import ModelRetry

import permissions

def read_file(path: str) -> str:
    """
    读取指定文件的内容。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:  
        raise ModelRetry(f"文件 {path} 不存在，请确认路径或换一个文件")
    except PermissionError:      
        raise ModelRetry(f"没有权限读取 {path}，请换一个可读的文件")
    except IsADirectoryError:    
        raise ModelRetry(f"{path} 是一个目录，请指定目录下的具体文件")
    except UnicodeDecodeError:     
        return f"错误：{path} 不是文本文件，无法读取"


def write_file(path: str, content: str) -> str:
    """
    将内容写入指定文件。
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已写入 {path}"
    except FileNotFoundError:     
        raise ModelRetry(f"目录不存在，无法写入 {path}，请换一个已存在的目录")
    except PermissionError:       
        raise ModelRetry(f"没有权限写入 {path}，请换一个可写的路径")
    except OSError as e:        
        return f"错误：写入 {path} 失败 ({e})"
    
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


def run_command_self_check(args: dict):
    """
    run_command 专属安全拦截器：扫一遍命令，命中高危特征则返回 "ask" 要求用户手动审批。
    """
    command = args.get("command", "")
    if any(re.search(pattern, command) for pattern in DANGEROUS_PATTERNS):
        return "ask"
    # 没命中高危特征，交给通用规则决定
    return None

#【注册表模式 (Registry Pattern)】在 Python 里，函数也是一种数据类型，
# 把 run_command_self_check 函数作为回调引用，注册到 permissions 模块的 "run_command" key下。
# 当权限引擎校验 run_command 工具时，会自动查表并触发此回调函数。
permissions.register_self_check("run_command", run_command_self_check)
# permissions.register_self_check(run_command.__name__, run_command_self_check)

# 导出工具列表，统一注册给 Agent 实例
TOOLS = [read_file, write_file, run_command]