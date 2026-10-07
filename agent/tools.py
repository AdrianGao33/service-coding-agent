"""
Coding Agent 用到的三个工具：读文件、写文件、跑 shell 命令。

1. Type hint（类型提示） + docstring（函数文档注释）会被 Pydantic AI 自动提取解析成 JSON Schema，
   作为 Tools 声明发送给大模型 API，大模型据此感知工具的名称、功能及入参格式。

2. 三层错误治理体系：
   - 分级 1：raise ModelRetry("提示...") —— 针对入参可修改救活的错误（如路径拼错、权限不足）。
     拦截后由 PydanticAI 框架全自动打回给大模型在同一轮次重试修正。
   - 分级 2：return "错误/日志信息..." —— 针对客观报错或调试上下文（如 Shell 报错、超时、二进制文件）。
     作为正常的 ToolReturnPart 供大模型读取日志并自主决策下一步 Debug 动作。
   - 分级 3：Global Hooks 防线 —— 由 @hooks.on_tool_execute_error 捕获底层未知严重崩溃，保护进程。

3. 工具安全与权限治理 (Tool Permissions & Audit)：
   - 配合 permissions 模块，利用“注册表模式 (Registry Pattern)”在工具执行前注入高危命令匹配自检规则。
"""
# re 是 Python 正则表达式库，这里用来精准扫描命令文本；通过 \b（单词边界）约束，
# 能严格区分独立的危险命令（如单独的 rm），避免误伤包含相同字母的正常单词（如 terminal 或 format）。
import re 
import subprocess

# 导入 PydanticAI 专门用于告知大模型“参数有误，请修改后重试”的特化异常类
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
            capture_output=True,  # 截获 stdout 和 stderr，供程序读取而非直接打屏
            text=True,            # 将输出自动转为 str 类型 (非 bytes)
            errors="replace",     # 遇到非法编码字符时用 ? 替换，不抛出 UnicodeDecodeError
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
    run_command 的权限自检：扫一遍命令字符串，命中高危特征就要求审批。
    args (dict): 大模型传入工具的入参字典，格式为 {"command": "..."}

    """
    command = args.get("command", "")
    if any(re.search(pattern, command) for pattern in DANGEROUS_PATTERNS):
        return "ask"
    # 没命中高危特征，交给通用规则决定
    return None

#【注册表模式 (Registry Pattern)】
# 将 run_command_self_check 函数作为回调引用，注册到 permissions 模块的 "run_command" key下。
# 当权限引擎校验 run_command 工具时，会自动查表并触发此回调函数。
permissions.register_self_check("run_command", run_command_self_check)

# Pydantic AI 支持 tools=[plain_function]，从函数签名 + docstring 自动生成 JSON Schema
TOOLS = [read_file, write_file, run_command]