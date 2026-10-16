"""
shell 工具：run_command，自检对高危命令要求审批，对简单的只读命令直接放行。
"""
# re 是 Python 正则表达式库，这里用来精准扫描命令文本；通过 \b（单词边界）约束，
# 能严格区分独立的危险命令（如单独的 rm），避免误伤包含相同字母的正常单词（如 terminal 或 format）。
import re 
import shlex
import subprocess

import permissions

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