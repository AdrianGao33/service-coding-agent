"""
Coding Agent 用到的三个工具：读文件、写文件、跑 shell 命令。
Type hint + docstring 会被 Pydantic AI 自动解析成 JSON Schema，生成可调用的函数列表给大模型。
报错不会被抛出，而是被捕获并返回给大模型，避免中断对话。
"""
import subprocess


def read_file(path: str) -> str:
    """
    读取指定文件的内容。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"错误：文件 {path} 不存在"


def write_file(path: str, content: str) -> str:
    """
    将内容写入指定文件。
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已写入 {path}"


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


# Pydantic AI 支持 tools=[plain_function]，从函数签名 + docstring 自动生成 JSON Schema
TOOLS = [read_file, write_file, run_command]
