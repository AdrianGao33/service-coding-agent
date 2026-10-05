"""
Coding Agent 用到的三个工具：读文件、写文件、跑 shell 命令。

1. Type hint（类型提示） + docstring（函数文档注释）会被 Pydantic AI 自动提取解析成 JSON Schema，
   作为 Tools 声明发送给大模型 API，大模型据此感知工具的名称、功能及入参格式。

2. 错误治理体系：
   - ModelRetry 意料之内的异常：抛出 ModelRetry，触发 PydanticAI 内部重试机制，提示模型修改参数；
   - return 不可纠正错误 / 过程结果（二进制文件、Shell 命令报错/超时）：直接 return 错误文本，作为正常上下文传给模型分析；
   - raise 意料之外的未捕获异常：由 hooks 里的 on_tool_execute_error 进行最外层防护兜底。
   # 核心区别：ModelRetry 是“入参有误打回强行纠错”（计入 retry 上限并强制改参重调），而 return 是“客观结果交卷反馈”（作为正常上下文供大模型自主决策下一步）。
"""
import subprocess

# 导入 PydanticAI 专门用于告知大模型“参数有误，请修改后重试”的特化异常类
from pydantic_ai.exceptions import ModelRetry


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


# Pydantic AI 支持 tools=[plain_function]，从函数签名 + docstring 自动生成 JSON Schema
TOOLS = [read_file, write_file, run_command]
