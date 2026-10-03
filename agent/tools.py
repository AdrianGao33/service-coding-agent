"""
Coding Agent 用到的三个工具：读文件、写文件、跑 shell 命令。
"""
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(path: str) -> Path:
    """
    把任意相对路径或 POSIX/Windows 风格路径统一解析到当前项目目录下。
    避免模型在 Windows 上产生类似 /src/main.py 这样的 Linux 路径导致失败。
    """
    if path is None:
        raise ValueError("路径不能为空")

    raw_path = str(path).strip().replace("\\", "/")
    if not raw_path:
        raise ValueError("路径不能为空")

    if raw_path.startswith("/"):
        candidate = PROJECT_ROOT / raw_path.lstrip("/")
    else:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate

    resolved = candidate.resolve(strict=False)
    root = PROJECT_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"路径必须位于项目目录内：{path}") from exc

    return resolved


def read_file(path: str) -> str:
    """
    读取指定文件的内容。
    """
    try:
        resolved = _resolve_project_path(path)
        return resolved.read_text(encoding="utf-8")
    except (FileNotFoundError, ValueError):
        return f"错误：文件 {path} 不存在或不在项目目录内"


def write_file(path: str, content: str) -> str:
    """
    将内容写入指定文件。
    """
    try:
        resolved = _resolve_project_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"已写入 {resolved}"
    except ValueError as exc:
        return str(exc)


def run_command(command: str) -> str:
    """
    执行一条 shell 命令并返回输出。
    """
    try:
        if os.name == "nt":
            cmd = ["powershell", "-NoProfile", "-Command", command]
        else:
            cmd = ["bash", "-lc", command]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=20,
            cwd=str(PROJECT_ROOT),
        )
        output = result.stdout
        if result.returncode != 0:
            output += f"\n[错误] {result.stderr}"
        return output or "(无输出)"
    except FileNotFoundError:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=20,
                cwd=str(PROJECT_ROOT),
            )
            output = result.stdout
            if result.returncode != 0:
                output += f"\n[错误] {result.stderr}"
            return output or "(无输出)"
        except subprocess.TimeoutExpired:
            return "[错误] 命令执行超时（20秒）"
    except subprocess.TimeoutExpired:
        return "[错误] 命令执行超时（20秒）"


# Pydantic AI 支持 tools=[plain_function]，从函数签名 + docstring 自动生成 JSON Schema
TOOLS = [read_file, write_file, run_command]
