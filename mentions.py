"""
mentions.py - @ 文件引用解析与历史伪造器 (File Mention Synthesizer)

它不需要修改 Agent 底层的图状态机或硬编码逻辑，而是通过在发送 Prompt 之前伪造一段 read_file 的工具调用历史，完美避开了“未读盲改”的安全性拦截！

三大核心设计：
1. 正则抽取：精准识别 `@path`（自动规避 `user@email.com`），去重并保留解析顺序。
2. 历史伪造 (History Spoofing)：将 `@` 文件读出后打包为合法的 `ToolCallPart` + `ToolReturnPart` 历史消息，假装模型已发起过 `read_file`。
3. 状态预登记：调用 `read_and_register` 同步更新 `ReadFileState` 账本，确保后续 `edit_file` / `write_file` 顺利过校验。
4. 动态候选补全：优先利用 `git ls-files` 高效扫描，非 Git 项目退回带黑名单 (`IGNORED_DIRS`) 的 `os.walk` 目录遍历。
"""
import os
import re
import subprocess
import uuid

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from agent.file_state import ReadFileState
from agent.tools import read_and_register

# 匹配 @ 引用：要求 @ 前面是行首或空白，避免把 foo@bar.com 这种邮箱误判成引用
# @ 后面跟一段路径常见字符（字母、数字、下划线、点、斜杠、反斜杠、连字符、波浪号），遇到空格或中文标点就停，不会把后面的句子也吞进来
# 只支持纯路径 @path 的最简形态，不处理带引号路径、行号区间、@agent 等分支
AT_MENTION_RE = re.compile(r"(?:^|\s)@([\w./~\\\-]+)")

# 用户Prompt里，提取需要记录的文件名字（可能是多个），给main用的
# “请帮我修改 @abc.py 和 @utils.py” --> seen = ["abc.py" , "utils.py"]
def extract_at_mentions(text: str) -> list[str]:
    """
    从用户 Prompt 中解析出所有的 @path 路径，去重并保持出现顺序
    """
    seen = [] 
    for match in AT_MENTION_RE.finditer(text):
        path = match.group(1) #拿掉没用的@符号：group(0) 为 " @abc.py" ──► group(1) 为 "abc.py" 
        if path not in seen:
            seen.append(path)
    return seen

# 把每个文件都读一遍、记录read_file_state、上下文append假模型和工具交互消息，给main用的
def build_mention_messages(paths: list[str], state: ReadFileState) -> list:
    """
    把每个被 @ 的文件读出来，为它伪造一对 read_file 的 tool_call + tool_return，返回要塞进历史的消息列表。
    read_and_register 顺手把文件登记进 readFileState，和真正的 read_file 调用一模一样，之后 edit_file 就不会被拦。
    """
    messages = []
    for path in paths:
        abs_path = os.path.abspath(os.path.expanduser(path))
        
        if not os.path.isfile(abs_path):
            continue    # 不存在的路径、目录都跳过，简化处理
        try:
            numbered = read_and_register(state, abs_path) #触发：读+记录
        except (OSError, UnicodeDecodeError):
            continue

        # tool_call 和 tool_return 要用同一个 id 配对，模型才认得这是一次完整的调用
        call_id = "mention_" + uuid.uuid4().hex[:8]


        # 假消息：模型尝试调用工具的消息（LLM -> agent : ModelResponse
        messages.append(
            ModelResponse(
                parts=[ToolCallPart(tool_name="read_file", args={"path": abs_path}, tool_call_id=call_id)]
            )
        )
        # 假消息：工具调用完返回消息（本地工具 -> LLM : ModelRequest）
        messages.append(
            ModelRequest(
                parts=[ToolReturnPart(tool_name="read_file", content=numbered, tool_call_id=call_id)]
            )
        )
    return messages


# 遍历目录时要跳过的垃圾/缓存目录（黑名单）
IGNORED_DIRS = {"__pycache__", "node_modules", "venv", "dist", "build", "target"}

# 不需要入参，用git、filesystem把当前项目根目录（以及所有子目录）下的可编辑文件全都列出来, 给inputUI用的
def list_candidate_files() -> list[str]:
    """
    @ 补全的候选文件列表。优先用 git ls-files（快，又自动跳过 .gitignore 的文件）；不在 git 仓库就退回遍历目录。
    """

    # 策略 1：优先用 Git 命令（`git ls-files` 命令）扫描（极速、精准、自动遵守 .gitignore）
    try:    
        result = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, timeout=2
        )
        # 如果 exit code == 0（说明当前目录是个 Git 仓库且执行成功）
        tracked = result.stdout.splitlines() if result.returncode == 0 else []

        # 拿到了已跟踪文件就直接用；空列表（非 git 仓库、或文件还没 add）则退回遍历目录
        if tracked:
            return tracked
    except (OSError, subprocess.SubprocessError):
        pass

    # 降级策略try那边报错，用os.walk递归全部子目录和文件来查找（IGNORED_DIRS这类的不去查找）
    files = []
    for root, dirs, names in os.walk("."):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in IGNORED_DIRS]
        for name in names:
            files.append(os.path.relpath(os.path.join(root, name)))
    return files