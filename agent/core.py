"""
agent/core.py - Agent 组装与依赖注入中心

职责：
1. 模型初始化：加载 DeepSeekProvider，配置 `deepseek-flash` OpenAIChatModel 实例。
2. 系统提示词 (System Prompt)：约束 Agent 的行为准则（修改前先读、优先用 edit_file、先运行验证）。
3. 依赖注入声明：指定 deps_type=ReadFileState，将会话级文件状态账本安全注入给各个工具。
4. 挂载能力：拼装全局 TOOLS 工具集与 hooks 生命周期切面。
"""
import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

from .file_state import ReadFileState
from .hooks import hooks
from .tools import TOOLS

# 从环境变量读取 API Key
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("请先设置环境变量 API_KEY")

MODEL_NAME = "deepseek-flash"

model = OpenAIChatModel(
    MODEL_NAME,
    provider=DeepSeekProvider(api_key=API_KEY),
)

agent = Agent(
    model,
    # system prompt的一部分，告诉大模型它的身份、能力、工作流程
    instructions=(
        "你是一个编程助手。你可以读写文件和执行命令来帮用户完成编程任务。\n"
        "修改已有文件前必须先用 read_file 读取它。"
        "改动局部内容时优先用 edit_file（只传改动的片段，省 token），新建文件或整体重写才用 write_file。\n"
        "工作流程：先理解需求，写代码，然后运行验证。"
        "如果有错误就修复并重新运行，直到确认正确。"
    ),
    # step1 pydantic 会根据函数签名 + docstring 自动生成 JSON Schema
    # step2 agent 会根据 JSON Schema 生成可调用的函数列表给大模型
    # step3 大模型判断调用哪个函数，并传入参数，Pydantic会执行函数并把结果返回给大模型
    tools=TOOLS,  
    # 依赖注入关键：声明本 Agent 运行期间需要携带 ReadFileState 状态实例对象
    deps_type=ReadFileState,
    # 工作原理是middleware模式，hooks会在每次模型调用前后被触发，可以用来记录日志、修改输入输出等
    capabilities=[hooks],
)
