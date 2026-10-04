"""
Agent 实例化：把 model / instructions / tools / hooks 拼起来。

配置大模型的地方，实例化一个agent对象（model + instructions + tools + hooks），然后在主循环里调用 agent.run()。
"""
import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

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
        "工作流程：先理解需求，写代码，然后运行验证。"
        "如果有错误就修复并重新运行，直到确认正确。"
        "你目前运行在一个windows环境"
    ),
    tools=TOOLS,  
    # step1 pydantic 会根据函数签名 + docstring 自动生成 JSON Schema
    # step2 agent 会根据 JSON Schema 生成可调用的函数列表给大模型
    # step3 大模型判断调用哪个函数，并传入参数，Pydantic会执行函数并把结果返回给大模型
    capabilities=[hooks],
    # 工作原理是middleware模式，hooks会在每次模型调用前后被触发，可以用来记录日志、修改输入输出等
)
