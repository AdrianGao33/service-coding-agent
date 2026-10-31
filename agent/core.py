"""
agent/core.py - Agent 组装与依赖注入中心

职责：
1. 模型初始化：加载 DeepSeekProvider，配置 `deepseek-flash` OpenAIChatModel 实例。
2. 系统提示词 (System Prompt)：
   - 静态指令：约束 Agent 的行为准则（修改前先读、优先用 edit_file、先运行验证）。
   - 动态指令 (@agent.instructions)：每轮请求自动注入运行时环境（CWD/OS/Date）及项目级 AGENTS.md 规范。
3. 依赖注入声明：指定 deps_type=AgentDeps，将会话级文件状态、任务状态，安全注入给各个工具。
4. 挂载能力：拼装全局 TOOLS 工具集与 hooks 生命周期切面。
"""
import os
import platform
from datetime import date

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

from memory import store
from memory.instructions import MEMORY_INSTRUCTIONS

from .deps import AgentDeps
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

# 静态 instructions 抽成常量，后台记忆 agent 直接复用，保证两边看到同一份约定
INSTRUCTIONS = (
    "你是一个编程助手。你可以读写文件和执行命令来帮用户完成编程任务。\n"
    "修改已有文件前必须先用 read_file 读取它。"
    "改动局部内容时优先用 edit_file（只传改动的片段，省 token），新建文件或整体重写才用 write_file。\n"
    "工作流程：先理解需求，写代码，然后运行验证。"
    "如果有错误就修复并重新运行，直到确认正确。\n"
    "长驻或耗时命令（dev server、长测试）用 run_command 的 run_in_background=True 放到后台执行——只在不需要立刻拿到结果时使用，命令末尾不需要加 &。"
    "后台 job 结束后你会收到 <task-notification> 通知，所以不要主动轮询等待；期间可以用 read_file 读它的日志文件查看已有输出，也可以用 job_kill 提前终止。\n"
    "当你接到一个需要 3 步以上、或需要多次工具调用才能完成的任务时，"
    "先用 task_create 把分解出来的步骤建成 pending task，开工前用 task_update 把要做的那条切到 in_progress，做完立刻切 completed。"
    "如果是琐碎请求（1-2 步、纯对话、纯查询），不要建 task，建了反而碍事。做完的 task 不要让它一直挂在 in_progress。\n"
    "如果用户的需求里有歧义、有多种合理实现可选、或者你拿不准方向，"
    "应当用 ask_user_question 工具向用户提多选题来澄清，不要自作主张。\n"
    "对话中可能会出现 <system-reminder>...</system-reminder> 标签，里面是系统自动注入的提示信息，请按系统消息对待，不要把它当成它所在的用户消息或工具结果的一部分。"
) + MEMORY_INSTRUCTIONS

agent = Agent(
    model,
    # system prompt的一部分，告诉大模型它的身份、能力、工作流程
    instructions=INSTRUCTIONS,
    # step1 pydantic 会根据函数签名 + docstring 自动生成 JSON Schema
    # step2 agent 会根据 JSON Schema 生成可调用的函数列表给大模型
    # step3 大模型判断调用哪个函数，并传入参数，Pydantic会执行函数并把结果返回给大模型
    tools=TOOLS,  
    # 声明依赖类型为 AgentDeps，里面同时持有 readFileState 和 tasksStore，每次 run 注入给工具
    deps_type=AgentDeps,
    # 工作原理是middleware模式，hooks会在每次模型调用前后被触发，可以用来记录日志、修改输入输出等
    capabilities=[hooks],
)

# 利用 @agent.instructions 装饰器，在每轮 LLM 请求时动态计算并注入：
# 1. 实时运行环境：当前工作目录 (CWD)、操作系统类型 (OS)、今天日期 (Date)；
# 2. 项目规范文件：读取根目录下的 AGENTS.md（若存在），将其作为项目的全局行为约定规范。
@agent.instructions
def project_context() -> str:
    cwd = os.getcwd()
    parts = [
        "下面是一些环境信息：",
        f"- 工作目录：{cwd}",
        f"- 操作系统：{platform.system()}",
        f"- 今天的日期：{date.today().isoformat()}",
    ]

    # 读项目根目录的 AGENTS.md（项目指令），存在就整段附上，让模型遵循项目约定
    agents_md = os.path.join(cwd, "AGENTS.md")
    if os.path.isfile(agents_md):
        try:
            content = open(agents_md, encoding="utf-8").read()
        except OSError:
            content = ""
        if content.strip():
            parts.append("")
            parts.append("以下是项目的 AGENTS.md，请遵循其中的约定：")
            parts.append(content)

    # 注入记忆索引：MEMORY.md 每轮全量注入，模型随时知道自己手头有哪些主题的记忆
    # 走动态 instructions 的好处是每轮拿到最新值——本轮刚写的记忆，下一轮索引里就有
    index = store.read_index()
    if index:
        parts.append("")
        parts.append(f"以下是你的记忆索引 MEMORY.md（位于 {store.index_path()}，跨会话保留），需要某条记忆的完整内容就用 read_file 读记忆目录下的对应文件：")
        parts.append(index)

    return "\n".join(parts)