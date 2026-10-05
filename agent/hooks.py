"""
挂在 Agent 上的 hooks，用来抓每次 model API 调用的元数据。

工作流：
1. 主循环在每轮 run_agent_loop 运行前清空全局 api_call_log 数组。
2. 每次与大模型交互时，通过 Hook 自动拦截并在 api_call_log 填充一次调用的元数据。
3. 跑完后快照到 SessionState 里，供 /api-detail 斜杠命令展示给用户。
4. 捕获模型层 HTTP/网络波动并进行指数退避重试。
5. 拦截工具执行时的未捕获异常，将其转化为提示文本供大模型自我纠错。

创建了一个ApiCall的类，用这个类创建了一个api_call_log的全局变量，来记录每次调用的元数据
两个hook中间件，在call前后分别记录请求和响应的元数据存到这个变量里
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any

# 从 pydantic_ai 中导入生命周期 Hook 管理类[cite: 3]
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ModelHTTPError, ModelAPIError

from ui.render import console

MAX_RETRIES = 3


@dataclass
class ApiCall:
    """
    一次 model API 调用的元数据。before_model_request 创建并填充上半部分，after_model_request 填充下半部分。
    """
    # request 侧
    model: str 
    messages_count: int #发送给模型的历史消息条数+当前消息条数
    # 这次发送给模型的 messages 中最后一条消息的最后一个 part
    last_part: Any
    tools: list 

    # response 侧（after hook 填充）
    finish_reason: str = ""
    parts_kinds: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


# 模块级全局共享单例列表，用来缓存当前这轮对话产生的 API 调用记录
api_call_log: list[ApiCall] = []

hooks = Hooks()

#ctx是单次用户提问的上下文对象（这里可以多轮次call llm api)，request_context，response是单轮次api请求上下文对象
@hooks.on.before_model_request
async def _record_request(ctx, request_context):
    """
    创建+填上半段：每次发起 model 调用之前，创建一条 ApiCall 记录。
    """
    msgs = list(request_context.messages)
    last_part = msgs[-1].parts[-1] if msgs and msgs[-1].parts else None
    try:
        tool_names = [t.name for t in request_context.model_request_parameters.function_tools]
    except AttributeError:
        tool_names = []

    api_call_log.append(ApiCall(
        model=request_context.model.model_name,
        messages_count=len(msgs),
        last_part=last_part,
        tools=tool_names,
    ))
    return request_context


@hooks.on.after_model_request
async def _record_response(ctx, request_context, response):
    """
    填充下半段：每次 model 调用返回后，填充上面这条 ApiCall 的 response 字段。
    """
    if api_call_log:
        call = api_call_log[-1]
        call.finish_reason = str(response.finish_reason) if response.finish_reason else "unknown"
        call.parts_kinds = [p.part_kind for p in response.parts]
        call.input_tokens = response.usage.input_tokens
        call.output_tokens = response.usage.output_tokens
    return response

# ---------- API 请求重试 ----------

@hooks.on.model_request
async def _retry_on_error(ctx, *, request_context, handler):
    """
    包裹 model 请求，遇到可重试错误时自动指数退避重试。

    重试在 wrap 内部完成，对话历史和 before/after hooks 不受影响。
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await handler(request_context)
        except ModelHTTPError as e:
            if e.status_code < 500: # 客户端错误（如 400 Bad Request, 401 Unauthorized, 404），重试无意义，直接抛出
                raise
            if attempt >= MAX_RETRIES:  # 如果达到最大重试次数，打印错误终端日志并向上抛出异常
                console.print(f"[bold red]✗ HTTP {e.status_code}，重试 {MAX_RETRIES} 次后仍失败[/]")
                raise
            wait = 2 ** attempt # 指数退避等待时长：2^0=1s, 2^1=2s, 2^2=4s...
            console.print(
                f"[bold yellow]⟳ HTTP {e.status_code}，{wait}s 后重试 "
                f"({attempt + 1}/{MAX_RETRIES})...[/]"
            )
            await asyncio.sleep(wait)
        except ModelAPIError as e:
            if attempt >= MAX_RETRIES: # 捕获网络连接超时、DNS 解析失败等底层 API 异常
                console.print(
                    f"[bold red]✗ 网络连接失败，重试 {MAX_RETRIES} 次后仍无法连接[/]"
                )
                raise
            wait = 2 ** attempt
            console.print(
                f"[bold yellow]⟳ 网络连接失败，{wait}s 后重试 "
                f"({attempt + 1}/{MAX_RETRIES})...[/]"
            )
            await asyncio.sleep(wait)


# ---------- 工具执行异常兜底 ----------

@hooks.on.tool_execute_error
async def _handle_tool_error(ctx, *, call, tool_def, args, error):
    """
    工具函数抛出未捕获异常时，不让进程崩溃，本地 Python 工具函数在执行过程中抛出未捕获异常（如 FileNotFoundError, KeyError 等）时，
    而是把错误信息作为 tool result 返回给模型，让它自行纠正。
    """
    console.print(f"[bold red]✗ 工具 {call.tool_name} 出错：{error}[/]")   # 1. 在终端友好打印警告日志
    return f"工具执行出错：{type(error).__name__}: {error}" # 2. 将异常转换为字符串格式的 tool_return，降级发还给大模型