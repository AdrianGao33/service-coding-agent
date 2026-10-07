"""
Agent 生命周期钩子与切面拦截器 (Hooks & Middleware System)

职责涵盖四大核心领域：
1. 监控与元数据审计 (Audit & Metrics):
   在 model 请求前后截获并记录 Token 消耗、消息条数、调用工具列表及响应 finish_reason，
   全量缓存至全局 api_call_log 供前端/控制台渲染展示。

2. API 网络容错与指数退避重试 (Resilience & Retry):
   利用 @hooks.on.model_request 拦截底层 HTTP 5xx 服务器错误及网络超时异常，
   自动进行 2^n 秒的指数退避重试，保证高可用性。

3. 工具执行前置权限拦截 (Permission Gate Control):
   利用 @hooks.on.tool_execute 拦截每一次工具调用。结合 permissions 模块进行
   计算决策与终端交互弹窗，决定放行、记住白名单或安全拒绝。

4. 工具未捕获异常兜底 (Fallback & Graceful Error Handling):
   利用 @hooks.on.tool_execute_error 捕获本地 Python 工具运行时的崩溃异常，
   将其转化为标准的错误响应文本吞下并降级还给大模型，避免主进程崩溃。
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any

# 从 pydantic_ai 中导入生命周期 Hook 管理类[cite: 3]
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ModelHTTPError, ModelAPIError

import permissions
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
    last_part: Any # 这次发送给模型的 messages 中最后一条消息的最后一个 part
    tools: list # 这次调用中使用的工具列表

    # response 侧（after hook 填充）
    finish_reason: str = ""
    parts_kinds: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


# 模块级全局共享单例列表，用来缓存当前这轮对话产生的 API 调用记录
api_call_log: list[ApiCall] = []

hooks = Hooks()

# ------------------------------------------------------------------------------
# 1. API 统计与元数据追踪 Hooks
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# 2. API 请求网络指数退避重试
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# 3. 工具执行前置权限拦截网 
# ------------------------------------------------------------------------------

@hooks.on.tool_execute
async def _check_permission(ctx, *, call, tool_def, args, handler):
    """
    【工具切面钩子】在任意 Python 本地工具函数真正执行之前触发。

    参数说明:
        call: 包含了当前工具调用信息的对象 (如 call.tool_name)
        tool_def: 工具的 Schema 定义对象
        args (dict): 大模型解析出来的工具输入参数
        handler (Callable): 真正执行底层 Python 工具的回调句柄！只有调用 handler(args) 才算真正执行工具

    返回说明:
        返回工具的实际执行结果，或者返回拒绝说明字符串给大模型。
    """
    decision = permissions.compute_decision(call.tool_name, args)
    if decision == "allow":
        # 放行，handler(args) 才是真正执行工具的那一步
        return await handler(args)

    # decision == "ask"，弹审批让用户决定
    choice = await permissions.prompt_approval(call.tool_name, args)
    if choice == "once":
        return await handler(args)
    if choice == "always":
        # 记进会话白名单，本会话内这个工具不再询问
        permissions.state.session_allowed.add(call.tool_name) # 动态将工具名注入到当前进程的状态白名单中
        return await handler(args)

    # 拒绝：不执行工具，把拒绝原因回填给模型，让它停下来等用户发话，而不是自作主张绕过去
    return f"用户拒绝了对 {call.tool_name} 的调用，这次调用没有执行。请停下手上的事，等用户告诉你接下来该怎么做。"


# ------------------------------------------------------------------------------
# 4. 工具未捕获异常兜底 (Fallback)
# ------------------------------------------------------------------------------

@hooks.on.tool_execute_error
async def _handle_tool_error(ctx, *, call, tool_def, args, error):
    """
    工具函数抛出未捕获异常时，不让进程崩溃，本地 Python 工具函数在执行过程中抛出未捕获异常（如 FileNotFoundError, KeyError 等）时，
    而是把错误信息作为 tool result 返回给模型，让它自行纠正。
    """
    console.print(f"[bold red]✗ 工具 {call.tool_name} 出错：{error}[/]")   # 1. 在终端友好打印警告日志
    return f"工具执行出错：{type(error).__name__}: {error}" # 2. 将异常转换为字符串格式的 tool_return，降级发还给大模型