"""
Agent 生命周期钩子与切面拦截器 (Hooks & Middleware System)

四大核心职能：
1. 元数据审计：在模型请求前后，记录 Token 消耗、工具列表与 finish_reason，打入全局 api_call_log

2. 网络容错重试：遇到 HTTP 5xx 或网络超时，以 2^n 秒指数退避自动重试（最多 3 次）

3. 工具权限拦截：工具执行前，调用 permissions 模块弹窗审批，决定放行或拒绝

4. 工具崩溃兜底：捕获本地 Python 工具运行异常，降级为错误字符串还给大模型，防主进程崩塌
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
    单次 API 调用的元数据结构：before 钩子填上半段（请求侧），after 钩子填下半段（响应侧）
    """
    # 请求侧元数据
    model: str 
    messages_count: int # 发送给模型的历史+当前消息总条数
    last_part: Any      # 发送消息中最后一条的最后一个 part
    tools: list         # 本次允许调用的工具清单

    # 响应侧元数据（after 钩子回填）
    finish_reason: str = ""
    parts_kinds: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


# 全局单例日志 Buffer：缓存当前轮次产生的 API 调用记录，供 UI 界面渲染
api_call_log: list[ApiCall] = []

hooks = Hooks()

# ------------------------------------------------------------------------------
# 1. API 统计与元数据追踪 Hooks
# ------------------------------------------------------------------------------

# ctx是单次用户提问的上下文对象（这里可以多轮次call llm api)，request_context，response是单轮次api请求上下文对象
# request_context：准备塞给大模型的数据（包含 Prompt、历史消息、工具列表等）
@hooks.on.before_model_request
async def _record_request(ctx, request_context):
    """
    请求前钩子：创建 ApiCall 记录并填充请求侧元数据（模型名、消息数、工具列表）
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

# response：大模型 API 刚返回的原始结果（包含生成文本、Token 消耗、finish_reason）。
@hooks.on.after_model_request
async def _record_response(ctx, request_context, response):
    """
    响应后钩子：回填最新 ApiCall 记录的响应元数据（Token 消耗、结束原因等）
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

# handler：包裹模型请求的底层网络调用句柄。
@hooks.on.model_request
async def _retry_on_error(ctx, *, request_context, handler):
    """
    请求包裹钩子：拦截 HTTP 5xx 与网络超时异常，按 1s -> 2s -> 4s 指数退避重试
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
        call: 包含了当前工具调用信息的对象 (如 call.tool_name：pydantic转换从tools那边传过来的工具名称)
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

# error：本地 Python 工具代码抛出的原始异常对象（如 FileNotFoundError）。
@hooks.on.tool_execute_error
async def _handle_tool_error(ctx, *, call, tool_def, args, error):
    """
    工具函数抛出未捕获异常时，不让进程崩溃，本地 Python 工具函数在执行过程中抛出未捕获异常（如 FileNotFoundError, KeyError 等）时，
    而是把错误信息作为 tool result 返回给模型，让它自行纠正。
    """
    console.print(f"[bold red]✗ 工具 {call.tool_name} 出错：{error}[/]")   # 1. 在终端友好打印警告日志
    return f"工具执行出错：{type(error).__name__}: {error}" # 2. 将异常转换为字符串格式的 tool_return，降级发还给大模型