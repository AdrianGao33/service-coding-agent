"""
Agent 生命周期钩子与切面拦截器 (Hooks & Middleware System)

四大核心职能：
1. 元数据审计：在模型请求前后，记录 Token 消耗、工具列表与 finish_reason，打入全局 api_call_log
2. 网络容错重试：遇到 HTTP 5xx 或网络超时，以 2^n 秒指数退避自动重试（最多 3 次）
3. 工具权限拦截：工具执行前，调用 permissions 模块弹窗审批，决定放行或拒绝，和 classifier 模块协作，自动审查工具调用安全性
4. 工具崩溃兜底：捕获本地 Python 工具运行异常，降级为错误字符串还给大模型，防主进程崩塌
5. 动态提醒注入：在模型请求前 (before_model_request)，将过时文件的 <system-reminder> 提示无缝注入上下文，引导模型重读
"""
import asyncio
import dataclasses
from dataclasses import dataclass, field
from typing import Any

# 从 pydantic_ai 中导入生命周期 Hook 管理类[cite: 3]
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ModelHTTPError, ModelAPIError
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart

import classifier
import permissions
from ui.render import console, print_step

from .deps import AgentDeps
from .reminders import build_file_reminder_text, build_task_reminder_text

MAX_RETRIES = 3

# task_reminder 触发阈值
# 模型连续几轮没用 task 工具才考虑提醒
TASK_REMINDER_TURNS_SINCE_WRITE = 6
# 两次 task 提醒之间至少间隔几轮，防止刷屏
TASK_REMINDER_TURNS_BETWEEN = 4

# 哪些工具算"task 管理动作"——调过它们就重置 since_write 计数
_TASK_MANAGEMENT_TOOLS = {"task_create", "task_update"}

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
# 2. system-reminder 注入
# ------------------------------------------------------------------------------

def _scan_task_turn_counters(messages) -> tuple[int, int]:
    """
    反向扫描对话历史，一次性算出：
    1. since_mgmt: 模型已经连续几轮没调过 task_create / task_update 了
    2. since_reminder: 系统已经连续几轮没给模型发过 Task 提醒了
    """
    since_mgmt = 0
    since_reminder = 0
    found_mgmt = False
    found_reminder = False
    # 倒序遍历每一个 message（从最新到最老）
    for msg in reversed(messages):          

        if isinstance(msg, ModelResponse):  
            if not found_mgmt:
                # 【内层小循环 A】：用来找这一条 ModelResponse 里有没有 task 工具
                for part in msg.parts:
                    if isinstance(part, ToolCallPart) and part.tool_name in _TASK_MANAGEMENT_TOOLS:
                        found_mgmt = True
                        break   # break from loop A

                if not found_mgmt:
                    since_mgmt += 1
            if not found_reminder:
                since_reminder += 1

        # 扫描发送给模型的请求，找有没有包含过 task 提醒文案        
        elif isinstance(msg, ModelRequest) and not found_reminder:
            # 【内层小循环 B】：用来找这一条 ModelRequest 里有没有提醒文案
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str) and _REMINDER_SENTINELS["task"] in part.content:
                    found_reminder = True
                    break   # break from loop B

        # 两个计数点都找到了，直接退出循环，不浪费CPU遍历更老的历史
        if found_mgmt and found_reminder:
            break
    return since_mgmt, since_reminder


def _build_file_reminder(ctx, messages) -> str | None:
    """过期文件提醒构建器：检测 ReadFileState 账本是否有文件被外部改过"""
    return build_file_reminder_text(ctx.deps.read_file_state)


def _build_task_reminder(ctx, messages) -> str | None:
    """Task 提醒构建器：只有在沉默轮数和上次提醒间隔均满足冷却阈值时才触发"""
    since_mgmt, since_reminder = _scan_task_turn_counters(messages)
    # 1. 距离上次用 task 工具还不够久，不提醒
    if since_mgmt < TASK_REMINDER_TURNS_SINCE_WRITE:
        return None
    # 2. 距离上次发 task 提醒还不够久（还在冷却期），不提醒
    if since_reminder < TASK_REMINDER_TURNS_BETWEEN:
        return None
    return build_task_reminder_text(ctx.deps.tasks_store)


# 注册要在 before_model_request 触发的 reminder builder：每条 (sentinel, builder)，sentinel 仅用于回扫识别（task reminder 复用）
_REMINDER_SENTINELS = {
    # task reminder 的识别串就是它正文里 builder 必定带的那句话，不再单独嵌一个 marker
    "task": "task 工具最近没有被使用",
}
_REMINDER_BUILDERS = (_build_file_reminder, _build_task_reminder)


@hooks.on.before_model_request
async def _inject_reminders(ctx, request_context):
    """
    合并触发提醒
    依次跑每个注册过的 reminder builder：返回非 None 的就拼到 request_context.messages 末尾。
    当前两条：过期文件提醒（看 readFileState）+ task 提醒（看 turn 计数 + tasks_store）。
    要加第 3 种 reminder 时只需多写一个 builder，不必再复制一次 hook 框架。
    """
    messages = list(request_context.messages)
    appended = []
    for builder in _REMINDER_BUILDERS:
        text = builder(ctx, messages + appended)
        if text is None:
            continue
        appended.append(ModelRequest(parts=[UserPromptPart(content=text)]))
        print_step("[dim]◇ system[/]", f"[dim]{text[:200]}[/]")
    # 如果没有任何提醒需要追加，原封不动放行
    if not appended:
        return request_context
    # 将所有生成的新提醒追加到消息列表末尾，返回更新后的 request_context
    return dataclasses.replace(request_context, messages=messages + appended)

# ------------------------------------------------------------------------------
# 3. API 请求网络指数退避重试
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
# 4. 工具执行前置权限拦截网 
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

 # 当规则要求询问 ("ask") 且处于 AUTO 模式：交由 classifier 智能审查
    if permissions.state.mode == permissions.AUTO:
        # 审查过程对齐成和 thinking、tool_call 一样的「图标 + 标签独占一行、内容换行」格式，用蓝色让用户一眼看到 classifier 在工作
        print_step("[blue]◆ auto_check[/]", f"[blue dim]正在请 LLM 审查 {call.tool_name}...[/]")
        verdict = await classifier.classify(ctx.messages, call.tool_name, args)

        if verdict.get("error"):
            # 分类器异常：打印警告，降级流转至下方的人工弹窗审批
            print_step("[yellow]⚠ auto_check[/]", f"[yellow]{verdict['reason']}[/]")
        elif not verdict["should_block"]:
            # 分类器判定安全：自动放行并打印依据理由
            print_step("[blue]✔ auto_check[/]", f"[blue dim]放行：{verdict['reason']}[/]")
            return await handler(args)
        else:
            # 分类器判定危险：打断执行，将拦截理由作为提示词发还模型，要求其停止并请示用户
            print_step("[red]✘ auto_check[/]", f"[red]拦截：{verdict['reason']}[/]")
            return (
                f"安全检查拦截了这次 {call.tool_name} 调用，没有执行。拦截理由：{verdict['reason']}。"
                "不要尝试绕过拦截，请停下来向用户说明情况，由用户决定接下来怎么做。"
            )

    # 规则要求询问且处于 DEFAULT 模式（或 AUTO 模式分类器出错降级）：弹出终端单选 UI 供用户人工审批
    choice = await permissions.prompt_approval(call.tool_name, args)
    if choice == "once":
        return await handler(args)
    if choice == "always":
        permissions.state.session_allowed.add(call.tool_name) # 记进会话白名单，本会话内这个工具不再询问
        return await handler(args)

    # 拒绝：不执行工具，把拒绝原因回填给模型，让它停下来等用户发话，而不是自作主张绕过去
    return f"用户拒绝了对 {call.tool_name} 的调用，这次调用没有执行。请停下手上的事，等用户告诉你接下来该怎么做。"


# ------------------------------------------------------------------------------
# 5. 工具未捕获异常兜底 (Fallback)
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