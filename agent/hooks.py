"""
挂在 Agent 上的 hooks，用来抓每次 model API 调用的元数据。

工作流：
1. 主循环在每轮 run_sync 运行前清空全局 api_call_log 数组。
2. 每次与大模型交互时，通过 Hook 自动拦截并在 api_call_log 填充一次调用的元数据。
3. 跑完后快照到 SessionState 里，供 /api-detail 斜杠命令展示给用户。

创建了一个ApiCall的类
用api_call_log来记录每次调用的元数据（这个目前是全局的，一次对话只记录上一次的）
两个hook中间件，在call前后分别记录请求和响应的元数据存到这个类里
"""
from dataclasses import dataclass, field
from typing import Any

# 从 pydantic_ai 中导入生命周期 Hook 管理类[cite: 3]
from pydantic_ai.capabilities import Hooks

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


# 主循环在每轮 run_sync 之前清空它 这边声明了在main.py里有一个全局的api_call_log
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
