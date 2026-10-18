"""
Task 工具集：暴露给大模型的结构化任务拆解与状态管理接口。

核心设计：
1. 依赖注入与分层解耦 (RunContext[AgentDeps])：
   - 工具本身不直接操作文件，而是通过 PydanticAI 的依赖注入机制 (ctx.deps.tasks_store) 统一调用底层 TasksStore 账本。
   - 实现工具层（协议接口与 Prompt 导引）与存储层（文件持久化、缓存与 ID 递增算法）的彻底解耦。

2. Prompt 刷屏防范与 Token 优化 (_SUMMARY_FIELDS)：
   - task_list 接口设计为“摘要模式”，仅返回 ('id', 'subject', 'status', 'blocked_by') 四个核心字段。
   - 过滤掉包含长文本的 description 字段，防止 LLM 在每轮查询任务列表时将庞大的文本重复刷入上下文历史，节省 Token 开销。

3. 自我修正机制与容错拦截 (ModelRetry)：
   - 当 LLM 查询或更新不存在的 task_id，或者调用 task_update 时未传入任何待更新字段时，抛出 raise ModelRetry(...)。
   - 引导 LLM 进行自我修正（如“先调 task_list 确认已有任务”），避免无效操作导致状态错乱。

4. 状态伪信号路由 (deleted 机制)：
   - 扩充工具层状态 TaskUpdateStatus，引入伪状态 'deleted'。
   - 当 LLM 传入 status='deleted' 时，工具层自动路由拦截并触发 store.delete() 销毁文件，对 LLM 屏蔽底层的删除细节，保持接口简洁统一。
"""
from dataclasses import asdict
from typing import Literal

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry

from ..deps import AgentDeps

# 工具层允许传入的状态类型：前 3 个会真正存盘，'deleted' 只是触发删文件的信号
TaskUpdateStatus = Literal["pending", "in_progress", "completed", "deleted"]

# task_list 列表接口仅裁剪返回这 4 个字段，防止长文本 description 刷屏消耗 Token
_SUMMARY_FIELDS = ("id", "subject", "status", "blocked_by")


def task_create(
    ctx: RunContext[AgentDeps],
    subject: str,
    description: str,
    active_form: str | None = None,
) -> dict:
    """
    在 task 列表里建一条结构化 task。遇到 3 步以上的复杂任务、用户列了一串待办、或者你想跟踪进度向用户体现严谨时，主动用它建 task；只有 1 步的小事或纯聊天就不要建 task，建了反而碍事。

    新建的 task 默认 status='pending'。开始做某条之前用 task_update 切到 'in_progress'，做完立刻切 'completed'。

    Args:
        subject: 简短的祈使句标题（如 "修复登录 bug"）
        description: 这条 task 到底要做什么
        active_form: 切到 in_progress 时显示在状态行里的进行时短语（如 "正在修复登录 bug"）；不传就显示 subject
    """
    store = ctx.deps.tasks_store
    task_id = store.create(subject=subject, description=description, active_form=active_form)
    return {"id": task_id, "subject": subject}


def task_list(ctx: RunContext[AgentDeps]) -> list[dict]:
    """
    列出所有 task 的精简摘要。用来看还有什么待办、谁在进行中、谁已经完成——通常做完一条 task 后调一次，看下一条做哪个。

    优先按 id 从小到大做，前面的 task 往往为后面的搭好上下文。
    """
    return [{k: asdict(t)[k] for k in _SUMMARY_FIELDS} for t in ctx.deps.tasks_store.list()]


def task_get(ctx: RunContext[AgentDeps], task_id: str) -> dict:
    """
    取回单条 task 的完整内容：subject / description / status / blocks / blocked_by。开始做一条 task 之前调一次拿完整描述，或者用来理清依赖关系。

    Args:
        task_id: 要查询的 task id
    """
    task = ctx.deps.tasks_store.get(task_id)
    if task is None:
        raise ModelRetry(f"task #{task_id} 不存在，可以先 task_list 看看现在有哪些 task")
    return asdict(task)


def task_update(
    ctx: RunContext[AgentDeps],
    task_id: str,
    status: TaskUpdateStatus | None = None,
    subject: str | None = None,
    description: str | None = None,
    active_form: str | None = None,
    add_blocks: list[str] | None = None,
    add_blocked_by: list[str] | None = None,
) -> dict:
    """
    更新已有 task 的字段。最常见的用法是开工前切 status='in_progress'、做完立刻切 'completed'——做完了不要让 task 一直挂在 in_progress。

    传 status='deleted' 会直接把这条 task 删掉（JSON 文件被删，不会留下 status='deleted' 的记录）。用于一开始建错了、或者后来不再需要的 task。

    实现没做完、测试还在挂、或者遇到没解决的错误，不要切 'completed'，让它继续挂在 'in_progress'，并把卡住的问题告诉用户。

    Args:
        task_id: 要更新的 task id
        status: 'pending' / 'in_progress' / 'completed' / 'deleted'；'deleted' 会触发删除而不会被存进去
        subject: 新标题；不需要改就别传
        description: 新描述
        active_form: 新的进行时短语
        add_blocks: 哪些 task 阻塞在这条上（追加，不覆盖）
        add_blocked_by: 这条要等哪些 task 先做完（追加，不覆盖）
    """
    store = ctx.deps.tasks_store
    task = store.get(task_id)
    if task is None:
        raise ModelRetry(f"task #{task_id} 不存在，可以先 task_list 看看现在有哪些 task")

    # 'deleted' 是个伪状态信号，命中后直接删除文件，下面其他字段的改动都没意义
    if status == "deleted":
        store.delete(task_id)
        return {"success": True, "task_id": task_id, "deleted": True}

    # 收集本次真正要改的字段（None 表示「不传 = 不改」），再交给 store.update 落盘
    candidates = {
        "status": status,
        "subject": subject,
        "description": description,
        "active_form": active_form,
        "add_blocks": add_blocks,
        "add_blocked_by": add_blocked_by,
    }
    fields = {k: v for k, v in candidates.items() if v is not None}

    if not fields:
        raise ModelRetry("task_update 没传任何要改的字段，至少要传 status / subject / description / active_form / add_blocks / add_blocked_by 中的一个")

    updated = store.update(task_id, **fields)
    return {
        "success": True,
        "task_id": task_id,
        "updated_fields": list(fields.keys()),
        "task": asdict(updated),
    }
