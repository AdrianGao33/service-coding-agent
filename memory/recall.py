"""
记忆召回：在用户刚输入完话、主 LLM 还没开始回答之前，先用一个极其便宜/快速的小模型去“挑菜”（从清单里挑出有用的记忆文件），然后再把文件内容打包成带有新鲜度警告的上下文注入给主 Agent

核心职责与工作机制：
1. 门槛过滤（Gatekeeping）：
   - 检查用户输入长度：若太短（< 4个字，如“好的”、“继续”）则直接跳过召回，节约 API 资源。
   - 会话内去重：跟踪记录已召回过（surfaced_memories）的文件，防止同一会话中重复注入相同的记忆。

2. 极简 LLM 二次筛选：
   - 独立使用DeepSeek-Flash，把用户最新输入与 store.py 格式化的【记忆清单（Manifest）】丢给它。
   - 秉持“宁缺毋滥（Fail-Open）”原则：让模型挑选出严格相关的记忆文件名（JSON 格式返回），若出错或无相关内容直接返回空列表。

3. 规范打包与注入（Injection & Persistence）：
   - 读取选中记忆文件的全文内容，经过 store.py 的截断保护机制防爆。
   - 自动计算保存时长：若属于“旧记忆（> 1天）”，在开头注入【新鲜度过期警告】。
   - 将记忆包装为 <system-reminder> 结构化消息，塞入当前对话历史中，并实时持久化到磁盘 session 文件中。
"""
import json
import os

from openai import AsyncOpenAI
from pydantic_ai.messages import ModelRequest, UserPromptPart

import session
from ui.render import print_step

from . import store

# 和 classifier 一样是主对话之外单独发起的请求，不经过 agent 框架
_client = AsyncOpenAI(
    api_key=os.environ.get("API_KEY"),
    base_url="https://api.deepseek.com",
)

RECALL_MODEL = "deepseek-flash"

# 每轮最多召回几条记忆
MAX_RECALL = 5
# 输入太短（如「继续」「好的」）多半是承接上文，不值得为它发一次召回请求
MIN_INPUT_CHARS = 4

SYSTEM_PROMPT = """你在为一个 coding agent 挑选记忆，帮它处理用户的最新输入。用户消息会给你这条输入，和一份可用记忆的清单（每行一个记忆文件，带类型、日期和描述）。

返回对处理这条输入明确有用的记忆文件名列表（最多 5 个）。只选那些凭文件名和描述就能确定有帮助的记忆。
- 拿不准某条记忆有没有用，就不要把它放进列表。要挑剔。
- 清单里没有明确有用的记忆，返回空列表就好。

只输出 JSON：{"selected_memories": ["文件名", ...]}
"""

# 单独调用判断用哪5条记忆文件做召回
async def _select(user_input: str, headers: list) -> list[str]:
    """
    【选菜逻辑】单独发起一次 LLM 请求，从元数据清单里挑出相关的记忆文件名。
    设计原则（Fail-Open）：召回属于锦上添花，如果网络请求失败或 JSON 解析报错，默默返回 []，不影响主流程。
    """
    manifest = store.format_manifest(headers)
    try:
        response = await _client.chat.completions.create(
            model=RECALL_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"最新输入：{user_input}\n\n记忆清单：\n{manifest}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        selected = json.loads(response.choices[0].message.content).get("selected_memories") or []
    except Exception:
        return []
    # 模型可能编造文件名，过滤成清单里真实存在的
    valid = {h.filename for h in headers}
    return [name for name in selected if isinstance(name, str) and name in valid][:MAX_RECALL]

# 打包读取记忆文件内容，打包提示词
def _build_reminder(header) -> str:
    """
    打包提示问：读取并把一个记忆文件包成 system-reminder 正文：相对保存时间（超过一天换成新鲜度警告）+ 文件路径 + 内容。
    """
    content = header.path.read_text(encoding="utf-8")
    content = store.truncate_content(
        content, store.MAX_MEMORY_LINES, store.MAX_MEMORY_BYTES,
        f"这条记忆被截断了，用 read_file 查看完整文件：{header.path}",
    )
    days = store.age_days(header.mtime)
    if days <= 1:
        head = f"记忆（保存于{store.age_text(days)}）：{header.path}："
    else:
        # 旧记忆的开头换成新鲜度警告，提醒模型先验证再使用
        head = f"{store.staleness_text(days)}\n\n记忆：{header.path}："
    return f"<system-reminder>\n{head}\n\n{content}\n</system-reminder>"


async def inject_memories(user_input: str, state) -> None:
    """
    【核心总入口】在每一轮用户发送消息时被主程序调用：触发场景有待确认todo
    完成：筛选未曝光记忆 -> 触发挑选 -> 格式化打包 -> 写入消息历史 -> 持久化写盘。
    """
    if len(user_input.strip()) < MIN_INPUT_CHARS:   #1. 用户输入太短，不做记忆召回动作
        return
    headers = [                                     #2. 已经召回过的记忆文件去重
        h for h in store.scan_memory_files()
        if h.filename not in state.surfaced_memories
    ]
    if not headers:                                 #3. 没有新的记忆文件，退出
        return
    selected = await _select(user_input, headers)   #4. 独立判断需要找回的记忆文件
    if not selected:                                #5. 判断不需要，退出
        return

    by_name = {h.filename: h for h in headers}      
    injected = []
    for filename in selected:                       #6. 选中的记忆文件，逐个读取，打包成reminder提示词
        header = by_name[filename]
        try:
            text = _build_reminder(header)
        except OSError:
            continue
        injected.append(ModelRequest(parts=[UserPromptPart(content=text)]))     #7. 包装成标准的ModelRequest模型消息块
        state.surfaced_memories.add(filename)                                   #8. 更新已召回的状态，防止之后重复召回
        # 终端回显召回了哪条记忆，让用户看到召回确实发生了
        print_step("[dim]◇ memory[/]", f"[dim]召回记忆：{filename}（{store.age_text(store.age_days(header.mtime))}）[/]")
    if not injected:
        return

    # 和 @ 引用一样塞进历史并持久化，/resume 恢复会话时召回过的记忆也能一起还原
    state.history += injected                                           #9. 添加到上下文历史
    session.append_messages(state.session_id, injected)                 #10. 添加到jsonl
