"""
核心逻辑：用一次 LLM 调用将整段历史提炼为结构化摘要，再把1摘要 + 2最近读过的文件，拼装成一份轻量化的全新历史

1. 双重保障（摘要 + 完整存档）：在重写当前会话前，先把完整对话存档到磁盘（transcript_path）。如果摘要里漏了细节，模型可以随时去读完整存档。

2. 文件状态恢复 (restore_file_messages)：摘要无法完美还原文件原始内容，因此程序会自动挑出最近读过的几个文件（在字符与文件数量预算内），伪装成 @ 引用 的 read_file 重新塞进新历史。

3. 水位线自适应机制 (auto_compact_if_needed)：在每次发送请求前检查 Token 占用，一旦越过水位线便自动触发压缩，同时具备熔断保护（连续失败多次则不再重试）。
"""
import os
import re

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, UserPromptPart
from rich.markdown import Markdown
from rich.padding import Padding

import session
from agent.core import model
from agent.file_state import ReadFileState
from mentions import build_mention_messages
from ui.render import console

# 模型的上下文窗口大小（最大 Token 限制：即为单次请求大模型所能接受的最大输入+模型回复总上限）
CONTEXT_WINDOW = 131_072
# 给压缩调用预留的<摘要输出空间>（防止输出时把窗口撑爆）
COMPACT_OUTPUT_RESERVE = 20_000
# 水位线之前再留的<安全余量>（留点缓冲区，免得刚到边缘极端情况）
AUTO_COMPACT_BUFFER = 10_000
# 自动压缩连续失败这么多次后，本会话不再尝试（熔断机制，防止死循环报错）
MAX_COMPACT_FAILURES = 3
# 压缩后恢复最近读过的文件：最多恢复几个、总字符预算多少（控制恢复的文件体积）
RESTORE_MAX_FILES = 5
RESTORE_MAX_CHARS = 30_000


COMPACT_PROMPT = (
    "重要：只输出纯文本，不要调用任何工具。\n\n"
    "- 不要使用 read_file、edit_file、write_file、run_command 或其他任何工具。\n"
    "- 上面的对话里已经有你需要的全部信息。\n"
    "- 工具调用会被拒绝，还会浪费你唯一的回合——你将无法完成任务。\n"
    "- 你的整个回复必须是纯文本：先一个 <analysis> 块，再一个 <summary> 块。\n\n"
    "你的任务是为到目前为止的对话写一份详细摘要，重点关注用户的明确请求和你已经做过的事情。摘要要保留足够的技术细节，让后续工作能基于它无缝继续。\n"
    "这条压缩指令本身不属于要总结的对话——任何段落都不要提到或引用它，不要把它算作用户消息，也不要把写摘要说成当前工作。\n\n"
    "写正式摘要之前，先在 <analysis> 标签里打草稿整理思路：按顺序过一遍对话，核对用户的请求、你的处理方式、关键决策、文件名和代码片段、报错和修复过程。\n\n"
    "然后在 <summary> 标签里输出正式摘要，包含以下几段：\n\n"
    "1. 主要请求和意图：详细记录用户的所有明确请求和意图\n"
    "2. 关键技术概念：列出涉及的重要技术概念、框架和方案\n"
    "3. 文件和代码：列出查看过、修改过、新建过的具体文件和代码段，附上关键代码片段\n"
    "4. 错误与修复：列出遇到过的报错和修复方法，特别注意用户要求你换种做法的反馈\n"
    "5. 全部用户消息：一字不改地列出所有用户消息（工具结果和这条压缩指令都不算），不要转述——它们是理解用户反馈和意图变化的关键\n"
    "6. 当前工作与下一步：精确描述写摘要前正在做的事情和紧接着的下一步，逐字引用最近几条对话原文，确保接续时不跑偏\n\n"
    "再次提醒：不要调用任何工具。只输出纯文本——先 <analysis> 块，后 <summary> 块。"
)

# 最终拼装进新历史的摘要外壳模板
SUMMARY_WRAPPER = (
    "本会话由一段因上下文写满而被压缩的对话延续而来，以下是之前对话的摘要：\n\n"
    "{summary}\n\n"
    "如果摘要里缺少你需要的细节（具体代码片段、报错原文等），可以用 read_file 读取压缩前的完整对话记录：{transcript}\n"
    "请基于这份摘要继续工作。不要向用户复述摘要内容，直接从对话中断的地方接着干。"
)


# 压缩专用 agent：不注册任何工具，模型想调也调不了（纯干活的打工人）
summarizer = Agent(model)

# 知识点：你的输入和大模型的回复，共用一个额度，此消彼长
# 所以计算结果是“用于本次压缩的历史内容输入token的最大值”
def compact_threshold() -> int:
    """
    触发自动压缩的水位线：从窗口上限往回退掉摘要输出预留，再退一段安全余量。
    """
    return CONTEXT_WINDOW - COMPACT_OUTPUT_RESERVE - AUTO_COMPACT_BUFFER

# 估算目前History的总量用多少了
def context_tokens(history) -> int:
    """
    用最近一条模型回复的 usage 估算当前上下文占用：
    input 覆盖当轮发出去的全部内容，再加上output它生成的部分。
    """
    for msg in reversed(history):
        if msg.kind == "response" and msg.usage.input_tokens:
            return msg.usage.input_tokens + msg.usage.output_tokens
    return 0


def extract_summary(text: str) -> str:
    """
    先整块剥掉 <analysis> 草稿，再取 <summary> 块的内容。
    先剥再取是有讲究的：草稿里可能字面提到 <summary> 标签，会带偏匹配。
    """
    text = re.sub(r"<analysis>.*?</analysis>", "", text, flags=re.DOTALL)
    match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
    text = match.group(1) if match else text
    # 正文里残留的标签字样一并清掉
    return re.sub(r"</?(analysis|summary)>", "", text).strip()

# 压缩后，新起点的用的，开头userpromt发送给llm，让他更好承上启下
def build_summary_message(summary: str, transcript_path) -> ModelRequest:
    """
    把提炼好的摘要包装成一条标准的“用户消息”，作为重构后新历史的唯一起点。
    """
    content = SUMMARY_WRAPPER.format(summary=summary, transcript=transcript_path)
    return ModelRequest(parts=[UserPromptPart(content=content)])

# 恢复已读状态挑选5个最近读过的（可通过环境变量更改上限）
def restore_file_messages(old_state: ReadFileState, new_state: ReadFileState) -> list:
    """
    重新注入最近读过的文件：因为文件原文没法凭空从摘要里还原，
    所以由程序从旧状态里挑选最近读过的文件带上，并且加上严格的字符和数量预算。
    """
    picked, used = [], 0
    for path in reversed(old_state.paths()):
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if len(picked) >= RESTORE_MAX_FILES or used + size > RESTORE_MAX_CHARS:
            break
        picked.append(path)
        used += size
    # 恢复顺序保持原来的读取顺序
    picked.reverse()
    # 和 @ 引用同一套机制：伪装成 read_file 调用塞进历史，顺带登记进新 readFileState
    return build_mention_messages(picked, new_state)

# command触动压缩的主流程
async def run_compact(state, custom_instructions: str = "") -> None:
    """
    核心：完整的压缩流程。
    一次 LLM 调用总结全部历史，然后用摘要 + 恢复的文件整体替换成一份精简的新历史。
    """
    if not state.history:
        console.print("(当前会话还没有对话内容，无需压缩)\n")
        return

    prompt = COMPACT_PROMPT
    extra = custom_instructions.strip()
    if extra:
        prompt += "\n\n补充要求：\n" + extra

    console.print("正在压缩上下文，可能需要一会儿...\n")

    # 1. 让大模型干活，把现有历史交进去，让它吐出结构化摘要
    result = await summarizer.run(prompt, message_history=state.history)
    summary = extract_summary(result.output)

    # 2. 压缩调用本身的开销也计入会话累计
    usage = result.usage
    state.input_tokens += usage.input_tokens
    state.output_tokens += usage.output_tokens

    # 3. 重写会话文件之前先存档完整对话记录，摘要还原不了的细节模型可以回头去读
    transcript_path = session.archive_session(state.session_id)

    # 4. 换新 readFileState：旧登记随旧历史一起作废，恢复的文件会重新登记进来
    old_state = state.read_file_state
    state.read_file_state = ReadFileState()
    restored = restore_file_messages(old_state, state.read_file_state)

    # 5. 组装新历史：第一条是摘要消息，后面跟着恢复出来的最近读过的文件消息
    state.history = [build_summary_message(summary, transcript_path)] + restored
    
    # 6. 用新历史整体重写磁盘上的会话文件
    session.rewrite_messages(state.session_id, state.history)

    # 7. 旧检查点的 history_index 指向的消息已经被压缩掉了，所以整体丢弃清理
    fh = state.file_history
    if fh is not None and fh.checkpoints:
        fh.drop_from(fh.checkpoints[0])

    # 8. 已召回的记忆随旧历史消失，允许重新召回；进程内 API 调用记录一并清空
    state.surfaced_memories = set()
    state.last_api_calls.clear()

    # 9. 打印成功提示和摘要预览
    console.print(f"[magenta]✻ 压缩完成：压缩前上下文 {usage.input_tokens:,} tokens，已重建为一份摘要 + {len(restored) // 2} 个最近读过的文件[/]")
    console.print(f"[dim]完整对话记录已存档：{transcript_path}[/]")
    console.print(Padding(Markdown(summary), (0, 0, 0, 2)), style="dim")
    console.print()

# 自动压缩，超过水位线自动触发
async def auto_compact_if_needed(state) -> None:
    """
    自动监控：每次给 LLM 发请求前调用。
    上下文一旦越过水位线就自动触发压缩；如果失败了只提示不阻断，连续失败多次就启动熔断不再重试。
    """
    if state.compact_failures >= MAX_COMPACT_FAILURES:
        return
    used = context_tokens(state.history)
    threshold = compact_threshold()
    if used < threshold: #还没到水位线
        return
    console.print(f"[yellow]上下文接近上限（{used:,} / {threshold:,} tokens），自动压缩中[/]\n")
    try:
        await run_compact(state)
        state.compact_failures = 0
    except Exception as e:
        state.compact_failures += 1
        console.print(f"[yellow]自动压缩失败（{type(e).__name__}: {e}），继续对话[/]\n")
        if state.compact_failures >= MAX_COMPACT_FAILURES:
            console.print("[yellow]自动压缩已连续失败多次，本会话不再尝试[/]\n")
