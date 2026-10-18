"""
Agent 依赖聚合器 (AgentDeps)：基于依赖注入模式的状态聚合容器。

核心设计：
1. 单一上下文类型约束 (deps_type Binding)：
   - PydanticAI 框架规范要求 Agent 的 deps_type 必须绑定单一类型标识。
   - 利用 dataclass 将会话生命周期内的强状态组件——文件状态管理器 (ReadFileState) 与任务持久化存储器 (TasksStore) 封装为统一的聚合类型 AgentDeps。

2. 状态上下文透传 (RunContext Pass-Through)：
   - 依赖项通过 RunContext[AgentDeps] 统一注入至各个工具函数 (Tools) 的执行上下文。
   - 保证了在多轮工具调用链中，底层 ReadFileState（过时文件检测）与 TasksStore（任务落盘与自增 ID）的单例引用与状态一致性。
"""
from dataclasses import dataclass

from tasks_store import TasksStore

from .file_state import ReadFileState


@dataclass
class AgentDeps:
    read_file_state: ReadFileState
    tasks_store: TasksStore
