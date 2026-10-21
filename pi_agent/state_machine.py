"""
状态机 — Pi-Agent Layer 2
==========================
Agent 生命周期状态及合法转移：

  IDLE → RUNNING → TOOL_EXECUTING → COMPRESSING → DONE
            ↑                                    |
            └────────────────────────────────────┘ (恢复时)

每个状态转移可注册进入/退出钩子函数。
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Callable, Optional


class AgentState(Enum):
    IDLE = auto()
    RUNNING = auto()
    TOOL_EXECUTING = auto()
    COMPRESSING = auto()
    DONE = auto()
    ERROR = auto()


VALID_TRANSITIONS = {
    AgentState.IDLE:           {AgentState.RUNNING, AgentState.DONE},
    AgentState.RUNNING:        {AgentState.TOOL_EXECUTING, AgentState.COMPRESSING, AgentState.DONE, AgentState.ERROR},
    AgentState.TOOL_EXECUTING: {AgentState.RUNNING, AgentState.DONE, AgentState.ERROR},
    AgentState.COMPRESSING:    {AgentState.RUNNING, AgentState.ERROR},
    AgentState.DONE:           set(),
    AgentState.ERROR:          {AgentState.IDLE},
}


class StateMachine:
    """Agent 状态机，含钩子系统。

    在状态转移前后触发 enter/exit 钩子，支持按状态注册或全局注册。
    """

    def __init__(self, initial_state: AgentState = AgentState.IDLE):
        self._state = initial_state
        self._previous: Optional[AgentState] = None
        self._enter_hooks: dict = {s: [] for s in AgentState}
        self._exit_hooks: dict = {s: [] for s in AgentState}
        self._any_hooks: list = []

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def previous(self) -> Optional[AgentState]:
        return self._previous

    @property
    def is_terminal(self) -> bool:
        return self._state in (AgentState.DONE, AgentState.ERROR)

    def on_enter(self, state: AgentState, callback: Callable[[], None]) -> None:
        self._enter_hooks[state].append(callback)

    def on_exit(self, state: AgentState, callback: Callable[[], None]) -> None:
        self._exit_hooks[state].append(callback)

    def on_any(self, callback: Callable[[AgentState, AgentState], None]) -> None:
        self._any_hooks.append(callback)

    def transition(self, to_state: AgentState) -> bool:
        if to_state not in VALID_TRANSITIONS.get(self._state, set()):
            return False
        from_state = self._state
        for hook in self._exit_hooks.get(from_state, []):
            try: hook()
            except Exception: pass
        self._previous = from_state
        self._state = to_state
        for hook in self._any_hooks:
            try: hook(from_state, to_state)
            except Exception: pass
        for hook in self._enter_hooks.get(to_state, []):
            try: hook()
            except Exception: pass
        return True
