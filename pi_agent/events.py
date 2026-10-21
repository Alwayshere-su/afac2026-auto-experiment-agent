"""
事件驱动架构 — Pi-Agent Layer 2
================================
基于发布/订阅模式的事件总线，10 种事件类型：

  agent_start, turn_start, message_start, message_update, message_end,
  tool_execution_start, tool_execution_update, tool_execution_end,
  turn_end, agent_end

监听器可按事件类型注册或全局注册。事件携带 payload 字典传递上下文数据。
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List

# ── Event types (mirrors Pi-Agent's 10 event types) ──
EVENT_AGENT_START = "agent_start"
EVENT_AGENT_END = "agent_end"
EVENT_TURN_START = "turn_start"
EVENT_TURN_END = "turn_end"
EVENT_TOOL_START = "tool_execution_start"
EVENT_TOOL_UPDATE = "tool_execution_update"
EVENT_TOOL_END = "tool_execution_end"

ALL_EVENTS = [
    EVENT_AGENT_START, EVENT_AGENT_END,
    EVENT_TURN_START, EVENT_TURN_END,
    EVENT_TOOL_START, EVENT_TOOL_UPDATE, EVENT_TOOL_END,
]


class Event:
    """A single event with type, payload, and timestamp."""

    def __init__(self, event_type: str, payload: Dict[str, Any] = None):
        self.type = event_type
        self.payload = payload or {}
        self.timestamp = time.time()

    def __repr__(self) -> str:
        return f"Event({self.type}, payload_keys={list(self.payload.keys())})"


# Type alias for listener callbacks
Listener = Callable[[Event], None]


class EventBus:
    """
    Synchronous pub/sub event bus.

    Usage:
        bus = EventBus()
        bus.on(EVENT_TOOL_START, lambda e: print(f"Tool: {e.payload['name']}"))
        bus.emit(Event(EVENT_TOOL_START, {"name": "read_file"}))
    """

    def __init__(self):
        self._listeners: Dict[str, List[Listener]] = {t: [] for t in ALL_EVENTS}
        self._global_listeners: List[Listener] = []
        self._history: List[Event] = []

    def on(self, event_type: str, callback: Listener) -> None:
        """Register a listener for a specific event type."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)

    def on_any(self, callback: Listener) -> None:
        """Register a listener for ALL events."""
        self._global_listeners.append(callback)

    def off(self, event_type: str, callback: Listener) -> None:
        """Remove a specific listener."""
        if event_type in self._listeners:
            self._listeners[event_type] = [
                cb for cb in self._listeners[event_type] if cb is not callback
            ]

    def emit(self, event: Event) -> None:
        """Emit an event to all registered listeners (synchronous)."""
        self._history.append(event)
        # Global listeners first
        for cb in self._global_listeners:
            try:
                cb(event)
            except Exception:
                pass  # Listener failures never propagate
        # Type-specific listeners
        for cb in self._listeners.get(event.type, []):
            try:
                cb(event)
            except Exception:
                pass

    def history(self, event_type: str = None) -> List[Event]:
        """Return event history, optionally filtered by type."""
        if event_type is None:
            return list(self._history)
        return [e for e in self._history if e.type == event_type]

    def clear(self) -> None:
        """Clear all listeners and history."""
        self._listeners = {t: [] for t in ALL_EVENTS}
        self._global_listeners.clear()
        self._history.clear()


# ── Built-in logging listener ──
def make_logging_listener(print_fn: Callable = print) -> Listener:
    """Create a listener that logs event summaries."""

    _emoji = {
        EVENT_AGENT_START: "🚀", EVENT_AGENT_END: "🏁",
        EVENT_TURN_START: "▶", EVENT_TURN_END: "◀",
        EVENT_TOOL_START: "🔧", EVENT_TOOL_END: "✅",
        EVENT_TOOL_UPDATE: "📊",
    }

    def _on_event(e: Event) -> None:
        emoji = _emoji.get(e.type, "•")
        if e.type == EVENT_TURN_START:
            print_fn(f"  {emoji} Turn {e.payload.get('iteration', '?')} | "
                     f"budget {e.payload.get('budget_remaining', '?')}s")
        elif e.type == EVENT_TOOL_START:
            print_fn(f"  {emoji} {e.payload.get('tool_name', '?')}: "
                     f"{e.payload.get('tool_args_summary', '')}")
        elif e.type == EVENT_TOOL_END:
            duration = e.payload.get("duration_ms", 0)
            print_fn(f"  {emoji} Done in {duration:.0f}ms | "
                     f"result_len={e.payload.get('result_len', 0)}")

    return _on_event
