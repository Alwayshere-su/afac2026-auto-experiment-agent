"""
消息类型系统 — Pi-Agent Layer 2
================================
定义 Agent 与 LLM 之间通信的 4 种消息类型，每种可转换为 LLM 兼容的 dict 格式：

  SystemMessage   → {"role": "system", "content": "..."}
  UserMessage     → {"role": "user", "content": "..."}
  AssistantMessage → {"role": "assistant", "content": "...", "tool_calls": [...]}
  ToolResult      → {"role": "tool", "tool_call_id": "...", "content": "..."}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """A single tool call within an assistant message."""
    id: str
    name: str
    arguments: str  # JSON string


# ── Message types ──

@dataclass
class SystemMessage:
    """System prompt — appears once at conversation start."""
    content: str
    role: str = field(default="system", init=False)

    def to_llm_dict(self) -> Dict[str, Any]:
        return {"role": "system", "content": self.content}


@dataclass
class UserMessage:
    """User message — instructions, feedback, dashboard, etc."""
    content: str
    role: str = field(default="user", init=False)

    def to_llm_dict(self) -> Dict[str, Any]:
        return {"role": "user", "content": self.content}


@dataclass
class AssistantMessage:
    """Assistant message — may contain thinking + tool calls."""
    content: str = ""                      # thinking text
    reasoning_content: str = ""            # raw reasoning from API
    tool_calls: List[ToolCall] = field(default_factory=list)
    role: str = field(default="assistant", init=False)

    @property
    def thinking(self) -> str:
        """Best available thinking text."""
        return self.content or self.reasoning_content

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def to_llm_dict(self) -> Dict[str, Any]:
        """Convert to LLM API format."""
        d: Dict[str, Any] = {
            "role": "assistant",
            "content": self.content or None,
        }
        if self.reasoning_content:
            d["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                }
                for tc in self.tool_calls
            ]
        return d


@dataclass
class ToolResultMessage:
    """Result of a tool execution."""
    tool_call_id: str
    tool_name: str
    content: str  # The result text
    role: str = field(default="tool", init=False)

    def to_llm_dict(self) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }
