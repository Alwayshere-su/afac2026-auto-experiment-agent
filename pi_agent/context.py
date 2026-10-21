"""
上下文压缩管线 — Pi-Agent Layer 2
==================================
当对话历史过长时自动压缩，保持在 LLM 上下文窗口限制内。

压缩策略：
  messages → [裁剪过长消息] → [提取摘要] → [修剪旧消息] → 压缩后的消息

保留头部（system + user_goal），压缩中间轮次为结构化摘要，保留尾部最近消息。

摘要格式（四段式）：
  1. 关键数据发现
  2. 有效方法（最优配置 + 分数）
  3. 失败记录（禁止重复）
  4. 当前优化方向
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


def _mechanical_summary(trajectory: List[Dict], experiments_completed: int,
                        recent_outputs: List[str]) -> str:
    """Simple mechanical summary — no LLM needed."""
    lines = [
        "1. Key data findings: see latest memory file",
        "2. Effective methods: see MEMORY.md index",
        f"3. Completed experiments: {experiments_completed}",
        "4. Current direction: incrementally improve from best known config",
    ]
    recent_thinking = []
    for t in trajectory[-8:]:
        content = str(t.get("agent_thinking", ""))[:200]
        if content:
            recent_thinking.append(f"   - {content}")
    if recent_thinking:
        lines.append("5. Recent thoughts:")
        lines.extend(recent_thinking[-3:])
    return "\n".join(lines)


def compress_messages(
    messages: List[Dict],
    trajectory: List[Dict],
    experiments_completed: int,
    compression_summary: str = "",
    print_fn: Callable = None,
    threshold: int = 3_500_000,
) -> List[Dict]:
    """
    Compress conversation context to prevent overflow.

    Structure after compression:
      [system][user_goal][compressed_summary]...[recent tail messages]

    Args:
        messages: full message list (system + user + assistant + tool)
        trajectory: recent trajectory entries for summary
        experiments_completed: count for summary
        compression_summary: previous summary text (carried forward)
        print_fn: optional logging
        threshold: char count that triggers compression
        llm_client: optional LLMClient for smarter summarization

    Returns:
        Compressed message list (may be unchanged if under threshold).
    """
    _print = print_fn or (lambda x: None)
    n = len(messages)

    if n <= 10:
        return messages

    # Only compress if total chars exceed threshold
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    if total_chars < threshold:
        return messages

    _print(f"\n  📦 Compressing context ({total_chars} → ...)")

    keep_recent = max(n // 2, 10)
    head = messages[:2]           # system + user_goal
    middle = messages[2:-keep_recent]
    tail = messages[-keep_recent:]

    # Truncate oversized individual messages in tail
    truncated_tail = []
    for m in tail:
        c = str(m.get("content", ""))
        if len(c) > 4000:
            m_copy = dict(m)
            m_copy["content"] = c[:4000] + f"\n...[truncated, was {len(c)} chars]"
            truncated_tail.append(m_copy)
        else:
            truncated_tail.append(m)

    # Strip leading tool messages from tail
    while truncated_tail and truncated_tail[0].get("role") == "tool":
        truncated_tail.pop(0)

    # Generate summary
    summary_text = _mechanical_summary(trajectory, experiments_completed, [])

    prev = f"Previous summary:\n{compression_summary}\n\n" if compression_summary else ""
    summary = {"role": "user", "content": f"[Context compressed — layered summary]\n{prev}{summary_text}"}

    result = head + [summary] + truncated_tail
    new_chars = sum(len(str(m.get("content", ""))) for m in result)
    _print(f"      → {new_chars} chars")

    return result
